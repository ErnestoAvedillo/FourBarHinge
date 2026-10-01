
"""
Colocación y dimensionado de muelles para conseguir una curva
esfuerzo-recorrido objetivo en el punto P del acoplador.

Cada muelle se describe con rangos (low, high) para sus parámetros;
low == high fija el valor. Se ajusta por mínimos cuadrados con
multiarranque:  min Σ w·(F(s) - F_objetivo(s))²
"""

from dataclasses import dataclass, field
import numpy as np
from pydantic import BaseModel
from scipy.optimize import least_squares

from ..models.barra import Point, convert_2_angular
from ..models.constants import BarId
from ..models.position import Position, Attachment
from ..models.compression_spring import CompressionSpring
from ..models.torsion_spring import TorsionSpring
from .effort_travel import EffortTravelCurve

Range = tuple[float, float]


class CompressionSpringDesign(BaseModel):
    """Muelle de compresión entre bar_a y bar_b.

    Anclajes en coordenadas cartesianas locales de cada barra.
    """
    bar_a: BarId
    bar_b: BarId
    a_x: Range
    a_y: Range
    b_x: Range
    b_y: Range
    constant: Range
    free_length: Range

    def variables(self) -> dict[str, Range]:
        return {"a_x": self.a_x, "a_y": self.a_y, "b_x": self.b_x,
                "b_y": self.b_y, "constant": self.constant,
                "free_length": self.free_length}

    def build(self, values: dict[str, float]) -> CompressionSpring:
        def attachment(bar: BarId, x: float, y: float) -> Attachment:
            length, angle = convert_2_angular(Point(x=x, y=y))
            return Attachment(bar=bar, position=Position(length=length,
                                                          angle=angle))
        return CompressionSpring(
            free_length=values["free_length"],
            constant=values["constant"],
            a=attachment(self.bar_a, values["a_x"], values["a_y"]),
            b=attachment(self.bar_b, values["b_x"], values["b_y"]),
        )


class TorsionSpringDesign(BaseModel):
    """Muelle de torsión entre dos barras (normalmente en su bisagra)."""
    bar_a: BarId
    bar_b: BarId
    constant: Range
    free_angle: Range

    def variables(self) -> dict[str, Range]:
        return {"constant": self.constant, "free_angle": self.free_angle}

    def build(self, values: dict[str, float]) -> TorsionSpring:
        return TorsionSpring(free_angle=values["free_angle"],
                             constant=values["constant"],
                             bar_a=self.bar_a, bar_b=self.bar_b)


@dataclass
class SpringSynthesisResult:
    compression_springs: list[CompressionSpring]
    torsion_springs: list[TorsionSpring]
    travel: np.ndarray
    effort: np.ndarray
    target: np.ndarray
    rms_error: float
    max_error: float
    spring_ranges: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"error RMS = {self.rms_error:.4g}  "
                 f"máx = {self.max_error:.4g}"]
        lines += self.spring_ranges
        return "\n".join(lines)


class SpringSynthesis:
    def __init__(
        self,
        curve: EffortTravelCurve,
        target_travel: np.ndarray,
        target_effort: np.ndarray,
        compression_designs: list[CompressionSpringDesign] | None = None,
        torsion_designs: list[TorsionSpringDesign] | None = None,
        weights: np.ndarray | None = None,
        min_spring_length: float = 0.0,
    ):
        """
        target_travel / target_effort: curva objetivo (se interpola sobre
        el recorrido de `curve`; fuera de su rango no se evalúa).
        min_spring_length: longitud mínima admisible del muelle de
        compresión (p.ej. su longitud a bloque) en todo el recorrido.
        """
        self.curve = curve
        self.compression_designs = compression_designs or []
        self.torsion_designs = torsion_designs or []
        if not self.compression_designs and not self.torsion_designs:
            raise ValueError("At least one spring design is required")
        self.min_spring_length = min_spring_length

        travel = curve.travel
        inside = ((travel >= np.min(target_travel) - 1e-9)
                  & (travel <= np.max(target_travel) + 1e-9))
        self.mask = curve.valid & inside
        if self.mask.sum() < 2:
            raise ValueError("Target curve does not overlap the travel")
        order = np.argsort(target_travel)
        self.target = np.interp(travel, np.asarray(target_travel)[order],
                                np.asarray(target_effort)[order])
        self.weights = (np.ones_like(travel) if weights is None
                        else np.interp(travel, np.asarray(target_travel)[order],
                                       np.asarray(weights)[order]))
        self.scale = max(float(np.max(np.abs(self.target[self.mask]))), 1e-9)

        self.slots: list[tuple[int, str, Range]] = []
        self.fixed: dict[tuple[int, str], float] = {}
        designs = [*self.compression_designs, *self.torsion_designs]
        for index, design in enumerate(designs):
            for name, (low, high) in design.variables().items():
                if high < low:
                    raise ValueError(f"Invalid range for {name}")
                if np.isclose(low, high):
                    self.fixed[(index, name)] = low
                else:
                    self.slots.append((index, name, (low, high)))

    def build_springs(
        self, x: np.ndarray
    ) -> tuple[list[CompressionSpring], list[TorsionSpring]]:
        values: dict[int, dict[str, float]] = {}
        for (index, name), value in self.fixed.items():
            values.setdefault(index, {})[name] = value
        for (index, name, _), value in zip(self.slots, x):
            values.setdefault(index, {})[name] = float(value)
        n_comp = len(self.compression_designs)
        compression = [design.build(values[i])
                       for i, design in enumerate(self.compression_designs)]
        torsion = [design.build(values[n_comp + i])
                   for i, design in enumerate(self.torsion_designs)]
        return compression, torsion

    def residuals(self, x: np.ndarray) -> np.ndarray:
        compression, torsion = self.build_springs(x)
        effort = self.curve.effort(compression, torsion)
        error = (np.sqrt(self.weights) * (effort - self.target)
                 / self.scale)[self.mask]
        penalties = []
        for spring in compression:
            length = self.curve.compression_spring_length(spring)[self.mask]
            penalties.append(10.0 * np.maximum(
                self.min_spring_length - length, 0.0)
                / max(self.min_spring_length, 1.0))
        return np.concatenate([error, *penalties])

    def solve(self, n_starts: int = 30, seed: int | None = 0
              ) -> SpringSynthesisResult:
        rng = np.random.default_rng(seed)
        lower = np.array([r[0] for _, _, r in self.slots])
        upper = np.array([r[1] for _, _, r in self.slots])
        best_x, best_cost = None, np.inf
        for _ in range(max(n_starts, 1)):
            x0 = lower + rng.random(len(lower)) * (upper - lower)
            if len(x0) == 0:
                best_x = x0
                break
            solution = least_squares(self.residuals, x0,
                                     bounds=(lower, upper), max_nfev=500)
            if solution.cost < best_cost:
                best_x, best_cost = solution.x, solution.cost
        return self.build_result(best_x)

    def build_result(self, x: np.ndarray) -> SpringSynthesisResult:
        compression, torsion = self.build_springs(x)
        effort = self.curve.effort(compression, torsion)
        error = (effort - self.target)[self.mask]
        ranges = []
        for k, spring in enumerate(compression, start=1):
            length = self.curve.compression_spring_length(spring)[
                self.curve.valid]
            force = self.curve.compression_spring_force(spring)[
                self.curve.valid]
            ranges.append(
                f"compresión {k}: k = {spring.constant:.4g}, "
                f"L0 = {spring.free_length:.4g}, L = {length.min():.4g} … "
                f"{length.max():.4g}, F = {force.min():.4g} … "
                f"{force.max():.4g}")
        for k, spring in enumerate(torsion, start=1):
            deflection = self.curve.torsion_spring_deflection(spring)[
                self.curve.valid]
            ranges.append(
                f"torsión {k}: k = {spring.constant:.4g}, φ0 = "
                f"{np.rad2deg(spring.free_angle):.2f}°, deflexión = "
                f"{np.rad2deg(deflection.min()):.1f}° … "
                f"{np.rad2deg(deflection.max()):.1f}°")
        return SpringSynthesisResult(
            compression_springs=compression,
            torsion_springs=torsion,
            travel=self.curve.travel,
            effort=effort,
            target=np.where(self.mask, self.target, np.nan),
            rms_error=float(np.sqrt(np.mean(error**2))),
            max_error=float(np.max(np.abs(error))),
            spring_ranges=ranges,
        )
