
"""
Síntesis dimensional por generación de trayectoria.

Dado un recorrido (puntos objetivo, ejes globales) de un punto P del
acoplador, busca las longitudes de las barras, los pivotes fijos y la
posición de P en el acoplador para que P pase por esos puntos.

Incógnitas (las fijadas por el usuario se eliminan):
    Ax, Ay, θg, Lg      barra fija (pivote A, orientación, longitud A-D)
    Li, Lc, Lo          entrada, acoplador, salida
    u, v                P en el marco local del acoplador (origen B, x -> C)
    ti_1, Δti_j         ángulo de entrada en cada punto (monótono)

Residuos (mínimos cuadrados, scipy.optimize.least_squares):
    P(ti_j) - P_objetivo_j
    penalización si el mecanismo no puede montarse en ti_j o entre puntos
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from pydantic import BaseModel

from ..models.barra import Bar, Point, FourBarGeometry
from ..models.constants import BarId
from ..kinematics.kinematics import FourBarKinematics

PARAMETERS = ("ax", "ay", "theta_ground", "lg", "li", "lc", "lo", "u", "v")


class LinkBounds(BaseModel):
    """Límites de diseño de la síntesis."""
    min_length: float
    max_length: float
    # Paso máximo de ángulo de entrada entre puntos consecutivos [rad]
    max_input_step: float = np.pi / 2
    # Ángulo de transmisión mínimo admisible entre acoplador y salida [rad]
    min_transmission_angle: float = np.deg2rad(20.0)


def coupler_point_positions(
    params: dict[str, float], theta_input: np.ndarray, branch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Versión vectorizada de FourBarKinematics.solve_configuration.
    Devuelve (P global (n,2), tc, to, alfa) — |alfa| > 1: no montable.
    """
    li, lc, lo, lg = params["li"], params["lc"], params["lo"], params["lg"]
    ti = np.asarray(theta_input, dtype=float)
    x = li * np.cos(ti) - lg
    y = li * np.sin(ti)
    r = np.sqrt(x**2 + y**2)
    alfa = (lo**2 + r**2 - lc**2) / (2.0 * lo * np.maximum(r, 1e-12))
    to = np.arctan2(y, x) + branch * np.arccos(np.clip(alfa, -1.0, 1.0))
    tc = np.arctan2(lo * np.sin(to) - li * np.sin(ti),
                    lg + lo * np.cos(to) - li * np.cos(ti))
    b = np.stack([li * np.cos(ti), li * np.sin(ti)], axis=1)
    u, v = params["u"], params["v"]
    p_local = b + np.stack([u * np.cos(tc) - v * np.sin(tc),
                            u * np.sin(tc) + v * np.cos(tc)], axis=1)
    cg, sg = np.cos(params["theta_ground"]), np.sin(params["theta_ground"])
    p_global = np.stack([
        params["ax"] + cg * p_local[:, 0] - sg * p_local[:, 1],
        params["ay"] + sg * p_local[:, 0] + cg * p_local[:, 1],
    ], axis=1)
    return p_global, tc, to, alfa


def transmission_angle(tc: np.ndarray, to: np.ndarray) -> np.ndarray:
    """Ángulo entre acoplador y salida, en [0, π/2] (90° es ideal)."""
    mu = np.abs(np.angle(np.exp(1j * (to - tc))))
    return np.minimum(mu, np.pi - mu)


@dataclass
class PathSynthesisResult:
    params: dict[str, float]
    branch: int
    theta_inputs: np.ndarray          # ángulo de entrada en cada objetivo
    target_points: np.ndarray
    achieved_points: np.ndarray
    rms_error: float
    max_error: float
    feasible: bool                    # montable en todo el recorrido
    min_transmission_angle: float     # en todo el recorrido [rad]

    @property
    def coupler_point(self) -> Point:
        """P en coordenadas locales del acoplador."""
        return Point(x=self.params["u"], y=self.params["v"])

    @property
    def theta_ground(self) -> float:
        return self.params["theta_ground"]

    @property
    def theta_start(self) -> float:
        return float(self.theta_inputs[0])

    @property
    def theta_end(self) -> float:
        return float(self.theta_inputs[-1])

    def to_geometry(
        self, linear_density: float, coupler_mass: float | None = None,
        coupler_center_of_mass: Point | None = None,
        coupler_inertia: float | None = None,
    ) -> FourBarGeometry:
        """
        Geometría con barras uniformes de densidad lineal dada (kg/longitud).
        El acoplador (normalmente la puerta) admite masa/COM/inercia propios.
        """
        p = self.params

        def uniform(length: float) -> Bar:
            mass = linear_density * length
            return Bar(length=length, mass=mass,
                       inertia=mass * length**2 / 12.0,
                       center_of_mass=Point(x=length / 2.0, y=0.0))

        coupler = uniform(p["lc"])
        if coupler_mass is not None:
            coupler = Bar(
                length=p["lc"], mass=coupler_mass,
                inertia=(coupler_inertia if coupler_inertia is not None
                         else coupler_mass * p["lc"]**2 / 12.0),
                center_of_mass=(coupler_center_of_mass
                                or Point(x=p["lc"] / 2.0, y=0.0)))
        return FourBarGeometry(
            position_ground=Point(x=p["ax"], y=p["ay"]),
            bar={
                BarId.GROUND: uniform(p["lg"]),
                BarId.INPUT: uniform(p["li"]),
                BarId.COUPLER: coupler,
                BarId.OUTPUT: uniform(p["lo"]),
            },
        )

    def to_kinematics(self, geometry: FourBarGeometry) -> FourBarKinematics:
        return FourBarKinematics(geometry=geometry,
                                 theta_ground=self.theta_ground)

    def path(self, n: int = 200) -> np.ndarray:
        """Trayectoria continua de P entre el primer y último objetivo."""
        ti = np.linspace(self.theta_start, self.theta_end, n)
        return coupler_point_positions(self.params, ti, self.branch)[0]

    def summary(self) -> str:
        p = self.params
        return (
            f"A = ({p['ax']:.3f}, {p['ay']:.3f})  θg = "
            f"{np.rad2deg(p['theta_ground']):.2f}°\n"
            f"Lg = {p['lg']:.3f}  Li = {p['li']:.3f}  Lc = {p['lc']:.3f}  "
            f"Lo = {p['lo']:.3f}\n"
            f"P en acoplador = ({p['u']:.3f}, {p['v']:.3f})  "
            f"rama = {self.branch:+d}\n"
            f"θ entrada: {np.rad2deg(self.theta_start):.2f}° -> "
            f"{np.rad2deg(self.theta_end):.2f}°\n"
            f"error RMS = {self.rms_error:.4g}  máx = {self.max_error:.4g}  "
            f"montable = {self.feasible}  μ_min = "
            f"{np.rad2deg(self.min_transmission_angle):.1f}°"
        )


class CouplerPathSynthesis:
    """
    Busca la geometría del cuadrilátero cuyo punto de acoplador pasa por
    `target_points` (en orden de recorrido).

    Fijar lo que se conozca reduce incógnitas y mejora el resultado:
      ground_pivot_a / ground_pivot_d: pivotes fijos A y D (global).
      theta_inputs: ángulos de entrada en cada punto (sincronización).
    Se necesitan ≥ 5 puntos para que el problema esté bien determinado;
    con menos hay infinitas soluciones y se devuelve una de ellas.
    """

    def __init__(
        self,
        target_points: np.ndarray,
        bounds: LinkBounds,
        branches: tuple[int, ...] = (1, -1),
        ground_pivot_a: Point | None = None,
        ground_pivot_d: Point | None = None,
        theta_inputs: np.ndarray | None = None,
        midpoints: int = 4,
    ):
        self.targets = np.asarray(target_points, dtype=float)
        if self.targets.ndim != 2 or self.targets.shape[1] != 2:
            raise ValueError("target_points must have shape (n, 2)")
        if len(self.targets) < 2:
            raise ValueError("At least 2 target points are required")
        self.bounds = bounds
        self.branches = branches
        self.theta_inputs = (None if theta_inputs is None
                             else np.asarray(theta_inputs, dtype=float))
        self.midpoints = midpoints
        self.fixed: dict[str, float] = {}
        if ground_pivot_a is not None:
            self.fixed["ax"] = ground_pivot_a.x
            self.fixed["ay"] = ground_pivot_a.y
        if ground_pivot_a is not None and ground_pivot_d is not None:
            d = ground_pivot_d.to_array() - ground_pivot_a.to_array()
            self.fixed["theta_ground"] = float(np.arctan2(d[1], d[0]))
            self.fixed["lg"] = float(np.linalg.norm(d))
        self.free = [k for k in PARAMETERS if k not in self.fixed]
        self.n_points = len(self.targets)
        self.scale = max(float(np.ptp(self.targets, axis=0).max()),
                         bounds.min_length)

    # ------------------------------------------------------------------

    def parameter_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        b = self.bounds
        center = self.targets.mean(axis=0)
        span = 2.0 * b.max_length
        limits = {
            "ax": (center[0] - span, center[0] + span),
            "ay": (center[1] - span, center[1] + span),
            "theta_ground": (-np.pi, np.pi),
            "lg": (b.min_length, b.max_length),
            "li": (b.min_length, b.max_length),
            "lc": (b.min_length, b.max_length),
            "lo": (b.min_length, b.max_length),
            "u": (-b.max_length, b.max_length),
            "v": (-b.max_length, b.max_length),
        }
        lower = [limits[k][0] for k in self.free]
        upper = [limits[k][1] for k in self.free]
        return np.array(lower), np.array(upper)

    def unpack(
        self, x: np.ndarray, direction: int
    ) -> tuple[dict[str, float], np.ndarray]:
        params = dict(self.fixed)
        params.update(zip(self.free, x[:len(self.free)]))
        if self.theta_inputs is not None:
            return params, self.theta_inputs
        rest = x[len(self.free):]
        steps = direction * rest[1:]
        theta = rest[0] + np.concatenate(([0.0], np.cumsum(steps)))
        return params, theta

    def sample_angles(self, theta: np.ndarray) -> np.ndarray:
        """Ángulos intermedios entre objetivos para vigilar el montaje."""
        t = np.linspace(0.0, 1.0, self.midpoints + 2)[1:-1]
        mids = theta[:-1, None] + t[None, :] * np.diff(theta)[:, None]
        return np.concatenate((theta, mids.ravel()))

    def residuals(
        self, x: np.ndarray, branch: int, direction: int
    ) -> np.ndarray:
        params, theta = self.unpack(x, direction)
        points, _, _, _ = coupler_point_positions(params, theta, branch)
        position_error = (points - self.targets).ravel() / self.scale
        _, tc, to, alfa = coupler_point_positions(
            params, self.sample_angles(theta), branch)
        assembly = 10.0 * np.maximum(np.abs(alfa) - 1.0, 0.0)
        mu = transmission_angle(tc, to)
        transmission = np.maximum(
            self.bounds.min_transmission_angle - mu, 0.0)
        return np.concatenate((position_error, assembly, transmission))

    # ------------------------------------------------------------------

    def solve(
        self, n_starts: int = 60, seed: int | None = 0,
        tolerance: float = 1e-10,
    ) -> PathSynthesisResult:
        """Multiarranque aleatorio; devuelve la mejor solución montable."""
        rng = np.random.default_rng(seed)
        lower, upper = self.parameter_bounds()
        if self.theta_inputs is None:
            step = self.bounds.max_input_step
            lower = np.concatenate((lower, [-np.pi], np.zeros(
                self.n_points - 1)))
            upper = np.concatenate((upper, [np.pi], np.full(
                self.n_points - 1, step)))
            directions: tuple[int, ...] = (1, -1)
        else:
            directions = (1,)

        best: PathSynthesisResult | None = None
        best_key = (False, np.inf)
        for start in range(n_starts):
            branch = self.branches[start % len(self.branches)]
            direction = directions[(start // len(self.branches))
                                   % len(directions)]
            x0 = lower + rng.random(len(lower)) * (upper - lower)
            try:
                solution = least_squares(
                    self.residuals, x0, bounds=(lower, upper),
                    args=(branch, direction), xtol=tolerance,
                    ftol=tolerance, max_nfev=2000)
            except ValueError:
                continue
            result = self.build_result(solution.x, branch, direction)
            key = (result.feasible, -result.rms_error)
            if best is None or key > best_key:
                best, best_key = result, key
        if best is None:
            raise RuntimeError("Synthesis failed for every starting point")
        return best

    def build_result(
        self, x: np.ndarray, branch: int, direction: int
    ) -> PathSynthesisResult:
        params, theta = self.unpack(x, direction)
        points, _, _, _ = coupler_point_positions(params, theta, branch)
        dense = np.linspace(theta[0], theta[-1], 400)
        _, tc, to, alfa = coupler_point_positions(params, dense, branch)
        mu = transmission_angle(tc, to)
        errors = np.linalg.norm(points - self.targets, axis=1)
        feasible = bool(np.all(np.abs(alfa) <= 1.0)
                        and mu.min() >= self.bounds.min_transmission_angle
                        - 1e-6)
        return PathSynthesisResult(
            params={k: float(v) for k, v in params.items()},
            branch=branch,
            theta_inputs=np.asarray(theta, dtype=float),
            target_points=self.targets,
            achieved_points=points,
            rms_error=float(np.sqrt(np.mean(errors**2))),
            max_error=float(errors.max()),
            feasible=feasible,
            min_transmission_angle=float(mu.min()),
        )
