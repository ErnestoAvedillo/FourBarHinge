
"""
Curva esfuerzo-recorrido en un punto P (normalmente del acoplador).

Esfuerzo F: fuerza estática en P necesaria para mantener el equilibrio
con muelles, gravedad y cargas externas (ṫi = 0, ẗi = 0). Dos modos:

Fuerza tangente (por defecto):
    s: longitud de arco recorrida por P desde theta_start.
    F: tangente a la trayectoria de P, en el sentido del recorrido.
       F > 0: hay que empujar en el sentido del recorrido.
       F < 0: el mecanismo avanza solo (hay que frenarlo).

Fuerza apoyada en un punto fijo S (force_origin, p.ej. un bulón):
    s: carrera = |SP| - |SP|₀ (> 0 alarga, < 0 acorta la línea S-P).
    F: en la línea S -> P.
       F > 0: empuja (aleja P de S; cilindro/amortiguador a compresión).
       F < 0: tira (acerca P a S).
    El trabajo sigue siendo ∫F·ds.

Para poder optimizar la colocación de muelles, la cinemática de cada
muestra se calcula una sola vez y la contribución de los muelles se
evalúa vectorizada con numpy.
"""

from dataclasses import dataclass
import numpy as np

from ..dynamics.dynamics import FourBarDynamics
from ..models.barra import Point
from ..models.constants import BarId
from ..models.external_force import Actuator
from ..models.compression_spring import CompressionSpring
from ..models.torsion_spring import TorsionSpring


def perpendicular(r: np.ndarray) -> np.ndarray:
    """k × r para arrays (..., 2)."""
    return np.stack([-r[..., 1], r[..., 0]], axis=-1)


@dataclass
class KinematicTable:
    """Estado cinemático precalculado en cada muestra (arrays por fila)."""
    theta_input: np.ndarray        # (n,)
    angle: np.ndarray              # (n, 4) ángulo absoluto global de cada barra
    origin: np.ndarray             # (n, 4, 2) origen global de cada barra
    gamma: np.ndarray              # (n, 3) Γ = d[ti, tc, to]/dti
    joint_a: np.ndarray            # (n, 2)
    joint_b: np.ndarray            # (n, 2)
    joint_d: np.ndarray            # (n, 2)

    def to_global(self, bar: BarId, local: np.ndarray) -> np.ndarray:
        """Punto local de una barra -> global en todas las muestras."""
        c = np.cos(self.angle[:, bar])
        s = np.sin(self.angle[:, bar])
        x, y = local
        return self.origin[:, bar] + np.stack([c * x - s * y, s * x + c * y],
                                              axis=1)

    def point_rate(self, bar: BarId, point: np.ndarray) -> np.ndarray:
        """dp/dti (n,2) de un punto de `bar` dado en global (n,2)."""
        if bar == BarId.GROUND:
            return np.zeros_like(point)
        g = self.gamma
        if bar == BarId.INPUT:
            return g[:, :1] * perpendicular(point - self.joint_a)
        if bar == BarId.COUPLER:
            return (g[:, :1] * perpendicular(self.joint_b - self.joint_a)
                    + g[:, 1:2] * perpendicular(point - self.joint_b))
        return g[:, 2:3] * perpendicular(point - self.joint_d)

    def reduced_force(self, bar: BarId, point: np.ndarray,
                      force: np.ndarray) -> np.ndarray:
        """Γᵀ·J_pᵀ·F para fuerza F (n,2) aplicada en point (n,2) global."""
        # Q = F·dp/dti
        return np.einsum("ij,ij->i", force, self.point_rate(bar, point))

    def angle_rate(self, bar: BarId) -> np.ndarray:
        """dθ_bar/dti (0 para la barra fija)."""
        if bar == BarId.GROUND:
            return np.zeros(len(self.theta_input))
        return self.gamma[:, int(bar) - 1]


class EffortTravelCurve:
    def __init__(
        self,
        dynamics: FourBarDynamics,
        coupler_point: Point,
        theta_start: float,
        theta_end: float,
        branch: int = 1,
        samples: int = 200,
        force_origin: Point | None = None,
        bar: BarId = BarId.COUPLER,
    ):
        """
        coupler_point: punto P de aplicación, en coordenadas locales de
            `bar` (por defecto el acoplador).
        force_origin: punto fijo S (global) donde se apoya la fuerza, p.ej.
            el bulón de un amortiguador o cilindro. Si se da, la fuerza
            actúa en la línea S -> P y el recorrido es la carrera |SP|-|SP|₀.
        """
        if bar == BarId.GROUND:
            raise ValueError("The force must act on a moving bar")
        self.dynamics = dynamics
        self.kinematics = dynamics.kinematics
        self.coupler_point = coupler_point
        self.bar = bar
        self.force_origin = force_origin
        self.branch = branch
        self.direction = float(np.sign(theta_end - theta_start)) or 1.0
        theta = np.linspace(theta_start, theta_end, samples)
        self.table, self.valid = self.build_table(theta)
        self.point = self.table.to_global(bar, coupler_point.to_array())
        self.arc_length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(
            np.diff(self.point, axis=0), axis=1))))
        rate = self.table.point_rate(bar, self.point)
        if force_origin is None:
            # Fuerza unitaria tangente en P -> fuerza generalizada
            self.force_direction = (self.direction * rate / np.maximum(
                np.linalg.norm(rate, axis=1), 1e-300)[:, None])
            self.travel = self.arc_length
        else:
            line = self.point - force_origin.to_array()
            self.force_length = np.linalg.norm(line, axis=1)
            if np.any(self.force_length[self.valid] < 1e-12):
                raise ValueError("force_origin coincides with the force point")
            self.force_direction = line / np.maximum(
                self.force_length, 1e-300)[:, None]
            first = int(np.argmax(self.valid))
            self.travel = self.force_length - self.force_length[first]
        self.actuator_q = np.einsum("ij,ij->i", self.force_direction, rate)
        self.valid &= np.abs(self.actuator_q) > 1e-12
        self.base = self.compute_base_terms()

    @property
    def theta_input(self) -> np.ndarray:
        return self.table.theta_input

    def actuator(self, index: int) -> Actuator:
        """Actuador equivalente (dirección global fija) en la muestra index,
        para comprobar con FourBarDynamics.solve_required_effort."""
        return Actuator(bar=self.bar, point=self.coupler_point,
                        direction=tuple(self.force_direction[index]))

    def build_table(
        self, theta: np.ndarray
    ) -> tuple[KinematicTable, np.ndarray]:
        n = len(theta)
        angle = np.zeros((n, 4))
        origin = np.zeros((n, 4, 2))
        gamma = np.zeros((n, 3))
        valid = np.ones(n, dtype=bool)
        kin = self.kinematics
        for i, ti in enumerate(theta):
            kin.solve_configuration(ti, self.branch)
            if kin.get_closure_error() > 1e-6:
                valid[i] = False
                continue
            try:
                gamma[i] = kin.get_velocity_ratios()
            except np.linalg.LinAlgError:
                valid[i] = False
                continue
            for bar in BarId:
                angle[i, bar] = self.dynamics.get_absolute_angle(bar)
                origin[i, bar] = kin.get_bar_origin_global(bar)
        table = KinematicTable(
            theta_input=theta, angle=angle, origin=origin, gamma=gamma,
            joint_a=origin[:, BarId.INPUT], joint_b=origin[:, BarId.COUPLER],
            joint_d=origin[:, BarId.OUTPUT])
        return table, valid

    def compute_base_terms(self) -> np.ndarray:
        """G_red - Q_ext en cada muestra (no dependen de los muelles)."""
        base = np.full(len(self.theta_input), np.nan)
        kin = self.kinematics
        for i, ti in enumerate(self.theta_input):
            if not self.valid[i]:
                continue
            kin.solve_configuration(ti, self.branch)
            kin.solve_theta_dot(0.0)
            base[i] = (self.dynamics.get_reduced_gravity()
                       - self.dynamics.get_reduced_external_force())
        return base

    # ------------------------------------------------------------------
    # Muelles (vectorizado)
    # ------------------------------------------------------------------

    def compression_attachments(
        self, spring: CompressionSpring
    ) -> tuple[np.ndarray, np.ndarray]:
        p_a = self.table.to_global(BarId(spring.a.bar),
                                   spring.a.position.to_array())
        p_b = self.table.to_global(BarId(spring.b.bar),
                                   spring.b.position.to_array())
        return p_a, p_b

    def compression_spring_length(
        self, spring: CompressionSpring
    ) -> np.ndarray:
        p_a, p_b = self.compression_attachments(spring)
        return np.linalg.norm(p_b - p_a, axis=1)

    def compression_spring_force(
        self, spring: CompressionSpring
    ) -> np.ndarray:
        """Fuerza del muelle (≥ 0, sólo compresión) en cada muestra."""
        length = self.compression_spring_length(spring)
        return spring.constant * np.maximum(spring.free_length - length, 0.0)

    def torsion_spring_deflection(self, spring: TorsionSpring) -> np.ndarray:
        phi = (self.table.angle[:, spring.bar_b]
               - self.table.angle[:, spring.bar_a])
        return np.angle(np.exp(1j * (phi - spring.free_angle)))

    def spring_reduced_force(
        self,
        compression_springs: list[CompressionSpring],
        torsion_springs: list[TorsionSpring],
    ) -> np.ndarray:
        q = np.zeros(len(self.theta_input))
        for spring in compression_springs:
            p_a, p_b = self.compression_attachments(spring)
            delta = p_b - p_a
            length = np.maximum(np.linalg.norm(delta, axis=1), 1e-12)
            u = delta / length[:, None]
            magnitude = spring.constant * np.maximum(
                spring.free_length - length, 0.0)
            force_b = magnitude[:, None] * u
            q += self.table.reduced_force(BarId(spring.b.bar), p_b, force_b)
            q += self.table.reduced_force(BarId(spring.a.bar), p_a, -force_b)
        for spring in torsion_springs:
            torque_b = -spring.constant * self.torsion_spring_deflection(
                spring)
            q += torque_b * (self.table.angle_rate(BarId(spring.bar_b))
                             - self.table.angle_rate(BarId(spring.bar_a)))
        return q

    # ------------------------------------------------------------------
    # Esfuerzo
    # ------------------------------------------------------------------

    def effort(
        self,
        compression_springs: list[CompressionSpring] | None = None,
        torsion_springs: list[TorsionSpring] | None = None,
    ) -> np.ndarray:
        """
        Esfuerzo tangente en P [fuerza]. Sin argumentos usa los muelles
        definidos en el FourBarDynamics. NaN donde no es montable.
        """
        if compression_springs is None and torsion_springs is None:
            compression_springs = self.dynamics.compression_springs
            torsion_springs = self.dynamics.torsion_springs
        q = self.spring_reduced_force(compression_springs or [],
                                      torsion_springs or [])
        effort = (self.base - q) / np.where(self.valid, self.actuator_q, 1.0)
        return np.where(self.valid, effort, np.nan)

    def work(self, effort: np.ndarray) -> float:
        """Trabajo necesario a lo largo del recorrido ∫F·ds."""
        ok = np.isfinite(effort)
        return float(np.trapezoid(effort[ok], self.travel[ok]))
