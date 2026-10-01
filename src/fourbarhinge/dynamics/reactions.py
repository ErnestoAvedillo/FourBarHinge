
"""
Reacciones en las articulaciones por Newton-Euler (plano, ejes globales).

Incógnitas x = [R_A, R_B, R_C, R_D, λ]  (9):
  R_A: fuerza de la barra fija sobre INPUT en A
  R_B: fuerza de INPUT sobre COUPLER en B   (-R_B sobre INPUT)
  R_C: fuerza de COUPLER sobre OUTPUT en C  (-R_C sobre COUPLER)
  R_D: fuerza de la barra fija sobre OUTPUT en D
  λ:   magnitud del actuador (par o fuerza)

Ecuaciones (3 por barra móvil):
  Σ F = m·a_G      Σ M_G = I·α
"""

from dataclasses import dataclass
import numpy as np
from ..models.constants import BarId
from ..models.external_force import Actuator
from .dynamics import FourBarDynamics, AppliedLoad

MOVING_BARS = (BarId.INPUT, BarId.COUPLER, BarId.OUTPUT)

# (articulación, barra, signo) de cada reacción incógnita
JOINTS = {
    "A": ((BarId.INPUT, +1.0),),
    "B": ((BarId.COUPLER, +1.0), (BarId.INPUT, -1.0)),
    "C": ((BarId.OUTPUT, +1.0), (BarId.COUPLER, -1.0)),
    "D": ((BarId.OUTPUT, +1.0),),
}


@dataclass
class ReactionResult:
    reactions: dict[str, np.ndarray]   # fuerza en cada articulación (global)
    actuator_value: float              # λ obtenido por Newton-Euler
    residual: float                    # ‖A·x − b‖ (≈0 si es consistente)


def moment_row(r: np.ndarray) -> np.ndarray:
    """(r × F)_z = r_x·F_y − r_y·F_x como fila lineal en F."""
    return np.array([-r[1], r[0]])


class FourBarReactions:
    def __init__(self, dynamics: FourBarDynamics):
        self.dynamics = dynamics
        self.kinematics = dynamics.kinematics
        self.geometry = dynamics.geometry

    def get_com_global(self, bar_id: BarId) -> np.ndarray:
        bar = self.geometry.bar[bar_id]
        if bar.center_of_mass is None:
            raise ValueError(f"Center of mass not defined for {bar_id}")
        return self.kinematics.get_local_to_global(bar_id, bar.center_of_mass)

    def solve(self, actuator: Actuator | None = None) -> ReactionResult:
        """
        Requiere estado cinemático completo (theta, theta_dot, theta_ddot),
        normalmente tras FourBarDynamics.solve_theta_input_ddot() o
        solve_required_effort().
        Sin actuador, λ se fuerza a 0 y el sistema queda 9x8 (mínimos
        cuadrados); el residuo indica la consistencia con la dinámica.
        """
        joints = self.kinematics.get_joint_positions()
        theta_ddot = self.kinematics.theta_ddot.theta_ddot
        row = {bar: 3 * i for i, bar in enumerate(MOVING_BARS)}

        A_mat = np.zeros((9, 9))
        b = np.zeros(9)

        # Reacciones incógnitas
        for j, (name, contributions) in enumerate(JOINTS.items()):
            for bar_id, sign in contributions:
                r = joints[name] - self.get_com_global(bar_id)
                k = row[bar_id]
                A_mat[k:k + 2, 2 * j:2 * j + 2] += sign * np.eye(2)
                A_mat[k + 2, 2 * j:2 * j + 2] += sign * moment_row(r)

        # Actuador (columna λ)
        if actuator is not None:
            unit = self.dynamics.get_actuator_unit_load(actuator)
            self._add_load(A_mat[:, 8], unit, row, sign=1.0)

        # Inercia - gravedad - cargas conocidas
        known_loads: list[AppliedLoad] = (
            self.dynamics.get_spring_loads()
            + self.dynamics.get_external_loads()
        )
        for bar_id in MOVING_BARS:
            bar = self.geometry.bar[bar_id]
            if bar.mass is None or bar.inertia is None:
                raise ValueError(f"Mass/inertia not defined for {bar_id}")
            a_G = self.kinematics.get_point_acceleration(
                bar_id, bar.center_of_mass)
            k = row[bar_id]
            b[k:k + 2] = bar.mass * a_G - bar.mass * self.dynamics.gravity
            b[k + 2] = bar.inertia * theta_ddot[bar_id]
        for load in known_loads:
            self._add_load(b, load, row, sign=-1.0)

        if actuator is None:
            x, *_ = np.linalg.lstsq(A_mat[:, :8], b, rcond=None)
            x = np.append(x, 0.0)
        else:
            x = np.linalg.solve(A_mat, b)
        residual = float(np.linalg.norm(A_mat @ x - b))

        reactions = {
            name: x[2 * j:2 * j + 2] for j, name in enumerate(JOINTS)
        }
        return ReactionResult(reactions, float(x[8]), residual)

    def _add_load(
        self,
        target: np.ndarray,
        load: AppliedLoad,
        row: dict[BarId, int],
        sign: float,
    ) -> None:
        if load.bar == BarId.GROUND:
            return
        k = row[load.bar]
        r = load.point - self.get_com_global(load.bar)
        target[k:k + 2] += sign * load.force
        target[k + 2] += sign * (moment_row(r) @ load.force + load.torque)
