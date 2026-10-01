
"""
Geometrical representation
coupler
     B ●────────────● C
      /              \
     /                \
   input             output
   /                    \
A ●──────────────────────● D
ground
"""

from ..models.barra import Bar, FourBarGeometry, Point
from ..models.constants import BarId
from math import cos, sin, atan2, sqrt, acos
from ..models.barra import FourBarConfiguration, FourBarVelocities, FourBarAccelerations
import numpy as np


def perpendicular(r: np.ndarray) -> np.ndarray:
    """k × r en el plano (giro de 90°)."""
    return np.array([-r[1], r[0]])


class FourBarKinematics:
    def __init__(self, geometry: FourBarGeometry, theta_ground: float):
        self.geometry = geometry
        self.configuration = FourBarConfiguration(
            theta={
                    BarId.GROUND: theta_ground,
                    BarId.INPUT: 0,
                    BarId.COUPLER: 0,
                    BarId.OUTPUT: 0
                   }
            )
        self.theta_dot = FourBarVelocities(
            theta_dot={
                       BarId.GROUND: 0,
                       BarId.INPUT: 0,
                       BarId.COUPLER: 0,
                       BarId.OUTPUT: 0
                       })
        self.theta_ddot = FourBarAccelerations(
            theta_ddot={
                        BarId.GROUND: 0,
                        BarId.INPUT: 0,
                        BarId.COUPLER: 0,
                        BarId.OUTPUT: 0
                        })

    def solve_configuration(
                            self,
                            theta_input: float,
                            branch: int = 1,
                            ) -> FourBarConfiguration:
        """
        Calculates the configuration of the 4 bar mechanism.
        Parameters:
            theta_input: Angle in radians of the input bar
            branch: configurantion can take only values 1 or 2.
        """
        # r = B - D (vector desde la articulación D hasta B)
        x = (self.geometry.bar[BarId.INPUT].length * cos(theta_input) -
             self.geometry.bar[BarId.GROUND].length)
        y = self.geometry.bar[BarId.INPUT].length * sin(theta_input)

        module_r_square = x ** 2 + y ** 2

        theta_r = atan2(y, x)

        alfa = ((self.geometry.bar[BarId.OUTPUT].length**2 +
                 module_r_square -
                 self.geometry.bar[BarId.COUPLER].length**2) /
                (2 * self.geometry.bar[BarId.OUTPUT].length * sqrt(module_r_square)))
        # protejo alfa por errores  numéricos.
        alfa = np.clip(alfa, -1.0, 1.0)
        theta_output = theta_r + branch * acos(alfa)

        # C - B = (D + Lo·e^(i·to)) - Li·e^(i·ti)
        vector_coupler_x = (self.geometry.bar[BarId.GROUND].length +
                            self.geometry.bar[BarId.OUTPUT].length * cos(theta_output) -
                            self.geometry.bar[BarId.INPUT].length * cos(theta_input))
        vector_coupler_y = (self.geometry.bar[BarId.OUTPUT].length * sin(theta_output) -
                            self.geometry.bar[BarId.INPUT].length * sin(theta_input))

        theta_coupler = atan2(vector_coupler_y, vector_coupler_x)

        self.configuration.theta[BarId.INPUT] = theta_input
        self.configuration.theta[BarId.COUPLER] = theta_coupler
        self.configuration.theta[BarId.OUTPUT] = theta_output
        return self.configuration

    def rotation_matrix(self, theta: float) -> np.ndarray:
        return np.array([
                         [cos(theta), -sin(theta)],
                         [sin(theta), cos(theta)],
                         ])

    def get_bar_origin_global(self, bar: BarId) -> np.ndarray:
        theta_g = self.configuration.theta[BarId.GROUND]
        Rg = self.rotation_matrix(theta_g)
        A = self.geometry.position_ground.to_array()

        if bar in (BarId.GROUND, BarId.INPUT):
            return A
        Li = self.geometry.bar[BarId.INPUT].length
        Lg = self.geometry.bar[BarId.GROUND].length
        ti = self.configuration.theta[BarId.INPUT]

        if bar == BarId.COUPLER:
            B_ground = np.array([
                Li * cos(ti),
                Li * sin(ti)
            ])
            return A + Rg @ B_ground

        if bar == BarId.OUTPUT:
            D_ground = np.array([Lg, 0.0])
            return A + Rg @ D_ground

        raise ValueError(f"Invalid bar: {bar}")

    def get_local_to_global(
        self,
        bar: BarId,
        point: Point
    ) -> np.ndarray:

        origin = self.get_bar_origin_global(bar)

        theta_g = self.configuration.theta[BarId.GROUND]

        if bar == BarId.GROUND:
            theta = theta_g
        else:
            theta = theta_g + self.configuration.theta[bar]

        R = self.rotation_matrix(theta)

        p_local = point.to_array()

        return origin + R @ p_local

    def get_global_to_local(
        self,
        bar: BarId,
        point: Point
    ) -> np.ndarray:

        origin = self.get_bar_origin_global(bar)

        theta_g = self.configuration.theta[BarId.GROUND]

        if bar == BarId.GROUND:
            theta = theta_g
        else:
            theta = theta_g + self.configuration.theta[bar]

        R = self.rotation_matrix(theta)

        p_global = point.to_array()

        return R.T @ (p_global - origin)

    def get_jacobian(self) -> np.ndarray:
        Li = self.geometry.bar[BarId.INPUT].length
        Lo = self.geometry.bar[BarId.OUTPUT].length
        Lc = self.geometry.bar[BarId.COUPLER].length

        ti = self.configuration.theta[BarId.INPUT]
        to = self.configuration.theta[BarId.OUTPUT]
        tc = self.configuration.theta[BarId.COUPLER]
        self.jacobian = np.array([
            [
                 -Li * sin(ti),
                 -Lc * sin(tc),
                 Lo * sin(to)
            ],
            [
                 Li * cos(ti),
                 Lc * cos(tc),
                 -Lo * cos(to)
            ],
            ])
        return self.jacobian

    def solve_theta_dot(self, theta_input_dot: float) -> np.ndarray:
        jacobian = self.get_jacobian()
        Ji = jacobian[:, 0]
        Jcp = jacobian[:, 1:]
        try:
            thetas_dot = np.linalg.solve(Jcp, -Ji * theta_input_dot)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                    "Error while solving theta_ddot with the Jacobian"
                    ) from exc
        self.theta_dot.theta_dot[BarId.INPUT] = theta_input_dot
        self.theta_dot.theta_dot[BarId.COUPLER] = thetas_dot[0]
        self.theta_dot.theta_dot[BarId.OUTPUT] = thetas_dot[1]

        return self.get_theta_vector_dot()

    def get_jacobian_dot(self) -> np.ndarray:
        Li = self.geometry.bar[BarId.INPUT].length
        Lo = self.geometry.bar[BarId.OUTPUT].length
        Lc = self.geometry.bar[BarId.COUPLER].length

        ti = self.configuration.theta[BarId.INPUT]
        to = self.configuration.theta[BarId.OUTPUT]
        tc = self.configuration.theta[BarId.COUPLER]

        ti_dot = self.theta_dot.theta_dot[BarId.INPUT]
        tc_dot = self.theta_dot.theta_dot[BarId.COUPLER]
        to_dot = self.theta_dot.theta_dot[BarId.OUTPUT]

        self.jacobian_dot = np.array([
            [
                 -Li * cos(ti) * ti_dot,
                 -Lc * cos(tc) * tc_dot,
                 Lo * cos(to) * to_dot,
            ],
            [
                 -Li * sin(ti) * ti_dot,
                 -Lc * sin(tc) * tc_dot,
                 Lo * sin(to) * to_dot
            ],
            ])
        return self.jacobian_dot

    def get_theta_vector_dot(self) -> np.ndarray:
        return self.theta_dot.theta_dot_to_array()

    def get_theta_vector_ddot(self) -> np.ndarray:
        return self.theta_ddot.theta_ddot_to_array()

    def solve_theta_ddot(self, theta_input_ddot: float) -> np.ndarray:
        jacobian = self.get_jacobian()
        Ji = jacobian[:, 0]
        Jcp = jacobian[:, 1:]
        equation = -Ji * theta_input_ddot - self.get_jacobian_dot() @ self.get_theta_vector_dot()
        try:
            thetas_ddot = np.linalg.solve(Jcp, equation)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                    "Error while solving theta_ddot with the Jacobian"
                    ) from exc
        self.theta_ddot.theta_ddot[BarId.INPUT] = theta_input_ddot
        self.theta_ddot.theta_ddot[BarId.COUPLER] = thetas_ddot[0]
        self.theta_ddot.theta_ddot[BarId.OUTPUT] = thetas_ddot[1]
        return self.get_theta_vector_ddot()

    def get_velocity_ratios(self) -> np.ndarray:
        """
        Γ = d[ti, tc, to]/d(ti): q̇ = Γ·ṫi. Reduce el sistema al único DOF.
        """
        jacobian = self.get_jacobian()
        Ji = jacobian[:, 0]
        Jcp = jacobian[:, 1:]
        ratios = np.linalg.solve(Jcp, -Ji)
        return np.array([1.0, ratios[0], ratios[1]])

    def get_velocity_product_acceleration(self) -> np.ndarray:
        """
        γ = Γ̇·ṫi: aceleraciones [ẗi, ẗc, ẗo] con ẗi = 0 (término centrípeto
        de la cadena cerrada). q̈ = Γ·ẗi + γ.
        """
        jacobian = self.get_jacobian()
        Jcp = jacobian[:, 1:]
        equation = -self.get_jacobian_dot() @ self.get_theta_vector_dot()
        thetas_ddot = np.linalg.solve(Jcp, equation)
        return np.array([0.0, thetas_ddot[0], thetas_ddot[1]])

    def get_point_jacobian(self, bar_id: BarId, point_local: Point) -> np.ndarray:
        """
        Matriz 2x3 J_p tal que v_p (global) = J_p @ [ṫi, ṫc, ṫo].
        """
        jacobian = np.zeros((2, 3))
        if bar_id == BarId.GROUND:
            return jacobian
        p = self.get_local_to_global(bar_id, point_local)
        if bar_id == BarId.INPUT:
            jacobian[:, 0] = perpendicular(p - self.get_bar_origin_global(BarId.INPUT))
        elif bar_id == BarId.COUPLER:
            B = self.get_bar_origin_global(BarId.COUPLER)
            jacobian[:, 0] = perpendicular(B - self.get_bar_origin_global(BarId.INPUT))
            jacobian[:, 1] = perpendicular(p - B)
        elif bar_id == BarId.OUTPUT:
            jacobian[:, 2] = perpendicular(p - self.get_bar_origin_global(BarId.OUTPUT))
        return jacobian

    def get_point_acceleration(self, bar_id: BarId, point_local: Point) -> np.ndarray:
        """
        Aceleración global de un punto de la barra: a = J_p·q̈ + J̇_p·q̇.
        Requiere haber resuelto theta_dot y theta_ddot.
        """
        if bar_id == BarId.GROUND:
            return np.zeros(2)
        theta_dot = self.theta_dot.theta_dot
        theta_ddot = self.theta_ddot.theta_ddot

        def rigid(r: np.ndarray, omega: float, alpha: float) -> np.ndarray:
            return alpha * perpendicular(r) - omega**2 * r

        p = self.get_local_to_global(bar_id, point_local)
        origin = self.get_bar_origin_global(bar_id)
        a_relative = rigid(p - origin, theta_dot[bar_id], theta_ddot[bar_id])
        if bar_id in (BarId.INPUT, BarId.OUTPUT):
            return a_relative
        A = self.get_bar_origin_global(BarId.INPUT)
        a_B = rigid(origin - A, theta_dot[BarId.INPUT], theta_ddot[BarId.INPUT])
        return a_B + a_relative

    def get_closure_error(self) -> float:
        """
        |‖C - D‖ - Lo|. ≠ 0 si theta_input está fuera del rango montable
        (solve_configuration recorta acos y devuelve una posición falsa).
        """
        joints = self.get_joint_positions()
        Lo = self.geometry.bar[BarId.OUTPUT].length
        return abs(float(np.linalg.norm(joints["C"] - joints["D"])) - Lo)

    def get_joint_positions(self) -> dict[str, np.ndarray]:
        """Posiciones globales de las articulaciones A, B, C, D."""
        Lc = self.geometry.bar[BarId.COUPLER].length
        return {
            "A": self.get_bar_origin_global(BarId.INPUT),
            "B": self.get_bar_origin_global(BarId.COUPLER),
            "C": self.get_local_to_global(BarId.COUPLER, Point(x=Lc, y=0.0)),
            "D": self.get_bar_origin_global(BarId.OUTPUT),
        }

    def get_point_velocity(self, bar_id: BarId, point_local: Point) -> np.ndarray:
        if bar_id == BarId.GROUND:
            return np.zeros(2)
        theta_ground = self.configuration.theta[BarId.GROUND]
        theta = theta_ground + self.configuration.theta[bar_id]
        theta_dot = self.theta_dot.theta_dot[bar_id]
        R = self.rotation_matrix(theta)
        r_global = R @ point_local.to_array()
        v_relative = theta_dot * np.array([-r_global[1], r_global[0]])
        if bar_id in (BarId.INPUT, BarId.OUTPUT):
            return v_relative

        if bar_id == BarId.COUPLER:
            Li = self.geometry.bar[BarId.INPUT].length
            ti = self.configuration.theta[BarId.INPUT]
            ti_dot = self.theta_dot.theta_dot[BarId.INPUT]

            R_input = self.rotation_matrix(theta_ground + ti)

            r_AB = R_input @ np.array([Li, 0.0, ])

            v_B = ti_dot * np.array([-r_AB[1], r_AB[0], ])

            return v_B + v_relative

        raise ValueError(f"Invalid bar: {bar_id}")
