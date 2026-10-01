
from dataclasses import dataclass
from ..kinematics.kinematics import FourBarKinematics
import numpy as np
from ..models.constants import BarId
import modern_robotics as mr
from ..models.barra import Point
from ..models.compression_spring import CompressionSpring
from ..models.torsion_spring import TorsionSpring
from ..models.external_force import ExternalForce, Actuator


@dataclass
class AppliedLoad:
    """Carga conocida sobre una barra, en ejes globales."""
    bar: BarId
    point: np.ndarray       # punto de aplicación (global)
    force: np.ndarray       # (Fx, Fy) global
    torque: float = 0.0     # par puro (z)


class FourBarDynamics():
    """
    Dinámica del cuadrilátero articulado con un único DOF (ti).

    Coordenadas:
      q    = [ti, tc, to]     ángulos absolutos (marco de la barra fija)
      θ_mr = T·q + cte        ángulos relativos de la cadena abierta MR
      q̇    = Γ·ṫi,  q̈ = Γ·ẗi + γ

    Ecuación reducida (Γᵀ elimina la reacción de cierre en D):
      M_red·ẗi + C_red + G_red = Q_springs + Q_ext + λ·Q_act
    """

    def __init__(
        self,
        four_bar_params: FourBarKinematics,
        compression_springs: list[CompressionSpring] | None = None,
        torsion_springs: list[TorsionSpring] | None = None,
        external_forces: list[ExternalForce] | None = None,
        gravity: tuple[float, float] = (0.0, -9.81),
    ):
        self.kinematics = four_bar_params
        self.geometry = self.kinematics.geometry
        self.compression_springs = compression_springs or []
        self.torsion_springs = torsion_springs or []
        self.external_forces = external_forces or []
        # Gravedad en ejes globales (unidades coherentes con longitudes).
        self.gravity = np.array(gravity, dtype=float)

    @staticmethod
    def get_mr_coordinate_matrix() -> np.ndarray:
        """T: θ̇_mr = T·[ṫi, ṫc, ṫo]."""
        return np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
        ])

    @staticmethod
    def translation_transform(position: np.ndarray) -> np.ndarray:
        M = np.eye(4)
        M[:3, 3] = position
        return M

    def get_glist(self) -> list[np.ndarray]:
        Glist = [
            self.kinematics.geometry.bar[BarId.INPUT].get_spatial_inertia(),
            self.kinematics.geometry.bar[BarId.COUPLER].get_spatial_inertia(),
            self.kinematics.geometry.bar[BarId.OUTPUT].get_spatial_inertia(),
        ]
        return Glist

    def get_slist(self) -> np.ndarray:
        Li = self.geometry.bar[BarId.INPUT].length
        Lc = self.geometry.bar[BarId.COUPLER].length

        return np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, -Li, -(Li + Lc)],
            [0.0, 0.0, 0.0],
        ])

    def get_mr_thetalist(self) -> np.ndarray:
        theta = self.kinematics.configuration.theta
        ti = theta[BarId.INPUT]
        tc = theta[BarId.COUPLER]
        to = theta[BarId.OUTPUT]

        return np.array([
            ti,
            tc - ti,
            to + np.pi - tc,
            ])

    def get_mr_dthetalist(self) -> np.ndarray:
        theta_dot = self.kinematics.theta_dot.theta_dot

        ti_dot = theta_dot[BarId.INPUT]
        tc_dot = theta_dot[BarId.COUPLER]
        to_dot = theta_dot[BarId.OUTPUT]

        return np.array([
            ti_dot,
            tc_dot - ti_dot,
            to_dot - tc_dot,
        ])

    def get_mr_ddthetalist(self) -> np.ndarray:
        theta_ddot = self.kinematics.theta_ddot.theta_ddot
    
        ti_ddot = theta_ddot[BarId.INPUT]
        tc_ddot = theta_ddot[BarId.COUPLER]
        to_ddot = theta_ddot[BarId.OUTPUT]

        return np.array([
            ti_ddot,
            tc_ddot - ti_ddot,
            to_ddot - tc_ddot,
        ])

    def get_mlist(self) -> list[np.ndarray]:
        input_bar = self.geometry.bar[BarId.INPUT]
        coupler_bar = self.geometry.bar[BarId.COUPLER]
        output_bar = self.geometry.bar[BarId.OUTPUT]

        if input_bar.center_of_mass is None:
            raise ValueError("Input center of mass is not defined")

        if coupler_bar.center_of_mass is None:
            raise ValueError("Coupler center of mass is not defined")

        if output_bar.center_of_mass is None:
            raise ValueError("Output center of mass is not defined")

        Li = input_bar.length
        Lc = coupler_bar.length
        Lo = output_bar.length

        pi_2d = input_bar.center_of_mass.to_array()
        pc_2d = coupler_bar.center_of_mass.to_array()

        # Output COM is originally expressed from D -> C.
        po_fourbar = output_bar.center_of_mass.to_array()

        # Convert it to the MR frame C -> D.
        po_2d = np.array([
            Lo - po_fourbar[0],
            -po_fourbar[1],
        ])

        pi = np.array([pi_2d[0], pi_2d[1], 0.0])
        pc = np.array([pc_2d[0], pc_2d[1], 0.0])
        po = np.array([po_2d[0], po_2d[1], 0.0])

        M01 = self.translation_transform(pi)

        M12 = self.translation_transform(
            np.array([Li, 0.0, 0.0]) - pi + pc
        )

        M23 = self.translation_transform(
            np.array([Lc, 0.0, 0.0]) - pc + po
        )

        M34 = self.translation_transform(
            np.array([Lo, 0.0, 0.0]) - po
        )

        return [
            M01,
            M12,
            M23,
            M34,
        ]

    def get_mass_matrix(self) -> np.ndarray:
        thetalist = self.get_mr_thetalist()
        Mlist = self.get_mlist()
        Glist = self.get_glist()
        Slist = self.get_slist()

        return mr.MassMatrix(
            thetalist,
            Mlist,
            Glist,
            Slist
        )

    def get_mr_home(self) -> np.ndarray:
        Mlist = self.get_mlist()
        M = np.eye(4)
        for Mi in Mlist:
            M = M @ Mi
        return M

    def get_kinetic_energy(self) -> float:
        M = self.get_mass_matrix()
        dtheta = self.get_mr_dthetalist()

        return float(0.5 * dtheta @ M @ dtheta)

    def get_kinetic_energy_mr(self) -> float:
        M = self.get_mass_matrix()
        dthetalist = self.get_mr_dthetalist()

        return float(
            0.5 * dthetalist @ M @ dthetalist
        )

    def get_kinetic_energy_direct(self) -> float:
        energy = 0.0

        for bar_id in (
            BarId.INPUT,
            BarId.COUPLER,
            BarId.OUTPUT,
        ):
            bar = self.geometry.bar[bar_id]

            if bar.mass is None:
                raise ValueError(f"Mass not defined for {bar_id}")

            if bar.inertia is None:
                raise ValueError(f"Inertia not defined for {bar_id}")

            if bar.center_of_mass is None:
                raise ValueError(f"Center of mass not defined for {bar_id}")

            com = bar.center_of_mass.to_array()

            point_com = Point(
                x=com[0],
                y=com[1],
            )

            velocity = self.kinematics.get_point_velocity(
                bar_id,
                point_com,
            )

            omega = self.kinematics.theta_dot.theta_dot[bar_id]

            translational_energy = (
                0.5
                * bar.mass
                * velocity @ velocity
            )

            rotational_energy = (
                0.5
                * bar.inertia
                * omega**2
            )

            energy += translational_energy + rotational_energy

        return float(energy)

    # ------------------------------------------------------------------
    # Transformación M_MR -> coordenadas del four-bar
    # ------------------------------------------------------------------

    def get_mass_matrix_fourbar(self) -> np.ndarray:
        """M_fb = Tᵀ·M_MR·T (3x3 en q = [ti, tc, to])."""
        T = self.get_mr_coordinate_matrix()
        return T.T @ self.get_mass_matrix() @ T

    # ------------------------------------------------------------------
    # Reducción al único DOF
    # ------------------------------------------------------------------

    def get_reduced_mass(self) -> float:
        """M_red = Γᵀ·M_fb·Γ (inercia equivalente referida a ti)."""
        gamma = self.kinematics.get_velocity_ratios()
        return float(gamma @ self.get_mass_matrix_fourbar() @ gamma)

    # ------------------------------------------------------------------
    # Coriolis / centrífugas
    # ------------------------------------------------------------------

    def get_reduced_coriolis(self) -> float:
        """
        C_red = Γᵀ·(M_fb·γ + Tᵀ·c_MR(θ, θ̇)).
        γ: aceleración de la cadena cerrada con ẗi = 0.
        c_MR: términos de velocidad de la cadena abierta (MR).
        Requiere haber llamado a kinematics.solve_theta_dot().
        """
        T = self.get_mr_coordinate_matrix()
        gamma = self.kinematics.get_velocity_ratios()
        closed_chain_acc = self.kinematics.get_velocity_product_acceleration()
        c_mr = mr.VelQuadraticForces(
            self.get_mr_thetalist(),
            self.get_mr_dthetalist(),
            self.get_mlist(),
            self.get_glist(),
            self.get_slist(),
        )
        bias = self.get_mass_matrix_fourbar() @ closed_chain_acc + T.T @ c_mr
        return float(gamma @ bias)

    # ------------------------------------------------------------------
    # Gravedad
    # ------------------------------------------------------------------

    def get_gravity_ground_frame(self) -> np.ndarray:
        """Gravedad expresada en el marco base MR (A, eje x según AD)."""
        theta_g = self.kinematics.configuration.theta[BarId.GROUND]
        Rg = self.kinematics.rotation_matrix(theta_g)
        g_local = Rg.T @ self.gravity
        return np.array([g_local[0], g_local[1], 0.0])

    def get_reduced_gravity(self) -> float:
        """G_red = Γᵀ·Tᵀ·g_MR(θ)."""
        T = self.get_mr_coordinate_matrix()
        gamma = self.kinematics.get_velocity_ratios()
        g_mr = mr.GravityForces(
            self.get_mr_thetalist(),
            self.get_gravity_ground_frame(),
            self.get_mlist(),
            self.get_glist(),
            self.get_slist(),
        )
        return float(gamma @ T.T @ g_mr)

    # ------------------------------------------------------------------
    # Muelles y fuerzas externas
    # ------------------------------------------------------------------

    def get_absolute_angle(self, bar_id: BarId) -> float:
        theta = self.kinematics.configuration.theta
        if bar_id == BarId.GROUND:
            return theta[BarId.GROUND]
        return theta[BarId.GROUND] + theta[bar_id]

    def get_spring_loads(self) -> list[AppliedLoad]:
        loads: list[AppliedLoad] = []

        for spring in self.compression_springs:
            bar_a = BarId(spring.a.bar)
            bar_b = BarId(spring.b.bar)
            p_a = self.kinematics.get_local_to_global(
                bar_a, Point(x=spring.a.position.to_array()[0],
                             y=spring.a.position.to_array()[1]))
            p_b = self.kinematics.get_local_to_global(
                bar_b, Point(x=spring.b.position.to_array()[0],
                             y=spring.b.position.to_array()[1]))
            delta = p_b - p_a
            length = float(np.linalg.norm(delta))
            if length < 1e-12:
                raise ValueError("Compression spring with zero length")
            u = delta / length
            # Compresión: sólo empuja (no trabaja a tracción).
            magnitude = spring.constant * max(spring.free_length - length, 0.0)
            loads.append(AppliedLoad(bar_a, p_a, -magnitude * u))
            loads.append(AppliedLoad(bar_b, p_b, magnitude * u))

        for spring in self.torsion_springs:
            bar_a = BarId(spring.bar_a)
            bar_b = BarId(spring.bar_b)
            phi = (self.get_absolute_angle(bar_b)
                   - self.get_absolute_angle(bar_a))
            # Deflexión envuelta a (-π, π]: los ángulos de atan2/acos
            # pueden saltar 2π entre configuraciones.
            deflection = np.angle(np.exp(1j * (phi - spring.free_angle)))
            torque = -spring.constant * deflection
            zero = np.zeros(2)
            loads.append(AppliedLoad(bar_b, zero, zero, torque))
            loads.append(AppliedLoad(bar_a, zero, zero, -torque))

        return loads

    def get_compression_spring_length(
        self, spring: CompressionSpring
    ) -> float:
        p_a = self.kinematics.get_local_to_global(
            BarId(spring.a.bar), Point(x=spring.a.position.to_array()[0],
                                       y=spring.a.position.to_array()[1]))
        p_b = self.kinematics.get_local_to_global(
            BarId(spring.b.bar), Point(x=spring.b.position.to_array()[0],
                                       y=spring.b.position.to_array()[1]))
        return float(np.linalg.norm(p_b - p_a))

    def get_spring_potential_energy(self) -> float:
        energy = 0.0
        for spring in self.compression_springs:
            length = self.get_compression_spring_length(spring)
            compression = max(spring.free_length - length, 0.0)
            energy += 0.5 * spring.constant * compression**2
        for spring in self.torsion_springs:
            phi = (self.get_absolute_angle(BarId(spring.bar_b))
                   - self.get_absolute_angle(BarId(spring.bar_a)))
            deflection = np.angle(np.exp(1j * (phi - spring.free_angle)))
            energy += 0.5 * spring.constant * deflection**2
        return float(energy)

    def get_gravity_potential_energy(self) -> float:
        """V_g = -Σ m·g·r_G (referencia: origen global)."""
        energy = 0.0
        for bar_id in (BarId.INPUT, BarId.COUPLER, BarId.OUTPUT):
            bar = self.geometry.bar[bar_id]
            if bar.mass is None or bar.center_of_mass is None:
                raise ValueError(f"Mass/COM not defined for {bar_id}")
            com = self.kinematics.get_local_to_global(
                bar_id, bar.center_of_mass)
            energy -= bar.mass * self.gravity @ com
        return float(energy)

    def get_external_potential_energy(self) -> float:
        """
        Potencial de las ExternalForce (constantes en ejes globales):
        V = -Σ (F·r_p + τ·θ). Su variación es menos el trabajo realizado.
        """
        energy = 0.0
        for load in self.get_external_loads():
            energy -= load.force @ load.point
            energy -= load.torque * self.get_absolute_angle(load.bar)
        return float(energy)

    def get_external_loads(self) -> list[AppliedLoad]:
        return [
            AppliedLoad(
                load.bar,
                self.kinematics.get_local_to_global(load.bar, load.point),
                np.array(load.force, dtype=float),
                load.torque,
            )
            for load in self.external_forces
        ]

    def get_actuator_unit_load(self, actuator: Actuator) -> AppliedLoad:
        if actuator.direction is None:
            return AppliedLoad(actuator.bar, np.zeros(2), np.zeros(2), 1.0)
        u = np.array(actuator.direction, dtype=float)
        u = u / np.linalg.norm(u)
        if actuator.local_direction:
            u = self.kinematics.rotation_matrix(
                self.get_absolute_angle(actuator.bar)) @ u
        p = self.kinematics.get_local_to_global(actuator.bar, actuator.point)
        return AppliedLoad(actuator.bar, p, u)

    def get_generalized_force(self, loads: list[AppliedLoad]) -> np.ndarray:
        """Q_fb = Σ J_pᵀ·F + par en la columna de su barra (trabajo virtual)."""
        Q = np.zeros(3)
        for load in loads:
            if load.bar == BarId.GROUND:
                continue
            p_local = self.kinematics.get_global_to_local(
                load.bar, Point(x=load.point[0], y=load.point[1]))
            J = self.kinematics.get_point_jacobian(
                load.bar, Point(x=p_local[0], y=p_local[1]))
            Q += J.T @ load.force
            Q[int(load.bar) - 1] += load.torque
        return Q

    def get_reduced_force(self, loads: list[AppliedLoad]) -> float:
        gamma = self.kinematics.get_velocity_ratios()
        return float(gamma @ self.get_generalized_force(loads))

    def get_reduced_spring_force(self) -> float:
        return self.get_reduced_force(self.get_spring_loads())

    def get_reduced_external_force(self) -> float:
        return self.get_reduced_force(self.get_external_loads())

    # ------------------------------------------------------------------
    # Resolver ẗi  /  esfuerzo requerido λ
    # ------------------------------------------------------------------

    def solve_theta_input_ddot(
        self,
        actuator: Actuator | None = None,
        actuator_value: float = 0.0,
    ) -> float:
        """
        Dinámica directa: ẗi = (Q + λ·Q_act − C_red − G_red) / M_red.
        Actualiza kinematics.theta_ddot.
        """
        rhs = (self.get_reduced_spring_force()
               + self.get_reduced_external_force()
               - self.get_reduced_coriolis()
               - self.get_reduced_gravity())
        if actuator is not None:
            unit = self.get_actuator_unit_load(actuator)
            rhs += actuator_value * self.get_reduced_force([unit])
        theta_input_ddot = rhs / self.get_reduced_mass()
        self.kinematics.solve_theta_ddot(theta_input_ddot)
        return theta_input_ddot

    def solve_required_effort(
        self,
        actuator: Actuator,
        theta_input_ddot: float = 0.0,
    ) -> float:
        """
        Dinámica inversa: magnitud λ del actuador para conseguir ẗi
        (ẗi = 0 y ṫi = 0 -> equilibrio estático con los muelles).
        Actualiza kinematics.theta_ddot.
        """
        self.kinematics.solve_theta_ddot(theta_input_ddot)
        q_act = self.get_reduced_force([self.get_actuator_unit_load(actuator)])
        if abs(q_act) < 1e-12:
            raise RuntimeError(
                "The actuator does no work in this configuration "
                "(singular position or force through the pivot)")
        lhs = (self.get_reduced_mass() * theta_input_ddot
               + self.get_reduced_coriolis()
               + self.get_reduced_gravity()
               - self.get_reduced_spring_force()
               - self.get_reduced_external_force())
        return lhs / q_act

