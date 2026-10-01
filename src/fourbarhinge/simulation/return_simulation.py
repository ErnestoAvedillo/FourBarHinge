
"""
Simulación del retorno libre: se suelta el mecanismo en reposo en
theta_start y los muelles/gravedad lo llevan hasta theta_home.
Integración RK4 de la ecuación reducida  M_red·ẗi + C_red + G_red = Q.
"""

from dataclasses import dataclass, field
import numpy as np
from ..dynamics.dynamics import FourBarDynamics
from ..models.constants import BarId


@dataclass
class ReturnTrajectory:
    theta_start: float
    theta_home: float
    time: np.ndarray
    theta_input: np.ndarray
    theta_input_dot: np.ndarray
    theta_input_ddot: np.ndarray
    kinetic_energy: np.ndarray
    spring_energy: np.ndarray
    gravity_energy: np.ndarray
    external_energy: np.ndarray
    # Ángulos de todas las barras (para dibujar/animar cada instante)
    theta: dict[BarId, np.ndarray] = field(default_factory=dict)
    reached_home: bool = False
    stop_reason: str = ""

    @property
    def total_energy(self) -> np.ndarray:
        return (self.kinetic_energy + self.spring_energy
                + self.gravity_energy + self.external_energy)

    @property
    def home_velocity(self) -> float:
        return float(self.theta_input_dot[-1])

    @property
    def duration(self) -> float:
        return float(self.time[-1])


class ReturnSimulation:
    def __init__(self, dynamics: FourBarDynamics, branch: int = 1):
        self.dynamics = dynamics
        self.kinematics = dynamics.kinematics
        self.branch = branch

    def set_state(self, theta_input: float, theta_input_dot: float) -> None:
        self.kinematics.solve_configuration(theta_input, self.branch)
        if self.kinematics.get_closure_error() > 1e-6:
            raise RuntimeError(
                f"theta_input={np.rad2deg(theta_input):.2f}° is outside "
                "the assembly range of the mechanism")
        self.kinematics.solve_theta_dot(theta_input_dot)

    def derivative(self, state: np.ndarray) -> np.ndarray:
        self.set_state(state[0], state[1])
        return np.array([state[1], self.dynamics.solve_theta_input_ddot()])

    def run(
        self,
        theta_start: float,
        theta_home: float,
        dt: float = 1e-4,
        t_max: float = 2.0,
    ) -> ReturnTrajectory:
        """
        Integra desde (theta_start, ṫi = 0) hasta cruzar theta_home.
        El último punto se interpola linealmente sobre theta_home.
        """
        direction = np.sign(theta_home - theta_start)
        state = np.array([theta_start, 0.0])
        t = 0.0
        samples: list[tuple[float, np.ndarray]] = [(t, state.copy())]
        reached_home = False
        stop_reason = f"no llega a home en t_max = {t_max} s"

        while t < t_max:
            try:
                k1 = self.derivative(state)
                k2 = self.derivative(state + 0.5 * dt * k1)
                k3 = self.derivative(state + 0.5 * dt * k2)
                k4 = self.derivative(state + dt * k3)
            except RuntimeError as exc:
                stop_reason = str(exc)
                break
            new_state = state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += dt

            if direction * (new_state[0] - theta_home) >= 0.0:
                s = (theta_home - state[0]) / (new_state[0] - state[0])
                new_state = state + s * (new_state - state)
                samples.append((t - dt + s * dt, new_state))
                reached_home = True
                stop_reason = "home alcanzado"
                break
            if direction * new_state[1] < 0.0 and t > dt:
                samples.append((t, new_state))
                stop_reason = "se detiene y retrocede antes de home"
                break
            state = new_state
            samples.append((t, state.copy()))

        return self.build_trajectory(
            theta_start, theta_home, samples, reached_home, stop_reason)

    def build_trajectory(
        self,
        theta_start: float,
        theta_home: float,
        samples: list[tuple[float, np.ndarray]],
        reached_home: bool,
        stop_reason: str,
    ) -> ReturnTrajectory:
        n = len(samples)
        time = np.array([s[0] for s in samples])
        theta_input = np.array([s[1][0] for s in samples])
        theta_input_dot = np.array([s[1][1] for s in samples])
        theta_input_ddot = np.zeros(n)
        kinetic = np.zeros(n)
        springs = np.zeros(n)
        gravity = np.zeros(n)
        external = np.zeros(n)
        theta = {bar: np.zeros(n) for bar in BarId}

        for i in range(n):
            self.set_state(theta_input[i], theta_input_dot[i])
            theta_input_ddot[i] = self.dynamics.solve_theta_input_ddot()
            kinetic[i] = (0.5 * self.dynamics.get_reduced_mass()
                          * theta_input_dot[i]**2)
            springs[i] = self.dynamics.get_spring_potential_energy()
            gravity[i] = self.dynamics.get_gravity_potential_energy()
            external[i] = self.dynamics.get_external_potential_energy()
            for bar in BarId:
                theta[bar][i] = self.kinematics.configuration.theta[bar]

        return ReturnTrajectory(
            theta_start=theta_start,
            theta_home=theta_home,
            time=time,
            theta_input=theta_input,
            theta_input_dot=theta_input_dot,
            theta_input_ddot=theta_input_ddot,
            kinetic_energy=kinetic,
            spring_energy=springs,
            gravity_energy=gravity,
            external_energy=external,
            theta=theta,
            reached_home=reached_home,
            stop_reason=stop_reason,
        )
