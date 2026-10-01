
"""
Genera el informe PDF (A4) de una bisagra de ejemplo.
Uso:  uv run python examples/informe_bisagra.py
Unidades coherentes mm-kg-s: fuerza mN, par mN·mm, energía µJ.
"""

import numpy as np
from fourbarhinge.models.barra import Bar, Point, FourBarGeometry
from fourbarhinge.models.constants import BarId
from fourbarhinge.models.position import Position, Attachment
from fourbarhinge.models.compression_spring import CompressionSpring
from fourbarhinge.models.torsion_spring import TorsionSpring
from fourbarhinge.models.external_force import Actuator
from fourbarhinge.kinematics.kinematics import FourBarKinematics
from fourbarhinge.dynamics.dynamics import FourBarDynamics
from fourbarhinge.simulation.return_simulation import ReturnSimulation
from fourbarhinge.plots.plots import FourBarPlotter

BRANCH = -1
THETA_START = np.deg2rad(150)
THETA_HOME = np.deg2rad(40)


def make_bar(length: float, mass: float) -> Bar:
    return Bar(length=length, mass=mass, inertia=mass * length**2 / 12,
               center_of_mass=Point(x=length / 2, y=0.0))


def main() -> None:
    geometry = FourBarGeometry(
        position_ground=Point(x=0.0, y=0.0),
        bar={
            BarId.GROUND: make_bar(60.0, 0.05),
            BarId.INPUT: make_bar(25.0, 0.02),
            BarId.COUPLER: make_bar(70.0, 0.06),
            BarId.OUTPUT: make_bar(55.0, 0.04),
        },
    )
    kinematics = FourBarKinematics(geometry=geometry, theta_ground=0.0)
    dynamics = FourBarDynamics(
        kinematics,
        compression_springs=[
            CompressionSpring(
                free_length=60.0,
                constant=500.0,
                a=Attachment(bar=BarId.GROUND,
                             position=Position(length=0.0, angle=0.0)),
                b=Attachment(bar=BarId.COUPLER,
                             position=Position(length=50.0, angle=0.0)),
            ),
        ],
        torsion_springs=[
            TorsionSpring(free_angle=np.deg2rad(20), constant=50_000.0,
                          bar_a=BarId.GROUND, bar_b=BarId.INPUT),
        ],
        gravity=(0.0, -9810.0),
    )
    # Empuje en el extremo del acoplador, siempre perpendicular a él
    actuator = Actuator(bar=BarId.COUPLER, point=Point(x=70.0, y=0.0),
                        direction=(0.0, 1.0), local_direction=True)

    trajectory = ReturnSimulation(dynamics, BRANCH).run(
        THETA_START, THETA_HOME, dt=1e-4, t_max=1.0)
    print(f"Retorno: {trajectory.stop_reason}, "
          f"t = {trajectory.duration * 1e3:.2f} ms, "
          f"ṫi(home) = {trajectory.home_velocity:.2f} rad/s")

    plotter = FourBarPlotter(dynamics, BRANCH, actuator)
    path = plotter.build_report("output/informe_bisagra.pdf", trajectory)
    print(f"Informe: {path}")


if __name__ == "__main__":
    main()
