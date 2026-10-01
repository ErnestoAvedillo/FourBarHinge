import numpy as np
import pytest

from fourbarhinge import (
    Attachment, Bar, BarId, CompressionSpring, FourBarDynamics,
    FourBarGeometry, FourBarKinematics, Point, Position, TorsionSpring,
    ExternalForce,
)

LENGTHS = {BarId.GROUND: 0.50, BarId.INPUT: 0.20, BarId.COUPLER: 0.45,
           BarId.OUTPUT: 0.40}


def make_geometry() -> FourBarGeometry:
    bars = {k: Bar(length=v, mass=1.0 + int(k), inertia=0.01 * (1 + int(k)),
                   center_of_mass=Point(x=0.4 * v, y=0.03))
            for k, v in LENGTHS.items()}
    return FourBarGeometry(position_ground=Point(x=0.1, y=0.2), bar=bars)


@pytest.fixture
def dynamics() -> FourBarDynamics:
    kinematics = FourBarKinematics(make_geometry(), np.deg2rad(20))
    return FourBarDynamics(
        kinematics,
        compression_springs=[CompressionSpring(
            free_length=0.45, constant=200.0,
            a=Attachment(bar=0, position=Position(length=0.1, angle=0.0)),
            b=Attachment(bar=2, position=Position(length=0.2, angle=0.0)))],
        torsion_springs=[
            TorsionSpring(free_angle=0.5, constant=3.0, bar_a=0, bar_b=1),
            TorsionSpring(free_angle=-0.8, constant=2.0, bar_a=2, bar_b=3)],
        external_forces=[ExternalForce(bar=BarId.COUPLER,
                                       point=Point(x=0.3, y=0.05),
                                       force=(1.0, -2.0))],
        gravity=(0.0, -9.81),
    )
