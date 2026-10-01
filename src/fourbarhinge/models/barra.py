
from typing import List, Optional
from math import cos, sin, atan2, sqrt
import numpy as np
from pydantic import BaseModel
from .constants import BarId
from .compression_spring import CompressionSpring
from .torsion_spring import TorsionSpring


class Point(BaseModel):
    x: float
    y: float

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])


class Bar (BaseModel):
    length: float
    mass: Optional[float] = None
    inertia: Optional[float] = None
    center_of_mass: Optional[Point] = None

    def get_spatial_inertia(self) -> np.ndarray:
        if self.mass is None:
            raise ValueError("Bar mass must be defined for dynamics")

        if self.inertia is None:
            raise ValueError("Bar inertia must be defined for dynamics")

        return np.diag([
            0.0,
            0.0,
            self.inertia,
            self.mass,
            self.mass,
            self.mass,
        ])


class FourBarGeometry(BaseModel):
    position_ground: Point
    bar: dict[BarId, Bar]


class FourBarConfiguration(BaseModel):
    theta: dict[BarId, float]

    def theta_to_array(self) -> np.ndarray:
        return np.array([
            self.theta[BarId.GROUND],
            self.theta[BarId.INPUT],
            self.theta[BarId.COUPLER],
            self.theta[BarId.OUTPUT],
            ])


class FourBarVelocities(BaseModel):
    theta_dot: dict[BarId, float]

    def theta_dot_to_array(self) -> np.ndarray:
        return np.array([
            self.theta_dot[BarId.INPUT],
            self.theta_dot[BarId.COUPLER],
            self.theta_dot[BarId.OUTPUT],
            ])


class FourBarAccelerations(BaseModel):
    theta_ddot: dict[BarId, float]

    def theta_ddot_to_array(self) -> np.ndarray:
        return np.array([
            self.theta_ddot[BarId.INPUT],
            self.theta_ddot[BarId.COUPLER],
            self.theta_ddot[BarId.OUTPUT],
            ])


class Mechanism(BaseModel):
    geometry: FourBarGeometry
    compression_springs: list[CompressionSpring]
    torsion_spring: List[TorsionSpring]


def convert_2_cartesian(length: float, angle: float):
    point = Point(x=length * cos(angle),
                  y=length * sin(angle))
    return point


def convert_2_angular(point: Point):
    length = sqrt(point.x**2 + point.y**2)
    angle = atan2(point.y, point.x)
    return length, angle

