from __future__ import annotations
import math
from cartesian_vector import CartesianVector


class PolarVector:
    def __init__(self, r, theta):
        self.r = r
        self.theta = theta
    def __str__(self):
        return f"({self.r}, {self.theta})"
    def get_module(self):
        return self.r
    def get_angle(self):
        return self.theta
    def to_cartesian(self):
        return CartesianVector(self.r * math.cos(self.theta), self.r * math.sin(self.theta))
