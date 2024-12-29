import numpy as np
import math
from algebra.barra import Barra

class IntersectionCircleCircle:
	def __init__(self, point1:np, circle1:float, point2:np, circle2:float):
		if circle1 <= 0.0 or circle2 <= 0.0:
			raise ValueError("The radius of the circles must be positive")
		self.point1 = point1
		self.circle1 = circle1
		self.point2 = point2
		self.circle2 = circle2
	def get_cartesian_intersection(self):
		vector_P1_p2 = self.point2 - self.point1
		d = np.linalg.norm(vector_P1_p2)
		if d > self.circle1 + self.circle2:
			return np.array([None,None]),np.array(([None,None]))
		if d < abs(self.circle1 - self.circle2):
			return np.array([None,None]),np.array(([None,None]))
		if d == 0.0:
			return np.array([None,None]),np.array(([None,None]))
		angle1 = (self.circle1**2 - self.circle2**2 + d**2) / (2 * d * self.circle1)
		angle_p1_p2 = math.atan2(vector_P1_p2[1],vector_P1_p2[0])
		angle_v1 = angle_p1_p2 - math.acos(angle1)
		v1 = self.circle1 * np.array([math.cos(angle_v1),math.sin(angle_v1)]) + self.point1
		angle_v2 = angle_p1_p2 + math.acos(angle1)
		v2 = self.circle1 * np.array([math.cos(angle_v2),math.sin(angle_v2)]) + self.point1
		return v1, v2
	def get_polar_intersection(self):
		intersection_point1,intersection_point2 = self.get_cartesian_intersection()
		if intersection_point1 is None:
			polar_intersection_point1 = None
		else:
			polar_intersection_point1 = np.array([np.linalg.norm(intersection_point1), np.arctan2(intersection_point1[1], intersection_point1[0])])
		if intersection_point2 is None:
			polar_intersection_point2 = None
		else:
			polar_intersection_point2 = np.array([np.linalg.norm(intersection_point2), np.arctan2(intersection_point2[1], intersection_point2[0])])
		return polar_intersection_point1, polar_intersection_point2

"""
p1 = np.array([0, 0])
p2 = np.array([0, 1])
c1 = 1
c2 = 1
a = IntersectionCircleCircle(p1, c1, p2, c2)
print(a.get_cartesian_intersection())
print(a.get_polar_intersection())
"""