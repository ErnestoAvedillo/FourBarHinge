import numpy as np
import math

class IntersectionLineCircle():
    def __init__(self, line, circle):
        self.line = line
        self.circle = circle
    def get_cartesian_intersection(self):
        # Solve for t and s in the parametric equations of the line and the circle
        A = np.array([self.line.vector, -2 * self.circle.center.vector]).T
        b = np.array(self.circle.center.vector**2 - self.circle.radius**2)
        try:
            t, s = np.linalg.solve(A, b)
            intersection_point = self.line.point.vector + t * self.line.vector
            return intersection_point
        except np.linalg.LinAlgError:
            return None
    def get_polar_intersection(self):
        return np.array([np.linalg.norm(self.cartesian_intersection()), math.atan2(self.cartesian_intersection()[1], self.cartesian_intersection()[0])])
    
class IntersectionCircleCircle():
    def __init__(self, circle1, circle2, point1, point2):
        self.circle1 = circle1
        self.circle2 = circle2
        self.point1 = point1
        self.point2 = point2
    def get_cartesian_intersection(self):
        # Solve for t and s in the parametric equations of the circles
        A = np.array([2 * (self.circle1.center.vector - self.circle2.center.vector)]).T
        b = np.array(self.circle2.center.vector**2 - self.circle1.center.vector**2 + self.circle1.radius**2 - self.circle2.radius**2)
        try:
            t = np.linalg.solve(A, b)
            intersection_point = self.circle1.center.vector + t * (self.point1.vector - self.point2.vector)
            return intersection_point
        except np.linalg.LinAlgError:
            return None
    def get_polar_intersection(self):
        return np.array([np.linalg.norm(self.cartesian_intersection()), math.atan2(self.cartesian_intersection()[1], self.cartesian_intersection()[0])])
