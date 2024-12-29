import numpy as np
import math
from distance_point_line import DistancePointLine

class IntersectionLineCircle:
    def __init__(self, Vector_line:np, point_line:np, point_radius:np, radius:float):
        try:
            self.Vector_line = Vector_line / np.linalg.norm(Vector_line)
        except:
            raise ValueError("The vector length must be different from the zero vector")
        self.point_line = point_line
        self.point_radius = point_radius
        if radius <= 0:
            raise ValueError("The radius must be greater than zero")
        self.radius = radius
        
    def get_cartessian_intersection_line_circle(self):
        # Find the distance from the center of the circle to the line
        distance = DistancePointLine(self.point_radius, self.Vector_line, self.point_line)
        # If the distance is greater than the radius of the circle, then the line does not intersect the circle
        if distance.get_distance() > self.radius:
            return None
        
        # Find the intersection points
        # Find the projection of the center of the circle onto the line
        projection = distance.get_cartessian_proyection_point()
        # Find the distance from the projection to the intersection points
        distance_projection_intersection = np.sqrt(self.radius**2 - distance.get_distance()**2)
        
        # Find the intersection points
        intersection1 = projection + distance_projection_intersection * self.Vector_line
        intersection2 = projection - distance_projection_intersection * self.Vector_line
        return intersection1, intersection2
    def get_polar_intersection_line_circle(self):
        intersection1, intersection2 = self.get_cartessian_intersection_line_circle()
        try:
            polar_intersection1 = np.array([np.linalg.norm(intersection1), math.atan2(intersection1[1],intersection1[0])])
        except:
            polar_intersection1 = None
        try:
            polar_intersection2 = np.array([np.linalg.norm(intersection2), math.atan2(intersection2[1],intersection2[0])])
        except:
            polar_intersection2 = None
        return polar_intersection1, polar_intersection2
"""
# Example
Vector_line = np.array([1,1])
point_line = np.array([0,0])
point_radius = np.array([1,1])
radius = 1
i = IntersectionLineCircle(Vector_line, point_line, point_radius, radius)
print(i.get_cartessian_intersection_line_circle())
print(i.get_polar_intersection_line_circle())

"""