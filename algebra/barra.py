import math
import numpy as np
from algebra.vector import Vector
from algebra.uniones import Union
from algebra.diverse import get_angle_vector, convert_vector_to_unitary, convert_cartessian_to_polar, convert_polar_to_cartessian

  
class Barra(Vector, Union):
	def __init__(self, start_point:Union, end_point:Union) -> None:
		self._displacement = start_point
		self._rotation = 0
		unionpoint = Union(end_point.get_point() - start_point.get_point())
		self.__vector = Vector (unionpoint, displacement=self._displacement, rotation = self._rotation)
		self.actuator = []
		self.__cir = start_point.get_point()
		self._gravity_center = convert_vector_to_unitary(self.__vector.get_cartessian_vector()) * np.linalg.norm(self.__vector.get_cartessian_vector()) / 2
		self._weight = 0

	def	set_new_geometry(self, end_point:Union, start_point:Union = None):
		if start_point is not None:
			self._displacement = start_point
		angle = get_angle_vector(end_point.get_point() - self._displacement.get_point())
		self.__vector.set_displacement(self._displacement.get_point())
		self.__vector.rotate_to(angle)
		for each in self.actuator:
			each.set_displacement(self._displacement.get_point())
			each.set_rotation(self.__vector.get_rotation())
  
	def set_actuator_point(self, actuator_point:Union):
		relative_point = actuator_point.get_point() - self._displacement.get_point()
		vector = Vector(Union(relative_point), displacement = self._displacement, rotation = self._rotation)
		self.actuator.append(vector)
  
	def get_actuator(self, item):
		return self.actuator[item]
 
	def get_actuator_point(self, item):
		return self.actuator[item].get_cartessian_end_point()

	def get_start_point(self):
		return self.__vector.get_displacement()

	def get_end_point(self):
		return self.__vector.get_cartessian_end_point()

	def get_start_union(self):
		return self._displacement

	def get_end_union(self):
		union_end = Union(self._displacement.get_point() + self.__vector.get_cartessian_vector())
		return union_end

	def get_start_torque(self):
		return self.__displacement.get_torque()

	def get_end_torque(self):
		return self.__vector.get_torque()

	def get_length(self):
		return self.__vector.get_polar_vector()[0]

	def get_angle(self):
		return self.__vector.get_angle()

	def get_rotation(self):
		return self.__vector.get_rotation()

	def rotate(self, angle:float, degrees:bool=True):
		if degrees:
			self._rotation = math.radians(angle)
		else:
			self._rotation = angle
		self.__vector.set_rotation(self._rotation)
		for each in self.actuator:
			each.set_rotation(self._rotation)
	def move(self, displacement:np):
		self._displacement.set_point(self._displacement + displacement)
		self.__vector.set_displacement(self._displacement)
		for each in self.actuator:
			each.set_displacement(self._displacement)
  
	def move_to(self, start_point:np):
		self._displacement.set_point(start_point)
		self.__vector.set_displacement(self._displacement)
		for each in self.actuator:
			each.set_displacement(self._displacement)
  
	def change_length(self, length:float):
		self.__vector = length * np.array([math.cos(self.angle + self.rotation), math.sin(self.angle + self.rotation)])
		self.length = length

	def get_cartessian_vector(self):
		return self.__vector.get_cartessian_vector()

	def get_polar_vector(self):
		return self.__vector.get_polar_vector()

	def get_array_to_plot(self):
		output = np.array([self.get_start_point(), self.get_end_point()]).T
		for actuator in range (0, len(self.actuator)):
			vector = np.array([self.get_actuator_point(actuator),self.get_end_point()])
			output = np.concatenate((output,vector.T),axis = 1)
		return output

	def set_cir(self, cir:np):
		self.__cir =  cir - self._displacement.get_point()

	def get_cir(self):
		return self.__cir + self._displacement.get_point()

	def set_gravity_center (self, gravity_center):
		self._gravity_center = gravity_center - self.__displacement

	def get_gravity_center (self):
		polar_GC = convert_cartessian_to_polar(self._gravity_center) + np.array([0, self._rotation])
		cartessian_CG = convert_polar_to_cartessian(polar_GC) + self._displacement
		return cartessian_CG

	def set_weight (self, weight):
		self._weight = weight

	def get_weight (self):
		return self._weight


"""
mi_barra = Barra(np.array([0,0]),np.array([2,2]))
mi_barra.set_actuator_point (np.array([3,1]))
mi_barra.rotate(90)
print(mi_barra.get_array_to_plot())"""