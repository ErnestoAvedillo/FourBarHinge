import numpy as np
from algebra.diverse import convert_polar_to_cartessian, convert_cartessian_to_polar
from algebra.uniones import Union
import copy
class Vector(Union):
	def __init__(self, cartessian:Union = None, polar:Union = None, displacement:Union = None, rotation:float = 0) -> None:
		if cartessian is not None and polar is not None:
			raise Exception("only one argument accepted")
		if polar is None:
			self.__cartessian = cartessian
			self.__polar = cartessian.convert_to_polar()

		if cartessian is None:
			self.__polar = polar
			self.__cartessian = self.__polar.convert_to_cartessian()
		self.__rotation = rotation
		if displacement is None:
			self.__displacement = Union(np.zeros(2))
		else:
			self.__displacement = displacement

	def __str__(self):
		end_pos = self.get_cartessian_end_point()
		start_pos = self.get_displacement()
		vect= self.get_cartessian_vector()
		pol = self.get_polar_vector()
		rot = self.get_rotation()
		return (f"Pos inicial cartesiana {start_pos} Pos final cartesiana {end_pos}\n"
          		f"Vector cartesiano {vect}  Vect polar {pol} Rotacion {rot}")

	def get_cartessian_vector(self):
		return self.__cartessian.get_point()
	def get_polar_vector(self):
		return self.__polar.get_point()
	def get_displacement(self):
		return self.__displacement.get_point()
	def get_rotation(self):
		return self.__rotation
	def get_angle (self):
		return (self.__polar.get_point()[1] + self.__rotation)
	def get_length(self):
		return self.__polar.get_point()[0]
	def get_cartessian_end_point(self)->np:
		polar = self.__polar.get_point() + np.array([0,self.__rotation])
		cartessian1 = convert_polar_to_cartessian(polar) 
		cartessian2 = self.__displacement.get_point()
		return cartessian1 + cartessian2
	def get_polar_end_point(self)->np:
		polar = self.__polar.get_point() + np.array([0,self.__rotation])
		cartessian = convert_polar_to_cartessian(polar) + self.__displacement.get_point()
		return convert_cartessian_to_polar(cartessian)
	def set_rotation (self, rotation:float):
		self.__rotation = rotation
	def rotate_to(self, angle:float):
		self.__rotation = angle - self.__polar.get_point()[1]
	def set_displacement(self, displacement:np):
		self.__displacement.set_point(displacement)
	def set_new_vector(self, cartessian:np = None, polar:np = None):
		if cartessian is not None and polar is not None:
			raise Exception("only one argument accepted")
		if polar is None:
			if len(cartessian) != 2:
				raise Exception ("Only dimension of vector 2 is accepted")
			self.__cartessian.set_point(cartessian)
			self.__polar = self.__cartessian.convert_to_polar()

		if cartessian is None:
			if len(polar) != 2:
				raise Exception ("Only dimension of vector 2 is accepted")
			self.__polar.set_point(polar)
			self.__cartessian = self.__polar.convert_to_cartessian()
	def __add__(self, other):
		return Vector(cartessian = self.__cartessian.get_point() + other.get_cartessian_vector())
	def __sub__(self, other):
		return Vector(cartessian = self.__cartessian.get_point() - other.get_cartessian_vector())
