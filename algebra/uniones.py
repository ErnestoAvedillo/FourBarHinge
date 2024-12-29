import numpy as np
import copy
from algebra.diverse import convert_polar_to_cartessian, convert_cartessian_to_polar

class Union:
    def __init__(self, point:np, force:np=None, Torque:float = None):
        if len (point) != 2:
            raise Exception ("Only np:array dimension 2 for point is accepted")
        self.__point = point
        if force is None:
            self.__force = np.zeros(2)
        else:
            if len (point) != 2:
                raise Exception ("Only np:array dimension 2 for point is accepted")
            self.__force = force
        if Torque is None:
            self.__torque = 0
        else:
            self.__torque = Torque
    def __str__(self):
        return (f"{self.__point}")
    
    def copy(self):
        return copy.deepcopy(self)

    def get_point(self)-> np:
        return self.__point
    
    def set_point(self, point:np):
        if point is None:
            self.__point = np.zeros(2)
        else:
            self.__point = point
        
    def move_point(self,point:np):
        self.__point = self.__point + point
    
    def set_forces(self, force:np):
        self.__force = force
    
    def get_force(self)-> np:
        return self.__force
    
    def __add__(self, other)-> np:
        self.move_point(other.get_point())
        return self
    
    def __sub__(self, other):
        self.move_point(-1 * other.get_point())
        return self
    
    def convert_to_cartessian (self):
        point = convert_polar_to_cartessian(self.__point)
        force = convert_polar_to_cartessian(self.__force)
        return Union(point, force)
    def convert_to_polar (self):
        point = convert_cartessian_to_polar(self.__point)
        force = convert_cartessian_to_polar(self.__force)
        return Union(point, force)
    
    def set_torque(self, torque:float):
        self.__torque = torque
    
    def get_torque(self):
        return self.__torque