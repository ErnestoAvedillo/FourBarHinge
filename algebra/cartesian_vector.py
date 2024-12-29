from __future__ import annotations
import math
import numpy as np

class CartesianVector:
    def __init__(self, x:float, y:float):
        self.vector = np.zeros(2)
        self.vector[0] = x
        self.vector[1] = y
        
    def __init (self, other:CartesianVector):
        self = other

    def __str__(self):
        return f"({self.vector[0]}, {self.vector[1]})"

    def __add__(self, other:CartesianVector) -> CartesianVector:
        return CartesianVector(self.vector + other.vector)

    def __sub__(self, other:CartesianVector) -> CartesianVector:
        return CartesianVector(self.vector - other.vector)
    
    def __mul__(self, other):
        if isinstance(other, CartesianVector):
            return self.vector[0] * other.vector[0] + self.vector[1] * other.vector[1]
        else:
            return CartesianVector(self.vector[0] * other, self.vector[1] * other)
        
    def __truediv__(self, other:float) ->float:
        return CartesianVector(self.vector[0] / other, self.vector[1] / other)

    def __truediv__(self, other:int) ->float:
        return CartesianVector(self.vector[0] / other, self.vector[1] / other)
        
    def __neg__(self):
        return CartesianVector(self.vector[0], self.vector[1])
    
    def __eq__(self, other: CartesianVector) -> bool:
        return self.vector[0] == other.vector[0] and self.vector[1] == other.vector[1]

    def __getitem__(self, index: int) -> float:
        if index < 0 or index >= len(self.vector):
            raise IndexError("Índice fuera de rango")
        return self.vector[index]
    
    def vectorial_product(self, other:CartesianVector) -> CartesianVector:
        return np.cross(self.vector, other.vector)
    
    def dot_product(self, other:CartesianVector) -> float:
        return (self.vector [0] * other[0] + self.vector [1] * other[1])
    
    def magnitude(self):
        return np.linalg.norm(self.vector)

    def angle(self):
        math.atan2(self.vector[1], self.vector[0])

    def unitary(self):
        return self.vector / self.magnitude()
    
    def angle(self, other:CartesianVector) -> CartesianVector:
        return self.dot_product(self.vector, other.vector) / (self.magnitude(self.vector) * self.magnitude(other.vector))
    
