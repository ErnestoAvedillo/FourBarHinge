import numpy as np
class MuelleCompresion:
	def __init__(self, longitud_inicial:float, constante:float):
		self.__characteristic = np.array([[longitud_inicial,constante]])
		self.longitud_inicial = longitud_inicial
		self.constante = constante
		
	def get_force (self, longitud_actual:float):
		returned_force = 0
		for pos in range(0, len(self.__characteristic)):
			if longitud_actual > self.__characteristic[pos,0]:
				return returned_force
			else:
				if pos < len(self.__characteristic) - 1:
					returned_force += (self.longitud_inicial - max(longitud_actual, self.__characteristic[pos + 1,0])) * self.__characteristic[pos, 1]
				else:
					returned_force += (self.__characteristic[pos, 0] - longitud_actual) * self.__characteristic[pos, 1]
					return returned_force

	def add_constant(self, longitud_inicial:float, constante:float):
		self.__characteristic = np.append(self.__characteristic, np.array([[longitud_inicial,constante]]), axis= 0)


"""muelle = MuelleCompresion(100,1)
muelle.add_constant(50, 2)
print (muelle.get_force(95))
print (muelle.get_force(49))
print (muelle.get_force(45))"""