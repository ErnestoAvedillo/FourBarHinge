import numpy as np
from algebra.intersection_circle_circle import IntersectionCircleCircle
from algebra.distance_point_line import DistancePointLine
from algebra.intersection_line_line import IntersectionLineLine
from algebra.barra import Barra
from algebra.vector import Vector
from algebra.uniones import Union
from algebra.diverse import convert_vector_to_unitary
from enum import Enum
from math import pi
class Reacciones(Enum):
    R10x = 0
    R10y = 1
    R11x = 2
    R11y = 3
    R20x = 4
    R20y = 5
    R21x = 6
    R21y = 7
    R30x = 8
    R30y = 9
    R31x = 10
    R31y = 11
    Fx = 12
    Fy = 13
    
class Bisagra(Barra):
	def __init__(self, barra1:Barra, barra2:Barra) -> None:
		self.barras = [
		Barra(barra1.get_start_union(), barra2.get_start_union()),
		barra1,
		barra2,
		Barra(barra1.get_end_union(), barra2.get_end_union())
		]
		self.angle_rot = 0
		self.cir = IntersectionLineLine(barra1.get_start_point(), barra1.get_end_point(), barra2.get_start_point(), barra2.get_end_point())
		self.actuators = []
		self.__alfa0 = 0
		self.__alfa1 = 0
		self.get_travel()
  
	def __str__(self):
		barra0_start = self.barras[0].get_start_point()
		barra0_end = self.barras[0].get_end_point()
		barra1_start = self.barras[1].get_start_point()
		barra1_end = self.barras[1].get_end_point()
		barra2_start = self.barras[2].get_start_point()
		barra2_end = self.barras[2].get_end_point()
		barra3_start = self.barras[3].get_start_point()
		barra3_end = self.barras[3].get_end_point()
		return (f"Barra0: Start {barra0_start}, End {barra0_end}\n"
      			f"Barra1: Start {barra1_start}, End {barra1_end}\n"
				f"Barra2: Start {barra2_start}, End {barra2_end}\n"
				f"Barra3: Start {barra3_start}, End {barra3_end}\n")
	def get_barra(self, index):
		return self.barras[index]

	def rotate(self, angle:float, degrees=True):
		self.barras[1].rotate(angle, degrees=degrees)
		intersection = IntersectionCircleCircle(self.barras[1].get_end_point(), self.barras[3].get_length(), self.barras[2].get_start_point(), self.barras[2].get_length())
		Intersection_point1, Intersection_point2 = intersection.get_cartesian_intersection()
		
		move1 = Intersection_point1 - self.barras[2].get_end_point()
		move2 = Intersection_point2 - self.barras[2].get_end_point()
		if np.linalg.norm(move1) < np.linalg.norm(move2):
			self.barras[2].set_new_geometry(Union(Intersection_point1))
		else:
			self.barras[2].set_new_geometry(Union(Intersection_point2))
		self.barras[3].set_new_geometry(Union(self.barras[2].get_end_point()), Union(self.barras[1].get_end_point()))
		interseccion = IntersectionLineLine(
			self.barras[1].get_cartessian_vector(),
			self.barras[1].get_start_point(),
			self.barras[2].get_cartessian_vector(),
			self.barras[2].get_start_point(),
		)
		self.barras[3].set_cir(interseccion.get_cartesian_intersection())
		return

	def get_complete_geometry(self, Theta:float, degrees = True):
		if not Theta == 0:
			self.rotate(Theta, degrees = degrees)
		return self.barras
	#actuator is defined as a dctionary : {"Barra":1, "Actuator": 1, "Input":True/False} 
	#input is defined Input = True Force done by a spring etc
	#input is defined Input = False Force to be calculated etc
	def define_actuators(self, actuator):
		index = self.actuators.append(actuator)
		return index

	def get_actuators_distance (self, index)->np.array:
		barra_a =  self.actuators[index]["Barra1"]["Barra"]
		actuator_a =  self.actuators[index]["Barra1"]["Actuator"]
		barra_b =  self.actuators[index]["Barra2"]["Barra"]
		actuator_b =  self.actuators[index]["Barra1"]["Actuator"]
		start_position = self.barras[barra_a].get_actuator_point(actuator_a)
		end_position = self.barras[barra_b].get_actuator_point(actuator_b)
		return start_position - end_position

	def get_actuator_force(self, item):
		distance =self.get_actuators_distance(item)
		if self.actuators[item]["Law"] is not None:
			force = self.actuators[item]["Law"]( np.linalg.norm(distance)) * convert_vector_to_unitary(distance)
		else:
			force =  np.zeros(2)
		return force

	def get_force(self, index, law):
		return law(self.get_actuators_distance(index))

	def get_travel (self):
		if self.barras[0].get_length() + self.barras[1].get_length() < self.barras[2].get_length() + self.barras[3].get_length():
			self.__alfa0 = 0
			self.__alfa1 = 2 * pi
		else:
			iterseciton= IntersectionCircleCircle(
				self.barras[1].get_start_point(),
				self.barras[1].get_length(),
				self.barras[2].get_start_point(),
				self.barras[2].get_length() + self.barras[3].get_length()
			)
			vector0, vector1 = iterseciton.get_polar_intersection()
			angulo0 = vector0[1] 
			angulo1 = vector1[1] 
			#print(f'Angulos iniciaales {angulo0} , {angulo1}')
			if angulo0 > 0 and angulo1 > 0:
				self.__alfa0 = angulo0 if angulo1 > angulo0 else angulo1
				self.__alfa1 = angulo1 if angulo1 > angulo0 else angulo0
			elif angulo0 < 0 and angulo1 < 0:
				angulo0 = angulo0 + (2 * pi if angulo0 < angulo1 else 0)
				angulo1 = angulo1 + (2 * pi if angulo1 < angulo0 else 0)
				self.__alfa0 = angulo0 if angulo1 > angulo0 else angulo1
				self.__alfa1 = angulo1 if angulo1 > angulo0 else angulo0
			else:
				angulo0 = angulo0 + (2 * pi if angulo0 < 0 else 0)
				angulo1 = angulo1 + (2 * pi if angulo1 < 0 else 0)
				self.__alfa0 = angulo0 if angulo0 < angulo1 else angulo1
				self.__alfa1 = angulo1 if angulo0 < angulo1 else angulo0
			#print(f'Angulos finales {angulo0} , {angulo1}')

	def get_start_travel(self):
		return self.__alfa0

	def get_end_travel(self):
		return self.__alfa1

	def calculo_dinamico(self):
		#Defino las ecuaciones de resolución:
		#Barra0,Barra1,Barra2(Entrada),Barra3(barra fija)
		#Descriocion de cada incógnita
  		#R00x,R00y,R01x,R01y,R10x,R10y,R11x,R11y,R20x,R20y,R21x,R21y,R30x,R30y,R31x,R31y,F2x,F2y,F3x,F3y,M0,M1,M2,M3,SF0x,SF0y+Peso,SF1x,SF1y+Peso,SF2x,SF2y+Peso,SF3x,SF3y
		#Ecuacion a resolver es del tipo AX = B
		#Matriz B 
		B = np.zeros(13)
		#Matriz A
		A = np.zeros((13,14))
		#Ecuación 0 y 1 R11 + R20 =0
#		print (Reacciones.Fx)
		A[0,Reacciones.R11x.value] =1
		A[0,Reacciones.R30x.value] =1
		A[1,Reacciones.R11y.value] =1
		A[1,Reacciones.R30y.value] =1
		#Ecuación 1 y 2 R21 + R30 = 0
		A[2,Reacciones.R21x.value] =1
		A[2,Reacciones.R31x.value] =1
		A[3,Reacciones.R21y.value] =1
		A[3,Reacciones.R31y.value] =1
		for barra in range(1, 4):
		#Equilibrio barra1
			equilibrio_Fx = 3 * barra + 1 
			equilibrio_Fy = 3 * barra + 2 
			equilibrio_M = 3 * barra + 3 
			A[equilibrio_Fx, Reacciones.R10x.value] =1
			A[equilibrio_Fx, Reacciones.R11x.value] =1

			A[equilibrio_Fy, Reacciones.R10y.value] =1
			A[equilibrio_Fy, Reacciones.R11y.value] =1
			distance_to_cir = self.barras[barra].get_end_point() - self.barras[barra].get_cir()
			A[equilibrio_M, Reacciones.R10x.value] = distance_to_cir[0]
			A[equilibrio_M, Reacciones.R11y.value] = distance_to_cir[1]

			B[equilibrio_Fy] = self.barras[barra].get_weight()
			for actuator_nr in range(0, len(self.actuators)):
				if self.actuators[actuator_nr]["Input"]:
					barra1_nr = self.actuators[actuator_nr]["Barra1"]["Barra"]
					barra1_actuator_nr = self.actuators[actuator_nr]["Barra1"]["Actuator"]
					barra2_nr = self.actuators[actuator_nr]["Barra2"]["Barra"]
					barra2_actuator_nr = self.actuators[actuator_nr]["Barra2"]["Actuator"]
					if barra1_nr == barra or barra2_nr == barra:
						distance =self.get_actuators_distance(actuator_nr)
						my_force = self.actuators[actuator_nr]["Law"](np.linalg.norm(distance)) if not self.actuators[actuator_nr]["Law"] is None else np.zeros(2)
						vector_force = distance / np.linalg.norm(distance) * my_force
      					#my_law = self.actuators[item]["Law"]
						#my_force = self.get_actuator_force(my_barra)
						B[equilibrio_Fx] = -vector_force[0]
						B[equilibrio_Fy] = -vector_force[1]
					if barra1_nr == barra:
						barra1_actuator_point =  self.barras[barra1_nr].get_actuator_point(barra1_actuator_nr)
						distance_to_cir = DistancePointLine(barra1_actuator_point,vector_force,self.barras[barra].get_cir())
						vector_distance = distance_to_cir.get_vector_distance()
						cross_prod = np.cross(vector_distance, vector_force)
						if cross_prod != 0:
							B[equilibrio_M] = cross_prod / abs(cross_prod) * np.linalg.norm(vector_distance) * np.linalg.norm(vector_force)
					if barra2_nr == barra:
						barra2_actuator_point =  self.barras[barra1_nr].get_actuator_point(barra2_actuator_nr)
						distance_to_cir = DistancePointLine(barra2_actuator_point,vector_force,self.barras[barra].get_cir())
						cross_prod = np.cross(vector_distance, vector_force)
						if cross_prod != 0:
							B[equilibrio_M] += cross_prod / abs(cross_prod) * np.linalg.norm(vector_distance) * np.linalg.norm(vector_force)
    
		#B[6] += self.barras[1].get_start_torque() + self.barras[barra].get_end_torque()
  
		A[10, Reacciones.Fx.value] =1
		A[11, Reacciones.Fy.value] =1
		A[12, Reacciones.Fx.value] = abs(self.barras[3].get_end_point()[0] - self.barras[3].get_cir()[0])
		A[12, Reacciones.Fy.value] = abs(self.barras[3].get_end_point()[1] - self.barras[3].get_cir()[1])
		Reactions = np.linalg.lstsq(A, B,rcond=0.01)
#		print ("Reaccion R10x =", Reactions[0][Reacciones.R10x.value])
#		print ("Reaccion R10y =", Reactions[0][Reacciones.R10y.value])
#		print ("Reaccion R11x =", Reactions[0][Reacciones.R11x.value])
#		print ("Reaccion R11y =", Reactions[0][Reacciones.R11y.value])
#		print ("Reaccion R20x =", Reactions[0][Reacciones.R20x.value])
#		print ("Reaccion R20y =", Reactions[0][Reacciones.R20y.value])
#		print ("Reaccion R21x =", Reactions[0][Reacciones.R21x.value])
#		print ("Reaccion R21y =", Reactions[0][Reacciones.R21y.value])
#		print ("Reaccion R30x =", Reactions[0][Reacciones.R30x.value])
#		print ("Reaccion R30y =", Reactions[0][Reacciones.R30y.value])
#		print ("Reaccion R31x =", Reactions[0][Reacciones.R31x.value])
#		print ("Reaccion R31y =", Reactions[0][Reacciones.R31y.value])
#		print ("Reaccion Fx =", Reactions[0][Reacciones.Fx.value])
#		print ("Reaccion Fy =", Reactions[0][Reacciones.Fy.value])

		return Reactions[0]