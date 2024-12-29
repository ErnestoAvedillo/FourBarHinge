from calculo_bisagra import Bisagra
from algebra.barra import Barra
from algebra.uniones import Union
from muelle_compresion import MuelleCompresion
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def generate_table_angles(max_rotation_angle:int, steps:int):
	angles = np.zeros(steps)
	for pos in range(1, len(angles) + 1):
		angles[pos - 1] = max_rotation_angle * pos / steps
	return angles

def generate_data(coord_ini:np.array, high:int, wide:int ,steps:int):
	array_of_coordinates = np.zeros( (steps * steps , 2))
	position = 0
	for pos_x in range (0, steps):
		for pos_y in range (0, steps):
			coord_ini_x = coord_ini[0] + wide * pos_x / steps
			coord_ini_y = coord_ini[1] + high * pos_y / steps
			array_of_coordinates [position] = [coord_ini_x, coord_ini_y]
			position = position + 1
	return array_of_coordinates

def array_of_positions (mi_bisagra:Bisagra, actuator, steps:int, angle:np.array, degrees = True):
	Fpositions = np.zeros((steps,2))
	for pos in range(0 ,steps):
		mi_bisagra.rotate(pos / steps * angle, degrees = degrees)
		position =  mi_bisagra.barras[actuator["Barra"]].get_actuator_point(actuator["actuator"])
		Fpositions[pos] = position
	return Fpositions
		
steps=5
curve_precision = 10
max_rotation=30
Point_2_arr = generate_data(np.array([0,50]), 50 , 50, steps)
Point_3_arr = generate_data(np.array([-50,100]), 50 , 50, steps)
Point_4_arr = generate_data(np.array([-50,0]), 50 , 40, steps)
Point_F_arr = generate_data(np.array([0,100]), 50 , 50, steps)
#test_test = np.zeros((int(math.pow(steps,4*2)),curve_precision * 2 + 9))
test_test = np.zeros((1,curve_precision * 2 + 9))
df = pd.DataFrame(test_test)

muelle = MuelleCompresion (65, 5)
muelle.add_constant(35, 10)

for point_2 in Point_2_arr:
	df2 = pd.DataFrame(test_test)
	df = pd.concat([df, df2] , ignore_index=True)
	df.to_csv('posiciones.csv', index=False)
	print (test_test)
	test_test = np.zeros((1,curve_precision * 2 + 9))
	input("press enter")
	for point_3 in Point_3_arr:
		#fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
		for point_4 in Point_4_arr:
			for point_f in Point_F_arr:
				#fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
				#fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))
				A = Barra(Union(np.array([0, 0])),Union(np.array(point_2)))  # Fijo en el origen
				B = Barra(Union(np.array(point_3)),Union(np.array(point_4)))  # Fijo en el origen
				mi_bisagra = Bisagra(A,B)
				mi_bisagra.barras[3].set_actuator_point(Union(np.array(point_f)))
				mi_bisagra.barras[0].set_actuator_point(Union(np.array([1000, 150])))
				mi_bisagra.define_actuators({"Barra1":{"Barra":1,"Actuator":0}, "Barra2":{"Barra":3,"Actuator":0}, "Input":True, "Law": muelle.get_force})
				if mi_bisagra.barras[1].get_angle() < mi_bisagra.get_start_travel():
					next
				else:
					max_rotation =  mi_bisagra.get_end_travel() - max(mi_bisagra.barras[1].get_angle(), mi_bisagra.get_start_travel())
					#print (f'rotación :{max_rotation} =  {mi_bisagra.get_end_travel()} - max({mi_bisagra.barras[1].get_angle()}, {mi_bisagra.get_start_travel()})')
				angles = generate_table_angles(max_rotation,steps)
				for angle in angles:
					try:
						F_positions = array_of_positions(mi_bisagra,{"Barra":3,"actuator":0},curve_precision, angle, degrees = False)
						#print(F_positions)
					except:
						print(f'Error with geometry {point_2[0]},{point_2[1]},{point_3[0]},{point_3[1]},{point_4[0]},{point_4[1]},{point_f[0]},{point_f[1]},{angle}')
						next
					data = np.zeros((1,curve_precision * 2 + 9))
					data[0, :curve_precision] = F_positions[:,0].T
					data[0, curve_precision:-9] = F_positions[:,1].T
					data[0, -9:] = [point_2[0],point_2[1],point_3[0],point_3[1],point_4[0],point_4[1],point_f[0],point_f[1],angle]
					#print(data)
					test_test = np.append(test_test , data, axis = 0)
					print(test_test)
					ax1.plot(F_positions[:,0], F_positions[:,1])
				plt.show()
	print(point_2[0],point_2[1],point_3[0],point_3[1],point_4[0],point_4[1],point_f[0],point_f[1],max_rotation)
				
					
df = pd.DataFrame(test_test[1:,:])
df.to_csv('posiciones.csv', index=False)
print (df)