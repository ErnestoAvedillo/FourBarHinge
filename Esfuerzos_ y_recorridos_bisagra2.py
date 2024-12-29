import xlwings as xw
import numpy as np 
from calculo_bisagra import Bisagra
from algebra.barra import Barra
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from muelle_compresion  import MuelleCompresion
from muelle_traccion import MuelleTraccion
from algebra.uniones import Union
from target_positions import target_positions
from math import radians, sin, cos, pi
@xw.func
def hello(name):
    return 'Hello {0}'.format(name)


# Ángulos iniciales
theta2 = 40  # Ángulo inicial de la barra de entrada

# Función de animación
def animate(i,mi_bisagra:Bisagra, ax):
    global theta2      
    theta = theta2 * i / 100  # Movimiento oscilante de la barra de entrada
    # Actualizar posiciones de las barras
    barras= mi_bisagra.get_complete_geometry(theta, degrees = True)
    #print (mi_bisagra)
    # Limpiar la figura
    ax.cla()
    # Dibujar barras
    A,B = barras[0].get_array_to_plot()
    ax.plot(A,B, 'ko-', lw=2, label="Barra 0 (Fija)")
    A,B = barras[1].get_array_to_plot()
    ax.plot(A,B, 'bo-', lw=2, label="Barra 1 (entrada)")
    A,B = barras[2].get_array_to_plot()
    ax.plot(A,B, 'ro-', lw=2, label="Barra 2 (acoplada)")
    A,B = barras[3].get_array_to_plot()
    ax.plot(A,B, 'go-', lw=2, label="Barra 3 (salida)")
  #  ax.plot([barra1.get_start_point()[0], barra2.get_start_point()[0]],[barra1.get_start_point()[1], barra2.get_start_point()[1]],  'ko-', lw=2, label="Barra fija")
    
    # Configuraciones de la gráfica
    ax.set_xlim(-100, 100)
    ax.set_ylim(-2, 150)
    ax.set_aspect('equal')
    ax.grid(True)
#    if i == 1:
#        ani.event_source.stop()
#        input("press enter to continue")
#        ani.event_source.start() 
    if i == 99:
        for i in range(99, -1, -1):
            theta2 = 90 * i / 100  # Movimiento oscilante de la barra de entrada
            mi_bisagra.rotate(theta2, degrees = True)



def print_forces(mi_bisagra:Bisagra, ax2, ax3, ax4, ax5):
    global theta2      
    steps = 100
    angle = theta2
    angles = np.arange(0, steps) * angle / steps  # Movimiento oscilante de la barra de entrada
    distance_actuator = np.zeros(steps)
    forces = np.zeros(steps)
    foot_forces = np.zeros(steps)
    distance = np.zeros(steps)
    distance_ref = mi_bisagra.get_actuators_distance(1)
    ActuatorPosition = np.zeros((steps,2))
    TargetPosition = target_positions(steps)
    #muelle = MuelleCompresion (100, 3)
    muelle = MuelleCompresion (65, 5)
    muelle.add_constant(35, 10)
    for value in range (0, steps):
        mi_bisagra.rotate(angles[value], degrees = True)
        #forces[value] = mi_bisagra.get_force(0, muelle1.get_force)
        distance_actuator[value] = np.linalg.norm(mi_bisagra.get_actuators_distance(0))
        forces[value] = muelle.get_force(distance_actuator[value])
        distance[value] = np.linalg.norm(mi_bisagra.get_actuators_distance(1) - distance_ref)
        ActuatorPosition[value] = mi_bisagra.barras[3].get_actuator_point(1)
        reacciones = mi_bisagra.calculo_dinamico()
        foot_forces[value] = np.linalg.norm(reacciones[12:13])
    for value in range(steps - 1, -1, -1):
        mi_bisagra.rotate(angles[value], degrees = True)
    ax2.plot(distance, foot_forces)
    ax3.plot(distance, forces)
    ax5.plot(distance, distance_actuator)
    ax4.plot(ActuatorPosition[:,0], ActuatorPosition[:,1])
    ax4.plot(TargetPosition[:steps], TargetPosition[steps:])
#inicio programa proncipal

#creo un muelle
muelle = MuelleCompresion (65, 5)
muelle.add_constant(45, 15)

A = Barra(Union(np.array([0, 0])),Union(np.array([-16, 85.5])))  # Fijo en el origen
#A = Barra(Union(np.array([0, 0])),Union(np.array([10,80])))  # Fijo en el origen
B = Barra(Union(np.array([-20, 15])),Union(np.array([-45, 45])))  # Fijo en el origen
#B = Barra(Union(np.array([-10, 140])),Union(np.array([-18, 40])))  # Fijo en el origen
C = Union(np.array([-40, 110]))
mi_bisagra = Bisagra(A,B)
print(mi_bisagra.get_start_travel() * 180 / pi)
print(mi_bisagra.get_end_travel() * 180 / pi)
mi_bisagra.barras[3].set_actuator_point(Union(np.array([-30, 75])))
mi_bisagra.barras[3].set_actuator_point(Union(np.array([40, 85])))
mi_bisagra.barras[0].set_actuator_point(Union(np.array([1000, 150])))
mi_bisagra.barras[2].set_actuator_point(Union(np.array([-0, 20])))
mi_bisagra.barras[1].set_actuator_point(Union(np.array([-15,70])))
#mi_bisagra.barras[1].set_actuator_point(Union(np.array([-40,30])))
#mi_bisagra.barras[2].set_actuator_point(Union(np.array([60, 60])))
#mi_bisagra.barras[2].set_actuator_point(Union(np.array([22.7, 69.37])))

#mi_bisagra.define_actuators(mi_bisagra.barras[1].get_actuator(0),mi_bisagra.barras[3].get_actuator(0))
#mi_bisagra.define_actuators(mi_bisagra.barras[0].get_actuator(1),mi_bisagra.barras[3].get_actuator(1))
mi_bisagra.define_actuators({"Barra1":{"Barra":1,"Actuator":0}, "Barra2":{"Barra":2,"Actuator":0}, "Input":True, "Law": muelle.get_force})
mi_bisagra.define_actuators({"Barra1":{"Barra":0,"Actuator":0}, "Barra2":{"Barra":3,"Actuator":1}, "Input":False, "Law":None})


# Configurar la figura
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 8))

print_forces(mi_bisagra, ax3, ax4, ax5, ax6)
# Crear la animación
animate(0, mi_bisagra, ax2)
ani = FuncAnimation(fig, animate, frames=100, interval=90, fargs=(mi_bisagra,ax1))

# Mostrar la animación
plt.show()

"""
Ángulo : ==  0.0
Barra1: Start [0 0], End [-16.   85.5]
Barra2: Start [30.7 26.6], End [ 14.68 112.14]

Ángulo : ==  0.0
Barra1: Start [0 0], End [-66.82796301  55.68009842]
Barra2: Start [30.7 26.6], End [-36.14930816  82.32164749]

"""