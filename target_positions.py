import numpy as np
from math import cos, sin, radians
# Cargar el modelo
def target_positions(steps):
    data = np.zeros((steps,2))  # Nuevos datos a predecir
    for value in  range (0, steps):
        alfa = radians(70 + 20 / steps * value)
        data[value] = np.array([-28.442,-99.352]) + 196 * np.array([cos(alfa), sin(alfa)])
        new_data = np.zeros(steps * 2)
        new_data[:steps] = data[:,0]
        new_data[steps:] = data[:,1]
    return new_data