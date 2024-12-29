import tensorflow as tf
import numpy as np
from math import cos, sin, radians
from target_positions import target_positions
# Cargar el modelo
steps = 10
model = tf.keras.models.load_model('prediccion_geométrica.keras')
new_data = target_positions(steps=steps)
rechaped  = new_data.reshape(1, 20)
#new_data = scaler.transform(new_data)
prediction = model.predict(rechaped)
print(prediction)