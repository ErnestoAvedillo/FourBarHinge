import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.src.models import Sequential
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from math import cos, sin, radians

# Cargar datos
data = pd.read_csv('posiciones.csv')
df = data.loc[(data!=0).any(axis=1)]
print(df)
# Separar características y etiquetas
X = df.iloc[:, :-9]  # Las primeras 20 columnas son las características
y = df.iloc[:, -9:]  # Las últimas 8 columnas son las etiquetas
print (X)
print (y)
# Normalizar los datos
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Crear el modelo   

model = Sequential()
model.add(Dense(64, activation='elu', input_dim=len(X[1])))
model.add(Dense(48, activation='elu'))
model.add(Dense(32, activation='elu'))
model.add(Dense(16, activation='elu'))
model.add(Dense(9, activation='linear'))  # 8 neuronas de salida

# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer='adam')

# Entrenar el modelo
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
model.save("prediccion_geométrica.keras")
# Hacer una predicción
steps = 10 
data = np.zeros((steps,2))  # Nuevos datos a predecir
for value in  range (0, steps):
    alfa = radians(70 + 20 / steps * value)
    data[value] = np.array([-28.442,-99.352]) + 196 * np.array([cos(alfa), sin(alfa)])
new_data = np.zeros(steps * 2)
new_data[:steps] = data[:,0]
new_data[steps:] = data[:,1]
reshaped  = new_data.reshape(1, 20)
#new_data = scaler.transform(new_data)
prediction = model.predict(reshaped)
print (f'data generated {new_data}')
print (f'data reshaped {reshaped}')
print (f'prediction {prediction}')