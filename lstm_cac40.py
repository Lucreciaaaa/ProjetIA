# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Chargement des données
data = pd.read_csv('cac40_04-23.csv', usecols=['Date', 'Price'])

# Conversion de la colonne 'Date' en format datetime et utilisation comme index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data['Price'] = data['Price'].str.replace(',', '').str.replace(' ', '').astype(float)
# Fractionnement des données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# Mise à l'échelle des données
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Préparation des séries temporelles
def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10  # Nombre de pas temporels pour prédire le prochain prix
X_train, y_train = prepare_data(train_scaled, time_steps)
X_test, y_test = prepare_data(test_scaled, time_steps)

# Construction du modèle LSTM
model = Sequential([
    LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    Dense(units=1)
])

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test)

# Affichage de l'évolution de la perte (loss) pendant l'entraînement
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
