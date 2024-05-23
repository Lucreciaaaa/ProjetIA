# -*- coding: utf-8 -*-
"""


@author: lfodo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, r2_score


# Fonction de conversion du volume
def convert_volume(volume):
    if isinstance(volume, str):
        if 'K' in volume:
            return float(volume.replace('K', '')) * 1_000
        elif 'M' in volume:
            return float(volume.replace('M', '')) * 1_000_000
    return float(volume)


# Chargement des données
data = pd.read_csv('CrudeOil-05_24.csv')

# Prétraitement des données
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Day_of_week'] = data['Date'].dt.dayofweek
data.drop(columns=['Date'], inplace=True)
data['Vol.'] = data['Vol.'].apply(convert_volume)  # Conversion du volume
data['Change %'] = data['Change %'].str.replace('%', '').astype(float)  # Conversion du pourcentage
data.fillna(data.median(), inplace=True)

# Séparation des features et de la cible
X = data[['Open', 'High', 'Low', 'Vol.', 'Change %', 'Year', 'Month', 'Day', 'Day_of_week']]
y = data['Price']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mise à l'échelle des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Préparation des données pour LSTM (reshape en 3D)
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Construction du modèle LSTM
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)  # Changer l'unité de cette couche à 1 pour correspondre à la dimension de la cible
])


# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Prédiction sur l'ensemble de test
y_pred = model.predict(X_test_reshaped)
y_pred = scaler.inverse_transform(y_pred).flatten()

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Affichage des métriques
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')

# Visualisation de l'évolution de la perte (loss) pendant l'entraînement
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

