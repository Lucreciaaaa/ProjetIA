# -*- coding: utf-8 -*-
"""


@author: lfodo
"""

# predictions avec prise en compte de la saisonnalité


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
data['Vol.'] = data['Vol.'].apply(convert_volume)
data['Change %'] = data['Change %'].str.replace('%', '').astype(float)
data.fillna(data.median(), inplace=True)

# Séparation des features et de la cible
X = data[['Open', 'High', 'Low', 'Vol.', 'Change %', 'Year', 'Month', 'Day', 'Day_of_week']]
y = data['Price']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle de forêt aléatoire
rf = RandomForestRegressor(n_estimators=100, random_state=42)  
rf.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
y_pred = rf.predict(X_test)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calcul du Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)

# Affichage des métriques
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')


# Visualisation des prédictions par rapport aux valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_test, color='blue', label='Prédictions vs Réelles')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ligne diagonale')
plt.xlabel('Prédiction')
plt.ylabel('Valeur réelle')
plt.title('Prédiction vs Réalité')
plt.legend()
plt.show()
