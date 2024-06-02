# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from itertools import product

# Chargement des données
data = pd.read_csv('cac40_04-23.csv')

# conversion de la colonne Date
data['Date'] = pd.to_datetime(data['Date'])

# Extraction des colonnes 'Date' et 'Price'
data = data[['Date', 'Price']]
data['Price'] = data['Price'].str.replace(',', '').str.replace(' ', '').astype(float)

# Tri par date
data = data.sort_values(by='Date', ascending=True)

# Réindexation
data.reset_index(drop=True, inplace=True)

# Affichage des valeurs manquantes
print("Nombre de valeurs manquantes avant l'imputation :")
print(data.isnull().sum())

# Imputation des valeurs manquantes (en utilisant la médiane dans ce cas)
data.fillna(data['Price'].median(), inplace=True)

print("\nNombre de valeurs manquantes après l'imputation :")
print(data.isnull().sum())

print("\nEtat du dataset après l'imputation :")

# Fonction pour évaluer un modèle ARIMA avec un ensemble de paramètres donné
def evaluate_arima_model(data, order):
    # Division des données en ensembles d'entraînement et de test
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]
    
    # Entraînement du modèle ARIMA
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()
    
    # Prédictions
    y_pred = model_fit.forecast(steps=len(test_data))
    
    # Calcul de l'erreur quadratique moyenne (MSE)
    mse = mean_squared_error(test_data, y_pred)
    return mse

# Différenciation de la série temporelle
diff_data = data['Price'].diff().dropna()

# Paramètres à tester pour ARIMA (sans la composante saisonnière)
p = range(0, 3)  # Ordre AR
d = range(0, 2)  # Ordre de différenciation
q = range(0, 3)  # Ordre MA

# Recherche des meilleurs paramètres ARIMA sur la série différenciée
best_mse = float('inf')
best_params = None
for order in product(p, d, q):
    try:
        mse = evaluate_arima_model(diff_data, order)
        if mse < best_mse:
            best_mse = mse
            best_params = order
    except:
        continue

print("Meilleurs paramètres ARIMA:", best_params)

# Entraînement du modèle ARIMA avec les meilleurs paramètres sur la série différenciée
model = ARIMA(data['Price'], order=best_params)
model_fit = model.fit()

# Prédictions sur l'ensemble de test
y_pred_diff = model_fit.forecast(steps=len(diff_data))

# Reconstruction des prédictions sur la série d'origine en inversant la différenciation
y_pred = np.cumsum(y_pred_diff) + data['Price'].iloc[-1]

# Calcul des métriques
mse = mean_squared_error(data['Price'][-len(diff_data):], y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Visualisation des prédictions
plt.plot(data['Date'][-len(diff_data):], data['Price'][-len(diff_data):], label='Observé')
plt.plot(data['Date'][-len(diff_data):], y_pred, label='Prédit')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Prédictions ARIMA vs Observations')
plt.legend()
plt.show()
