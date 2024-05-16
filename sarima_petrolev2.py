# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from itertools import product

# Charger les données
df_coil_1 = pd.read_csv("CrudeOil-94_14.csv")
df_coil_2 = pd.read_csv("CrudeOil_14-24.csv")

# Sélectionner les colonnes 'Date' et 'Price'
df_coil_1 = df_coil_1[['Date','Price']]
df_coil_2 = df_coil_2[['Date','Price']]

# Inverser les lignes pour mettre les dates dans l'ordre chronologique
df_coil_1 = df_coil_1.iloc[::-1]
df_coil_2 = df_coil_2.iloc[::-1]

# Réinitialiser les index
df_coil_1.reset_index(drop=True, inplace=True)
df_coil_2.reset_index(drop=True, inplace=True)

# Concaténer les deux DataFrames
df_coil_combined = pd.concat([df_coil_1, df_coil_2])

# Correction du format de la date
df_coil_combined['Date'] = pd.to_datetime(df_coil_combined['Date'], format='%m/%d/%Y')

# Définir la colonne 'Date' comme index
df_coil_combined.set_index('Date', inplace=True)

# Supprimer les valeurs manquantes
df_coil_combined.dropna(inplace=True)

# Visualiser les premières lignes du DataFrame combiné
print(df_coil_combined.head())

# Tracer la série temporelle
plt.figure(figsize=(10, 6))
plt.plot(df_coil_combined.index, df_coil_combined['Price'], label='Prix')
plt.title('Prix du pétrole brut combiné')
plt.xlabel('Date')
plt.ylabel('Prix')
plt.legend()
plt.show()

# Plot de l'autocorrélation et de la corrélation partielle
plot_acf(df_coil_combined['Price'], lags=30)
plot_pacf(df_coil_combined['Price'], lags=30)
plt.show()

# Décomposition saisonnière
result = seasonal_decompose(df_coil_combined['Price'], model='additive', period=365)  # Spécifiez la période si connue
result.plot()
plt.show()

# Calculer la taille des ensembles de train et de test
total_length = len(df_coil_combined)
train_length = int(total_length * 0.85)

# Diviser les données en ensembles de train et de test
train_data = df_coil_combined['Price'][:train_length]
test_data = df_coil_combined['Price'][train_length:]

# Réinitialisation des index avec une fréquence mensuelle
train_data = train_data.asfreq('MS')
test_data = test_data.asfreq('MS')

# Définir les ordres à tester
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)
seasonal_pdq_values = [(x[0], x[1], x[2], 12) for x in product(range(0, 3), range(0, 2), range(0, 3))]

best_aic = np.inf
best_order = None
best_seasonal_order = None

for p in p_values:
    for d in d_values:
        for q in q_values:
            for seasonal_pdq in seasonal_pdq_values:
                try:
                    model = SARIMAX(train_data, order=(p, d, q), seasonal_order=seasonal_pdq)
                    results = model.fit()
                    if results.aic < best_aic:
                        best_aic = results.aic
                        best_order = (p, d, q)
                        best_seasonal_order = seasonal_pdq
                except:
                    continue

print("Best SARIMA order:", best_order)
print("Best seasonal SARIMA order:", best_seasonal_order)

# Construire le modèle SARIMA
order = (1, 0, 2)
seasonal_order = (2, 0, 2, 12)
model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
results = model.fit()

# Prédictions
predictions = results.predict(start=test_data.index[0], end=test_data.index[-1])

# Réinitialisation de l'index des prédictions avec une fréquence mensuelle
predictions = predictions.asfreq('MS')

# Visualisation
plt.figure(figsize=(10, 6))
plt.plot(train_data.index, train_data, label='Train')
plt.plot(test_data.index, test_data, label='Test')
plt.plot(predictions.index, predictions, label='Predictions')
plt.title('Prix du pétrole brut avec prédictions')
plt.xlabel('Date')
plt.ylabel('Prix')
plt.legend()
plt.show()


