# -*- coding: utf-8 -*-
"""


@author: lfodo
"""

# dataset de données journalières sur 19 ans
# predictions avec prise en compte de la saisonnalité


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from webscrapping import get_price  # fichier python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from decimal import Decimal
from selenium.webdriver.support.select import Select


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




''' PREDICTIONS FUTURES '''

# Récupérer le prix actuel du pétrole brut
current_oil_price = get_price()
if current_oil_price:
    print("Prix actuel du pétrole brut :", current_oil_price, " (USD)")
else:
    print("Prix non trouvé")
    
    
    
    
    
    

def generate_features_for_date(date):
    # Ici, vous pouvez implémenter la logique pour générer les caractéristiques
    # en fonction de la date donnée. Vous pouvez utiliser des données historiques
    # ou tout autre paramètre pertinent pour créer les caractéristiques.

    # Par exemple, pour cet exemple, je vais simplement créer des caractéristiques
    # factices pour illustrer le processus.
    features = {
        'Open': 70.0,  # Prix d'ouverture fictif
        'High': 72.0,  # Prix le plus haut fictif
        'Low': 68.0,   # Prix le plus bas fictif
        'Vol.': 500000.0,  # Volume fictif
        'Change %': -2.0,  # Variation fictive du prix en pourcentage
        'Year': date.year,
        'Month': date.month,
        'Day': date.day,
        'Day_of_week': date.weekday()  # Jour de la semaine (0 pour lundi, 6 pour dimanche)
    }
    return features

# Exemple d'utilisation de la fonction pour générer les caractéristiques pour le 23 mai 2024
date_23_may_2024 = pd.Timestamp('2024-05-23')
features_for_23_may_2024 = generate_features_for_date(date_23_may_2024)
print("Caractéristiques pour le 23 mai 2024 :", features_for_23_may_2024)


# Prédiction avec le modèle RandomForestRegressor
predicted_price_23_may_2024 = rf.predict([features_for_23_may_2024])

# Affichage de la prédiction
print("Prédiction du prix du pétrole brut pour le 23 mai 2024 :", predicted_price_23_may_2024)

# Comparaison avec le prix actuel du pétrole
if current_oil_price:
    print("Prix actuel du pétrole brut :", current_oil_price, " (USD)")
    price_difference = predicted_price_23_may_2024 - current_oil_price
    print("Différence entre la prédiction et le prix actuel :", price_difference)
else:
    print("Prix actuel non trouvé")

    

