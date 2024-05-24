# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# dataset de données journalières sur 19 ans
# la date n'est pas prise en compte pour les predictions

''' PRETRAITEMENT '''


data = pd.read_csv('CrudeOil-05_24.csv')


print(data.head())

# conversion de la colonne Date
data['Date'] = pd.to_datetime(data['Date'])


data = data.sort_values(by='Date', ascending=True)

# afficher les valeurs de la plus ancienne a la plus recente (avec inversion des indices)
data.reset_index(drop=True, inplace=True)
#print(data.head())


#  Gérer les données manquantes

print("Nombre de valeurs manquantes avant l'imputation :")
print(data.isnull().sum())

# Nettoyer et convertir la colonne 'Vol.' en type numérique
# Fonction pour convertir les valeurs en millions ou milliers en float
def convert_volume(volume):
    if isinstance(volume, str):
        if 'K' in volume:
            return float(volume.replace('K', '')) * 1_000
        elif 'M' in volume:
            return float(volume.replace('M', '')) * 1_000_000
    return float(volume)

data['Vol.'] = data['Vol.'].apply(convert_volume)


print("\nNombre de valeurs manquantes après la conversion de 'Vol.':")
print(data.isnull().sum())

data['Vol.'] = data['Vol.'].fillna(data['Vol.'].median())

print("\nNombre de valeurs manquantes après l'imputation :")
print(data.isnull().sum())


print("\nEtat du dataset après l'imputation :")
#print(data.info())


# -------------------------------------------

# Normaliser les features pour le SVR
# Définir les features et la target
features = ['Open', 'High', 'Low', 'Vol.', 'Change %']
target = 'Price'

# Préparation des données
X = data[features].copy()
y = data[target].copy()

# Convertir 'Change %' en float après avoir retiré le symbole '%'
X['Change %'] = X['Change %'].astype(str)
X['Change %'] = X['Change %'].str.replace('%', '').astype(float)

# Standardiser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)




''' ENTRAINEMENT  '''




# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Vérifier les shapes des ensembles de données
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

# Entraînement du modèle SVR
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)





''' PREDICTION '''



y_pred = svr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
# Calcul du Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(y_test, y_pred)


print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R^2 Score: {r2}')
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')


# visualisation
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_test, color='blue', label='Prédiction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Réalité')
plt.xlabel('Valeurs prédites pour le modèle')
plt.ylabel('Valeurs réelles')
plt.title('Prix du pétrole')
plt.legend()
plt.show()

