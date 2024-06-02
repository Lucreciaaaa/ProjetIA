# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


#Metriques SVR
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score


# Chargement des données
data = pd.read_csv('CAC40_stocks_2010_2021.csv')

# Supprimer les colonnes inutiles
data = data.drop(['CompanyName', 'StockName','Date'], axis=1)

# Séparation des données en entrée (X) et variables cibles (y)
X = data[['High', 'Low', 'Open', 'Volume']] # features
y = data['Close']  # variables cibles

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Normalisation des données 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = SVR()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# métriques
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)

#  resultats
print('SVR')
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
print("Explained Variance Score:", explained_variance)