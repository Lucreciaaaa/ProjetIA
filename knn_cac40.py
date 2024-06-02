# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score

# Chargement des données
data = pd.read_csv('CAC40_stocks_2010_2021.csv')

# Supprimer les colonnes inutiles
data = data.drop(['CompanyName', 'StockName', 'Date'], axis=1)

# Séparation des données en entrée (X) et variables cibles (y)
X = data[['High', 'Low', 'Open', 'Volume']]  # features
y = data['Close']  # variable cible (convertie en série 1D)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données (si nécessaire)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialisation et entraînement du modèle KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=5) 
model.fit(X_train_scaled, y_train)

# Prédictions
y_pred = model.predict(X_test_scaled)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)

# Calcul du MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    return (abs(y_true - y_pred) / y_true).mean() * 100

mape = mean_absolute_percentage_error(y_test, y_pred)

# Affichage des résultats
print('KNN')
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
print("Explained Variance Score:", explained_variance)
print("Mean Absolute Percentage Error:", mape)




from sklearn.model_selection import GridSearchCV


# hyperparamètres à tester
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}


model = KNeighborsRegressor()

# Recherche par grille avec vc
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)


best_params = grid_search.best_params_
print("Meilleurs hyperparamètres:", best_params)

# Meilleur modèle
best_model = grid_search.best_estimator_

# Prédictions avec le meilleur modèle
y_pred_best = best_model.predict(X_test_scaled)

# Calcul métriques
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)
mae_best = mean_absolute_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)
explained_variance_best = explained_variance_score(y_test, y_pred_best)
mape_best = mean_absolute_percentage_error(y_test, y_pred_best)

# Affichage des résultats avec le meilleur modèle
print("\nRésultats avec les meilleurs hyperparamètres:")
print("Mean Squared Error:", mse_best)
print("Root Mean Squared Error:", rmse_best)
print("Mean Absolute Error:", mae_best)
print("R2 Score:", r2_best)
print("Explained Variance Score:", explained_variance_best)
print("Mean Absolute Percentage Error:", mape_best)


