# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# Chargement des données
co_data = pd.read_csv('crude-oil-price-83_24.csv', parse_dates=['date'], index_col='date', usecols=['date', 'price'])
co_data = co_data.dropna()

# Division des données en fonction de la taille
X = co_data.drop(columns=['price'])  # Variables explicatives
y = co_data['price']  # Variable cible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Prétraitement des données
class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

# Définition du pipeline
pipeline = Pipeline([
    ('scaler', DataFrameScaler()),
    ('ridge', Ridge())
])

# Grille des hyperparamètres à rechercher
param_grid = {
    'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]  # Valeurs à tester pour alpha
}

# Recherche des meilleurs hyperparamètres avec validation croisée
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Meilleur modèle
best_model = grid_search.best_estimator_

# Prédictions sur l'ensemble de test
y_pred = best_model.predict(X_test)

# Calcul des métriques d'évaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Calcul de la MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)

# Affichage des performances du modèle
print('Best Mean Squared Error (MSE):', mse)
print('Best Root Mean Squared Error (RMSE):', rmse)
print('Best R-squared (R2):', r2)
print('Best Mean Absolute Percentage Error (MAPE):', mape)

# Affichage des meilleurs hyperparamètres
print('Best hyperparameters:', grid_search.best_params_)
