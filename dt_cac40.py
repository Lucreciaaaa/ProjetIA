# -*- coding: utf-8 -*-
"""


@author: lfodo
"""

#Metriques DecisionTree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
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

# Initialisation et entraînement du modèle DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
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
print('Decision Tree')
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
print("Mean Absolute Percentage Error:", mape)
print("Explained Variance Score:", explained_variance)


from sklearn.model_selection import GridSearchCV

# Définition de la grille des hyperparamètres à tester
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialisation du GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeRegressor(random_state=42),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',  # Métrique à optimiser
                           cv=5,  # Nombre de folds de validation croisée
                           verbose=1,  # Affichage des détails
                           n_jobs=-1)  # Utilisation de tous les cœurs du processeur

# Exécution de la recherche sur grille sur les données d'entraînement
grid_search.fit(X_train_scaled, y_train)

# Affichage des meilleurs hyperparamètres et de la meilleure métrique
print("Meilleurs hyperparamètres:", grid_search.best_params_)
print("Meilleur score (RMSE):", -grid_search.best_score_)  # On prend l'opposé car GridSearchCV minimise la métrique

