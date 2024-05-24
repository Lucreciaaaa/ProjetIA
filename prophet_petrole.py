# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.diagnostics import cross_validation 
from prophet.diagnostics import performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid


# dataset de données mensuelles sur 40 ans
# (prix du pétrol en baril -USD-)



''' GESTION '''



# Chargement des données
co_data = pd.read_csv('crude-oil-price-83_24.csv', parse_dates=['date'])
# print(co_data.head())
# print(co_data.describe())
# print(co_data.isnull().sum()) => il n'y a que 2 valeurs manquantes (colonne percentChange et change) donc on peut juste supprimer 2 lignes
co_data = co_data.dropna()
# print(co_data.isnull().sum())







''' VISUALISATION'''



plt.figure(figsize=(10, 8))
sns.lineplot(data=co_data, x='date', y='price')
plt.title('Prix du pétrole (baril) depuis 1983')
plt.show()


plt.figure(figsize = (10,8))
sns.lineplot(data=co_data, x = 'date', y = 'change')
plt.title('Variations du prix du pétrole')
plt.show()
# La colonne "change" représente la variation du prix du pétrole par rapport à une période de référence : ici, par rapport au mois précédent







''' PREDICTIONS '''


# Renommage des colonnes pour Prophet
co_data = co_data.rename(columns={'date':'ds', 'price':'y'})

# Modèle Prophet
model = Prophet()
model.fit(co_data[['ds','y']])

# Génération des prévisions
future = model.make_future_dataframe(periods=24, freq='M')
forecast = model.predict(future)

# Visualisation des prévisions
fig1 = model.plot(forecast)

# Ajout des métriques de performance
def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

# Calcul des métriques pour les prévisions
y_true = co_data['y']
y_pred = forecast.loc[:len(y_true)-1, 'yhat']
metrics = evaluate_forecast(y_true, y_pred)

# Affichage des métriques
print("Résultats des métriques de performance :")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")

# Optimisation des hyperparamètres
def grid_search_cv(param_grid, df, horizon='30 days', period='15 days'):
    best_params = None
    best_metric = float('inf')
    
    for params in ParameterGrid(param_grid):
        model = Prophet(**params)
        model.fit(df)

        # Cross-validation
        cv_results = cross_validation(model, horizon=horizon, period=period)
        performance = performance_metrics(cv_results, metrics=['mae', 'mse', 'rmse', 'mape'])
        
        # Calcul de la moyenne des métriques sur toutes les périodes de validation
        avg_performance = performance.mean()
        
        if avg_performance['mae'] < best_metric:
            best_metric = avg_performance['mae']
            best_params = params
            
    return best_params, best_metric

# Grille d'hyperparamètres
param_grid = {
    'seasonality_mode': ['additive', 'multiplicative'],
    'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0],
    
}

# Optimisation des hyperparamètres
best_params, best_metric = grid_search_cv(param_grid, co_data)

print("\nMeilleurs hyperparamètres trouvés:", best_params)
print("Meilleure performance (MAE) sur la validation croisée:", best_metric)
