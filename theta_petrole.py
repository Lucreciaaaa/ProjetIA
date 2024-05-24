# -*- coding: utf-8 -*-
"""


@author: lfodo
"""



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from statsmodels.tsa.forecasting.theta import ThetaModel
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

# dataset de données mensuelles sur 40 ans



# Chargement des données
co_data = pd.read_csv('crude-oil-price-83_24.csv', parse_dates=['date'])
co_data = co_data.dropna()

''' ENTRAINEMENT / VALIDATION '''

size = 102  
data_train = co_data[:-size] #80% données en entrainement
data_test  = co_data[-size:]

print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")
print()

fig, ax = plt.subplots(figsize=(12, 4))
data_train['price'].plot(ax=ax, label='train')
data_test['price'].plot(ax=ax, label='test')
ax.legend()
fig.autofmt_xdate()
plt.tight_layout()

''' PREDICTIONS '''

def rolling_forecast(co_data: pd.DataFrame, train_len: int, horizon: int, window: int, method: str, period: int) -> list:
    
    # total length & index
    total_len = train_len + horizon
    end_idx = train_len
    
    if method == 'theta':
        pred_theta = []
        
        for i in range(train_len, total_len, window):
            theta_model = ThetaModel(endog=co_data[:i], period=period)
            res = theta_model.fit()
            pred = res.forecast(window)
            pred_theta.extend(pred)
            
        return pred_theta


TRAIN_LEN = len(data_train)
HORIZON = len(data_test)
WINDOW = 102

# Paramètres à tester pour la période de saisonnalité
periods_to_test = [24, 48, 72, 96, 120]

# Initialisation des variables pour stocker les meilleures métriques et paramètres
best_period = None
best_mse = float('inf')
best_rmse = float('inf')
best_mape = float('inf')
best_r2 = float('-inf')

# Recherche de la meilleure période
for period in periods_to_test:
    pred_theta = rolling_forecast(co_data['price'], TRAIN_LEN, HORIZON, WINDOW, 'theta', period)
    test = data_test.copy()
    test.loc[:, 'pred_theta'] = pred_theta
    
    # Calcul des métriques
    mse_theta = mean_squared_error(test['price'], test['pred_theta'])
    rmse_theta = np.sqrt(mse_theta)
    mape_theta = mean_absolute_percentage_error(test['price'], test['pred_theta'])
    r2_theta = r2_score(test['price'], test['pred_theta'])
    
    # Affichage des métriques
    print(f'Period: {period}')
    print(f'Mean Squared Error (MSE) : {mse_theta}')
    print(f'Root Mean Squared Error (RMSE) : {rmse_theta}')
    print(f'Mean Absolute Percentage Error (MAPE) : {mape_theta}%')
    print(f'R2 Score : {r2_theta}')
    print()
    
    # Mise à jour des meilleures métriques et paramètres
    if mse_theta < best_mse:
        best_mse = mse_theta
        best_rmse = rmse_theta
        best_mape = mape_theta
        best_r2 = r2_theta
        best_period = period

# Affichage des meilleures métriques et paramètres
print(f'Best Period: {best_period}')
print(f'Best Mean Squared Error (MSE) : {best_mse}')
print(f'Best Root Mean Squared Error (RMSE) : {best_rmse}')
print(f'Best Mean Absolute Percentage Error (MAPE) : {best_mape}%')
print(f'Best R2 Score : {best_r2}')
