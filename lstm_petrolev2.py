# -*- coding: utf-8 -*-
"""


@author: lfodo
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error



# Chargement des données
co_data = pd.read_csv('crude-oil-price-83_24.csv', parse_dates=['date'], index_col='date', usecols=['date', 'price'])
co_data = co_data.dropna()
co_data.sort_index(inplace=True)

# Normalisation des données
sc = MinMaxScaler(feature_range=(0, 1))
co_data['price'] = sc.fit_transform(co_data[['price']])

# Division des données en ensembles d'entraînement et de test
train_size = int(len(co_data) * 0.7)
train_data = co_data.iloc[:train_size]
test_data = co_data.iloc[train_size:]

# créer les séquences d'entraînement
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

# Paramètres
look_back = 90
batch_size = 15
epochs = 20

# ensembles d'entraînement et test
X_train, Y_train = create_dataset(train_data.values, look_back)
X_test, Y_test = create_dataset(test_data.values, look_back)

# Redimensionner les données pour le LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Modèle LSTM
def create_model(activation='relu', dropout_rate=0.1, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units=60, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=60, return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=60))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1, activation=activation))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# grille de recherche d'hyperparamètres
param_grid = {
    'activation': ['relu', 'tanh'],
    'dropout_rate': [0.1, 0.2],
    'optimizer': ['adam', 'rmsprop']
}

# VC et recherche des meilleurs paramètres
cv = TimeSeriesSplit(n_splits=5)
best_params = None
best_model = None
best_score = float('inf')


for activation in param_grid['activation']:
    for dropout_rate in param_grid['dropout_rate']:
        for optimizer in param_grid['optimizer']:
            print(f"Training with activation={activation}, dropout_rate={dropout_rate}, optimizer={optimizer}")
            model = create_model(activation=activation, dropout_rate=dropout_rate, optimizer=optimizer)
            history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.2)
            score = model.evaluate(X_test, Y_test, verbose=0)
            print("Validation loss:", score)
            if score < best_score:
                best_score = score
                best_params = {'activation': activation, 'dropout_rate': dropout_rate, 'optimizer': optimizer}
                best_model = model

# Meilleurs hyperparamètres
print("Meilleurs hyperparamètres:", best_params)

# Prédictions sur l'ensemble de test
test_predict = best_model.predict(X_test)

# Désnormalisation des prédictions et des vraies valeurs
test_predict = sc.inverse_transform(test_predict)
Y_test = sc.inverse_transform(Y_test)

# Calcul des métriques
mae_test = mean_absolute_error(Y_test, test_predict)
rmse_test = np.sqrt(mean_squared_error(Y_test, test_predict))
r2_test = r2_score(Y_test, test_predict)
medae_test = median_absolute_error(Y_test, test_predict)

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_test = mean_absolute_percentage_error(Y_test, test_predict)


print('Test Mean Absolute Error:', mae_test)
print('Test Root Mean Squared Error:', rmse_test)
print('Test R-squared:', r2_test)
print('Test Median Absolute Error:', medae_test)
print('Test Mean Absolute Percentage Error:', mape_test)
      
# Tracé de la perte sur les ensembles d'entraînement et de validation
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(loc='upper right')
plt.show()

# Tracé des prédictions et des vraies valeurs
plt.figure(figsize=(8, 4))
plt.plot(Y_test[:180], marker='.', label="Actual")
plt.plot(test_predict[:180], 'r', label="Prediction")
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Price', size=15)
plt.xlabel('Time Step', size=15)
plt.legend(fontsize=15)
plt.show()
