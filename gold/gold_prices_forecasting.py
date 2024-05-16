# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


''' IMPORATATION DES LIBRAIRIES '''


import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import plotly.offline as py
import plotly.graph_objs as go
from IPython.display import SVG
import datetime, time
import random
import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import plotly.offline as py
import plotly.graph_objs as go
import keras_tuner as kt
from IPython.display import SVG
import os
import datetime
import time
import random
from keras import backend as K
from keras.regularizers import L1L2
from keras.models import load_model



# pip install tensorflow  (tensorflow pour construire et entrainer rapidement des petits réseaux de neurones)
# pip install keras
# pip install keras-tuner






df = pd.read_csv("commodities_12_22.csv")
df=df.iloc[::-1]  # inverser les lignes du fichier
df[["Year", "Month", "Day"]] = df["Date"].str.split("-", expand = True)
df_copy=df.copy()
da=["Month", "Day"]
for j,i in enumerate(da):
    col_date=df.pop(i)
    df.insert(j+2,i,col_date)
for i in df:
    if i!="Date":
        df[i]=df[i].astype('float64')
        print(df[i])
    else:
        pass
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')




''' NETTOYAGE, PRETRAITEMENT ET FORMATION DES DONNEES '''





# Nous prendrons les données des 500 derniers jours comme données de test
# et le reste sera utilisé pour entraîner le modèle.


df_Gold=df[['Date','Gold']]
# df_Gold.head()
df_Gold=df_Gold.dropna()  # supprimer les valeurs manquantes
# print(df_Gold.info())

prediction_days = 500  # nb de jours à prédire dans le future
# divise les données en un ensemble d'entrainement puis les reconvertit un en tableau à 1 colonne (reshape)
df_train_g= df_Gold['Gold'][:len(df_Gold['Gold'])-prediction_days].values.reshape(-1,1)
# divise les données en un ensemble test 
df_test_g= df_Gold['Gold'][len(df_Gold['Gold'])-prediction_days:].values.reshape(-1,1)
# MinMaxScaler transforme les données en mettant à l'échelle chaque caractéristique dans une plage donnée, par défaut entre 0 et 1.
scaler_train = MinMaxScaler(feature_range=(0, 1))
# applique la normalisation MinMaxScaler à l'ensemble d'entraînement et transforme les données
scaled_train = scaler_train.fit_transform(df_train_g)
# normalisation pour l'ensemble test
scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test_g)





''' 'PREVISION DU PRIX DE L'OR A L'AIDE D'UN LSTM (Réseau de neurones récurrents - RNN) '''




def create_dataset(dataset, look_back=30):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(scaled_train)
testX, testY = create_dataset(scaled_test)
print(trainX.shape)  # résultat : 2212 exemples de séquences, où chaque séquence contient 30 jours de données
print(testX.shape)  #  470 exemples de séquences, où chaque séquence contient également 30 jours de données
# chaque séquence fournit au modèle des informations sur la variation des prix
# de l'or au fil du temps sur une fenêtre de 30 jours.
print(" 151ème séquence de l'ensemble de données test :", testX[150])
print(" 1ère séquence de l'ensemble de données d'entrainement :", trainX[0])

# modifie la forme des données d'entréé
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))





def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

# créé un modèle de réseau LSTM
def model_builder(hp):
    model = Sequential()
    model.add(LSTM(hp.Int('input_unit',min_value=32,max_value=512,step=32), return_sequences=True, input_shape= ( trainX.shape[1], trainX.shape[2]), bias_regularizer = L1L2(0.009, 0.004)))
    for i in range(hp.Int('n_layers', 1, 4)):
        model.add(LSTM(hp.Int(f'lstm_{i}_units',min_value=32,max_value=512,step=32),return_sequences=True))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.05)))
    model.add(Dense(1, activation=hp.Choice('dense_activation',values=['relu', 'sigmoid'],default='relu')))
   
    model.compile(loss=root_mean_squared_error, optimizer='adam',metrics = ['mse'])
    
    return model

# recherche des meilleurs hyperparamètres 
tuner = kt.RandomSearch(model_builder, objective='val_loss', max_trials = 10, executions_per_trial =1,directory = "./gold/")
# les résultats de l'optimisation seront stockés dans le répertoire gold

callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', # Monitor the validation loss
                                                     min_delta=0,    # until it doesn't change (or gets worse)
                                                     patience=5,  # patience > 1 so it continutes if it is not consistently improving
                                                     verbose=0, 
                                                     mode='auto')]

# penser a diminuer le nb d'époques (= itérations) car risque de surapprentissage + temps de chargement trop long
tuner.search(x=trainX, y=trainY, epochs = 200, batch_size =512, validation_data=(testX, testY), callbacks=[callbacks],shuffle = True)


# afficher le récap des essais
tuner.results_summary()

# récupère le meilleur modèle (trial 00)
best_model = tuner.get_best_models(num_models=1)[0]
# réentraine le modèle pour 150 époques
history = best_model.fit(x=trainX, y=trainY, epochs = 150, batch_size =128, validation_data=(testX, testY), shuffle=False, verbose=0)
plt.plot(history.history['loss'], label='train') # loss (entrainement)
plt.plot(history.history['val_loss'], label='test') # loss (validation)
plt.legend()
plt.show()  # erreur en fct du nb d'époques


# pour évaluer visuellement les performances du modèle
predicted_gold_price = best_model.predict(testX)
predicted_gold_price = scaler_test.inverse_transform(predicted_gold_price.reshape(-1, 1))
true = scaler_test.inverse_transform(testY.reshape(-1, 1))
plt.plot(predicted_gold_price, label='predict')
plt.plot(true, label='true')
plt.legend()
plt.show()  # en abscisses, les observations (indices) de l'ensemble test

# evaluation du modèle
scores = best_model.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (best_model.metrics_names[1], scores[1]*100))
best_model.save("models/goldPriceModel.h5")
print("Saved model to disk")

# load json and create model
# load model
model_g = load_model("models/goldPriceModel.h5", custom_objects={'root_mean_squared_error':                   
root_mean_squared_error})
print("dd")
# summarize model.
model_g.summary()
score = model_g.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (model_g.metrics_names[1], score[1]*100))