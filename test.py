# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:48:01 2024

@author: lfodo
"""


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

df_Gold=df[['Date','Gold']]
# df_Gold.head()
df_Gold=df_Gold.dropna()  # supprimer les valeurs manquantes
# print(df_Gold.info())

prediction_days = 500  # nb de jours à prédire dans le future
# divise les données en un ensemble d'entrainement puis les reconvertit un en tableau à 1 colonne (reshape)
df_train_g= df_Gold['Gold'][:len(df_Gold['Gold'])-prediction_days].values.reshape(-1,1)
# divise les données en un ensemble test 
df_test_g= df_Gold['Gold'][len(df_Gold['Gold'])-prediction_days:].values.reshape(-1,1)
scaler_train = MinMaxScaler(feature_range=(0, 1))
# applique la normalisation MinMaxScaler à l'ensemble d'entraînement et transforme les données
scaled_train = scaler_train.fit_transform(df_train_g)
# normalisation pour l'ensemble test
scaler_test = MinMaxScaler(feature_range=(0, 1))
scaled_test = scaler_test.fit_transform(df_test_g)




def create_dataset(dataset, look_back=30):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

trainX, trainY = create_dataset(scaled_train)
testX, testY = create_dataset(scaled_test)

# modifie la forme des données d'entréé
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))





def root_mean_squared_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))


# load model
model_g = load_model("models/goldPriceModel.h5", custom_objects={'root_mean_squared_error':                   
root_mean_squared_error})
print("dd")


score = model_g.evaluate(trainX, trainY, verbose=0)
print("%s: %.2f%%" % (model_g.metrics_names[1], score[1]*100))