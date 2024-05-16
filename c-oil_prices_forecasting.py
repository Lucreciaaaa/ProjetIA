# -*- coding: utf-8 -*-
"""


@author: lfodo
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose  
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import statsmodels.api as sm







df_coil = pd.read_csv("CrudeOil-00_24.csv")
df_coil = df_coil[['Date','Price']]
df_coil = df_coil.loc[::-1]
df_coil.reset_index(drop=True, inplace=True)
# Conversion de la colonne 'Date' en format datetime avec le format spécifié
df_coil['Date'] = pd.to_datetime(df_coil['Date'], format='%d/%m/%Y')
df_coil.set_index('Date', inplace=True)
df_coil.isnull().sum()

# graphique
plt.figure(figsize=(10,6))
plt.plot(df_coil.index, df_coil["Price"],color="green")
plt.title("Evolution du prix du pétrole depuis 2000")
plt.xlabel('Date')
plt.ylabel('Prix du pétrole')
plt.grid(True)
plt.show  # => série non stationnaire



# -----------------------------------------------------


# Calcul de l'ACF et du PACF
acf = sm.tsa.stattools.acf(df_coil["Price"], nlags=40)
pacf = sm.tsa.stattools.pacf(df_coil["Price"], nlags=40)

# Tracé de l'ACF
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.stem(acf)
plt.title("Fonction d'autocorrélation (ACF)")
plt.xlabel('Pas')
plt.ylabel('ACF')
'''
un "lag"(ou pas) de 1 indique la corrélation entre une observation et l'observation
précédente, un "lag" de 2 indique la corrélation entre une observation et
l'observation deux périodes précédentes, et ainsi de suite.
L'axe des ordonnées (vertical) représente les valeurs de l'ACF ou du PACF,
qui mesurent la force de la corrélation entre les observations. Plus la valeur
est proche de 1 (ou -1), plus la corrélation est forte (ou négative), tandis
qu'une valeur proche de 0 indique une corrélation faible ou nulle.
'''

# Tracé du PACF
plt.subplot(212)
plt.stem(pacf)
plt.title("Fonction d'autocorrélation partielle (PACF)")
plt.xlabel('Pas')
plt.ylabel('PACF')

plt.tight_layout()
plt.show()


# -------------------------------------------




''' DIFFERENCIATION (STATIONNARISATION) '''

df_coil_diff = df_coil.diff().dropna()

# Affichage du graphique
plt.figure(figsize=(10, 6))
plt.plot(df_coil_diff.index, df_coil_diff.values, color="blue")
plt.title("Série temporelle différenciée")
plt.xlabel("Date")
plt.ylabel("Différence du prix du pétrole")
plt.grid(True)
plt.show()



# ------------------------------------------


acf_diff = sm.tsa.stattools.acf(df_coil_diff, nlags=40)
pacf_diff = sm.tsa.stattools.pacf(df_coil_diff, nlags=40)

# Tracé de l'ACF et du PACF sur le même graphique
plt.figure(figsize=(12, 6))

plt.stem(acf_diff, linefmt='b-', markerfmt='bo', basefmt=' ', label='ACF')
plt.stem(pacf_diff, linefmt='r-', markerfmt='ro', basefmt=' ', label='PACF')

plt.title("Fonction d'autocorrélation (ACF) et d'autocorrélation partielle (PACF) de la série différenciée")
plt.xlabel('Lag')
plt.ylabel('Corrélation')
plt.legend()

plt.grid(True)
plt.show()

# => ACF : il n'y a plus de lente décroissance donc on peut valider la stationarité
# on s'arrete à une différenciation d'ordre 1 => d = 1
# on obtient p = 2 (PACF) et q = 2



# ---------------------------------------------

''' ESTIMATION DU MODELE '''



# Détermination de la taille de l'ensemble de test
test_size = int(len(df_coil) * 0.2)  # Environ 20% des données pour le test

# Séparation des données en ensembles d'entraînement et de test
train_data = df_coil.iloc[:-test_size]
test_data = df_coil.iloc[-test_size:]

# Vérification des tailles des ensembles d'entraînement et de test
print("Taille de l'ensemble d'entraînement :", len(train_data))
print("Taille de l'ensemble de test :", len(test_data))

model = ARIMA(train_data, order = (2,1,2)) # (p,d,q)
model_fit = model.fit()
print(model_fit.summary())






# MAUVAIS RESULTATS 
