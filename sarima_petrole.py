import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from itertools import product




# ------------------------Charger les données

df_coil = pd.read_csv("CrudeOil-83_24.csv")
df_coil = df_coil[['Date','Price']]
df_coil = df_coil.loc[::-1]
df_coil.reset_index(drop=True, inplace=True)
# Correction du format de la date
df_coil['Date'] = pd.to_datetime(df_coil['Date'], format='%m/%d/%Y')
df_coil.set_index('Date', inplace=True)
print(df_coil.head())



