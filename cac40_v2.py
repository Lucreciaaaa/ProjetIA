# -*- coding: utf-8 -*-
"""


@author: lfodo
"""



import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("CAC40_stocks_2010_2021.csv")



# Afficher les informations sur les types de données et les valeurs manquantes
print(data.info())
print(data.isnull().sum())

# Imputer les données manquantes
data.fillna(method='ffill', inplace=True)

# Convertir la colonne 'Date' en datetime
data['Date'] = pd.to_datetime(data['Date'])

# Visualisation des données
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['High'], label='High')
plt.plot(data['Date'], data['Low'], label='Low')
plt.plot(data['Date'], data['Close'], label='Close')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('CAC 40 High, Low, and Close Prices')
plt.legend()
plt.grid(True)
plt.show()





import matplotlib.pyplot as plt

# Filtrer les données pour ne conserver que les lignes correspondant à chaque entreprise
data_thales = data[data['CompanyName'] == 'Orange']
data_carrefour = data[data['CompanyName'] == 'Carrefour']
data_renault = data[data['CompanyName'] == 'Renault']

# Créer un graphique distinct pour chaque entreprise
plt.figure(figsize=(10, 6),facecolor="beige")

# Plot pour ThalesGroup
plt.plot(data_thales['Date'], data_thales['Close'], label='Close (ThalesGroup)', color='orange')

# Plot pour Carrefour
plt.plot(data_carrefour['Date'], data_carrefour['Low'], label='Low (Carrefour)', color='green')

# Plot pour Renault
plt.plot(data_renault['Date'], data_renault['High'], label='High (Renault)', color='pink')

plt.xlabel('Date')
plt.ylabel('Value')
plt.title("Performance de 3 entreprises françaises du CAC40 depuis 2010")
plt.legend()
plt.grid(True)
plt.show()


print(data.describe())

# Obtenir le nombre de lignes et de colonnes
nb_lignes, nb_colonnes = data.shape

print("Nombre de lignes :", nb_lignes)
print("Nombre de colonnes :", nb_colonnes)



