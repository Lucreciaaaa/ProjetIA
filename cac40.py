# -*- coding: utf-8 -*-
"""

@author: lfodo
"""

import pandas as pd
import matplotlib.pyplot as plt

# Fonction de conversion du volume
def convert_volume(volume):
    if isinstance(volume, str):
        if 'K' in volume:
            return float(volume.replace('K', '')) * 1000
        elif 'M' in volume:
            return float(volume.replace('M', '')) * 1000000
    return float(volume)

# Charger les données
data = pd.read_csv("cac40-95_24.csv")

# Convertir la colonne 'Date' en datetime
data['Date'] = pd.to_datetime(data['Date'])

# Convertir les colonnes pertinentes en float
cols_to_convert = ['Price', 'Open', 'High', 'Low']
for col in cols_to_convert:
    data[col] = data[col].str.replace(',', '').astype(float)

# Convertir la colonne 'Vol.' en utilisant la fonction de conversion
data['Vol.'] = data['Vol.'].apply(convert_volume)

# Retirer le signe % pour la colonne 'Change %' avant de la convertir en float
data['Change %'] = data['Change %'].str.rstrip('%').astype(float)

# Imputer les données manquantes en utilisant la méthode ffill (forward fill)
data = data.interpolate(method='ffill')

# Inverser les lignes et leurs index
data = data.iloc[::-1].reset_index(drop=True)

# Tracer la figure pour le Volume
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Vol.']/1000000, label='Volume (Millions)', color="green")
plt.title("Volume d'échanges sur les entreprises du CAC 40 depuis 1995")
plt.ylabel('Volume (Millions)')
plt.xlabel('Date')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Tracer la figure pour le Prix
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], data['Price'], label="Valeur", color="orange")
plt.title("Niveau de l'indice du CAC 40 depuis 1995")
plt.ylabel('Valeur')
plt.xlabel('Date')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Afficher les premières lignes du tableau avec un style
features_table = data.head()
styled_table = features_table.style.set_properties(**{'text-align': 'center'})
styled_table.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
print(styled_table)
