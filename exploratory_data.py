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




df = pd.read_csv("commodities_12_22.csv")
#df.head()
#df.info()
#print(df.describe().T)
#print(df.values)

df=df.iloc[::-1]  # inverser les lignes du fichier
#print(df.head())
# Affichage de NaN presque sur toutes les colonnes => bcp de données manquantes

df[["Year", "Month", "Day"]] = df["Date"].str.split("-", expand = True)
#print(df.head())
# création de 2 nouvelles colonnes : Month et Day pour mieux afficher la date

df_copy=df.copy()
da=["Month", "Day"]
for j,i in enumerate(da):
    col_date=df.pop(i)
    df.insert(j+2,i,col_date)
# réorganise les colonnes du DataFrame df pour déplacer les colonnes "Month" et "Day" juste après la colonne "Date"

for i in df:
    if i!="Date":
        df[i]=df[i].astype('float64')
        print(df[i])
    else:
        pass

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
''' 
assure que toutes les colonnes (à l'exception de 'Date') sont 
converties en type float64, et la colonne 'Date' est convertie en
type datetime.
'''




''' EXPLORATION DES DONNEES '''


# argent (silver)
plt.rcParams["figure.figsize"] = (15,8)
sns.lineplot(data=df, x="Date", y="Silver")
plt.xticks(rotation = 45) # pivote les titres des axes pour éviter chevauchements
# ax = plt.gca()
plt.show()


# tracer un graphique linéaire des prix de l'or au fil du temps (10 dernères années)
plt.rcParams["figure.figsize"] = (15,8) # ajuster taille 
sns.lineplot(data=df, x="Date", y="Gold")
plt.xticks(rotation = 45) # pivote les titres des axes pour éviter chevauchements
# ax = plt.gca()
plt.show()

# pétrole brut (crude oil)
plt.rcParams["figure.figsize"] = (15,8)
sns.lineplot(data=df, x="Date", y="Crude Oil")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax = plt.gca()
plt.show()

# pétrole brent (brent oil)
plt.rcParams["figure.figsize"] = (15,8)
sns.lineplot(data=df, x="Date", y="Brent Oil")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax = plt.gca()
plt.show()

# gaz naturel (natural gas)
plt.rcParams["figure.figsize"] = (15,8)
sns.lineplot(data=df, x="Date", y="Natural Gas")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax = plt.gca()
plt.show()

# cuivre (copper)
plt.rcParams["figure.figsize"] = (15,8)
sns.lineplot(data=df, x="Date", y="Copper")
plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
ax = plt.gca()
plt.show()


# corrélation entre les colonnes du dataframe (valeurs variant de -1 à 1)
df.corr()

# matrice de corrélation (sur les 10 dernieres années)
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(method='pearson'),cbar=True,cmap='BuPu',annot=True)
'''Les prix du pétrole brut sont positivement corrélés avec le pétrole Brent, le gaz naturel, l’argent et le cuivre.
Le pétrole Brent et le pétrole brut sont fortement corrélés positivement.
Le pétrole Brent est positivement corrélé au gaz naturel, à l’argent et au cuivre.
Le gaz naturel est positivement corrélé avec le pétrole Brent, le pétrole brut et le cuivre.
Les prix de l’or sont positivement corrélés à ceux de l’argent et du cuivre.
L'argent est positivement corrélé avec le pétrole Brent, le pétrole brut, l'or et le cuivre.
Le cuivre est positivement corrélé avec le pétrole Brent, le pétrole brut, l'or et le cuivre.
Le cuivre est positivement corrélé avec tous les autres.'''

# corrélation sur les 5 dernières années
df_last5=df[df["Year"]>=2017]
df_last5.head()
df_last5[["Crude Oil","Brent Oil","Natural Gas","Gold","Silver","Copper"]].corr(method='pearson')
plt.figure(figsize=(12,12))
sns.heatmap(df_last5[["Crude Oil","Brent Oil","Natural Gas","Gold","Silver","Copper"]].corr(method='pearson'),cbar=True,cmap='BuPu',annot=True)
# Les prix du pétrole brut sont fortement corrélés positivement avec le pétrole Brent et le gaz naturel.



# Comparer chaque matière première avec tous les autres

'''
# relations entre l'or et toutes les autres premières (problème car prend en compte les colonnes Year et Month)
plt.figure(figsize=(35,35))
j=0
for i in enumerate(df): 
    if i[1]!="Gold" and i[1]!=["Date","Month,Year"] :
        j+=1
        plt.subplot(7,7,j+1)
        sns.scatterplot(y=df[i[1]],x=df["Gold"])
        plt.title('Relationship between Gold and '+str(i[1]))
        
        
 # relations entre le pétrole brut et toutes les autres matières premières
plt.figure(figsize=(40,40))
j=0
for i in enumerate(df): 
    if i[1]!="Crude Oil" and i[1]!=("Date","Month","Year"):
        j+=1
        plt.subplot(7,7,j+2)
        sns.scatterplot(y=df[i[1]],x=df["Crude Oil"])
        plt.title('Relationship between Crude Oil and '+str(i[1]))       
 '''       
    


# Amélioration

# relations entre l'or et toutes les autres matières premières
plt.figure(figsize=(35, 35))
# Initialisation du compteur de sous-graphiques
j = 0
# Boucle à travers les colonnes du DataFrame
for i, column in enumerate(df.columns):
    # Si la colonne est "Gold" ou "Year" ou "Month" ou Day ou Date, passer à la prochaine colonne
    if column in ["Gold", "Year", "Month","Day","Date"]:
        continue
    # Incrémentation du compteur de sous-graphiques
    j += 1
    # Crée un sous-graphique
    plt.subplot(7, 7, j)
    # Trace le nuage de points
    sns.scatterplot(x=df["Gold"], y=df[column])
    # Ajoute un titre au sous-graphique
    plt.title('Relationship between Gold and ' + str(column))

plt.show()

  
# # relations entre l'argent et toutes les autres matières premières
plt.figure(figsize=(35, 35))
# Initialisation du compteur de sous-graphiques
j = 0
# Boucle à travers les colonnes du DataFrame
for i, column in enumerate(df.columns):
    # Si la colonne est "Gold" ou "Year" ou "Month" ou Day ou Date, passer à la prochaine colonne
    if column in ["Silver", "Year", "Month","Day","Date"]:
        continue
    # Incrémentation du compteur de sous-graphiques
    j += 1
    # Crée un sous-graphique
    plt.subplot(7, 7, j)
    # Trace le nuage de points
    sns.scatterplot(x=df["Silver"], y=df[column])
    # Ajoute un titre au sous-graphique
    plt.title('Relationship between Silver and ' + str(column))

plt.show()      



# relations entre le pétrole brut et toutes les autres matières premières
plt.figure(figsize=(45, 45))
# Initialisation du compteur de sous-graphiques
j = 0
# Boucle à travers les colonnes du DataFrame
for i, column in enumerate(df.columns):
    # Si la colonne est "Gold" ou "Year" ou "Month" ou Day ou Date, passer à la prochaine colonne
    if column in ["Crude Oil", "Year", "Month","Day","Date"]:
        continue
    # Incrémentation du compteur de sous-graphiques
    j += 1
    # Crée un sous-graphique
    plt.subplot(7, 7, j)
    # Trace le nuage de points
    sns.scatterplot(x=df["Crude Oil"], y=df[column])
    # Ajoute un titre au sous-graphique
    plt.title('Relationship between Crude Oil and ' + str(column))

plt.show()


# relations entre le pétrole brent et toutes les autres matières premières
plt.figure(figsize=(45, 45))
# Initialisation du compteur de sous-graphiques
j = 0
# Boucle à travers les colonnes du DataFrame
for i, column in enumerate(df.columns):
    # Si la colonne est "Gold" ou "Year" ou "Month" ou Day ou Date, passer à la prochaine colonne
    if column in ["Brent Oil", "Year", "Month","Day","Date"]:
        continue
    # Incrémentation du compteur de sous-graphiques
    j += 1
    # Crée un sous-graphique
    plt.subplot(7, 7, j)
    # Trace le nuage de points
    sns.scatterplot(x=df["Brent Oil"], y=df[column])
    # Ajoute un titre au sous-graphique
    plt.title('Relationship between Brent Oil and ' + str(column))

plt.show()



# relations entre le gaz naturel et toutes les autres matières premières
plt.figure(figsize=(45, 45))
# Initialisation du compteur de sous-graphiques
j = 0
# Boucle à travers les colonnes du DataFrame
for i, column in enumerate(df.columns):
    # Si la colonne est "Gold" ou "Year" ou "Month" ou Day ou Date, passer à la prochaine colonne
    if column in ["Natural Gas", "Year", "Month","Day","Date"]:
        continue
    # Incrémentation du compteur de sous-graphiques
    j += 1
    # Crée un sous-graphique
    plt.subplot(7, 7, j)
    # Trace le nuage de points
    sns.scatterplot(x=df["Natural Gas"], y=df[column])
    # Ajoute un titre au sous-graphique
    plt.title('Relationship between Natural Gas and ' + str(column))

plt.show()



# relations entre le cuivre et toutes les autres matières premières
plt.figure(figsize=(35, 35))
# Initialisation du compteur de sous-graphiques
j = 0
# Boucle à travers les colonnes du DataFrame
for i, column in enumerate(df.columns):
    # Si la colonne est "Gold" ou "Year" ou "Month" ou Day ou Date, passer à la prochaine colonne
    if column in ["Copper", "Year", "Month","Day","Date"]:
        continue
    # Incrémentation du compteur de sous-graphiques
    j += 1
    # Crée un sous-graphique
    plt.subplot(7, 7, j)
    # Trace le nuage de points
    sns.scatterplot(x=df["Copper"], y=df[column])
    # Ajoute un titre au sous-graphique
    plt.title('Relationship between Copper and ' + str(column))

plt.show()