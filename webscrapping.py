# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


''' WEBSCRAPPING PRIX DU PETROLE BRUT DEPUIS 2000 '''



from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
from decimal import Decimal
from selenium.webdriver.support.select import Select


# Export_Path = "./exports/data_c-oil"

options = webdriver.ChromeOptions()
#options.add_argument('headless')
driver = webdriver.Chrome()
driver.get("https://www.investing.com/commodities/crude-oil-historical-data")
time.sleep(2)

# ---------------------------------






# récupérer la data (historique des prix de pétrole)

soup = BeautifulSoup(driver.page_source, 'html.parser')
current_price = soup.find("div", {"data-test": "instrument-price-last"})

if current_price:
    price = Decimal(current_price.get_text(strip=True)) 
    print("Prix actuel du pétrole brut :", price, " (USD)")
else:
    print("Prix non trouvé")


time.sleep(2)



# --------------------------------



# Trouver l'élément contenant le timeframe
timeframe_element = soup.find("div", class_="flex flex-1 flex-col justify-center text-sm leading-5 text-[#333]")

# Correspondance des mois en français
months_translation = {
    "01": "janvier", "02": "février", "03": "mars", "04": "avril",
    "05": "mai", "06": "juin", "07": "juillet", "08": "août",
    "09": "septembre", "10": "octobre", "11": "novembre", "12": "décembre"
}

# Récupérer le contenu de l'élément
if timeframe_element:
    timeframe_text = timeframe_element.text.strip()
    # Diviser le contenu en fonction du séparateur "-"
    dates = timeframe_text.split(" - ")
    if len(dates) == 2:
        date_start, date_end = dates
        # Séparer les parties de la date
        start_parts = date_start.split("/")
        end_parts = date_end.split("/")
        # Traduire les mois en français
        date_start_fr = f"{start_parts[1]} {months_translation[start_parts[0]]} {start_parts[2]}"
        date_end_fr = f"{end_parts[1]} {months_translation[end_parts[0]]} {end_parts[2]}"
        print("Observations des prix réalisée du", date_start_fr, "au", date_end_fr)
    else:
        print("Format de date incorrect")
else:
    print("Timeframe non trouvé")
    
    
time.sleep(2)      

    
# ---------------------------------


  

rows = soup.findAll("tr")
dataset = []

# Parcourir chaque ligne du tableau
for row in rows:
    # Trouver tous les éléments td dans la ligne
    cells = row.find_all("td")
    # Extraire le texte de chaque cellule et l'ajouter à la liste du dataset
    row_data = [cell.get_text(strip=True) for cell in cells]
    dataset.append(row_data)

print(dataset)

driver.quit()


