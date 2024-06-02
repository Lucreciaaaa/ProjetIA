# -*- coding: utf-8 -*-
"""


@author: lfodo
"""


''' WEBSCRAPPING PRIX DU PETROLE BRUT '''



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
driver.get("https://www.investing.com/commodities/crude-oil")
time.sleep(2)

# ---------------------------------






# récupérer le prix du pétrole actuel

def get_price():
 soup = BeautifulSoup(driver.page_source, 'html.parser')
 current_price = soup.find("div", {"data-test": "instrument-price-last"})

 if current_price:
    price = Decimal(current_price.get_text(strip=True)) 
    print("Prix actuel du pétrole brut :", price, " (USD)")
 else:
    print("Prix non trouvé")


 time.sleep(600)
 driver.quit()










