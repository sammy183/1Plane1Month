# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:49:27 2026

@author: NASSAS
"""

import requests
from bs4 import BeautifulSoup
import random
from urllib.parse import urljoin
from tqdm import tqdm
import json
import webbrowser

def regather_links():
    base_url = "http://richard.ferriere.free.fr/3vues/3vues.html"
    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    
    # collect header links
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#") or href.startswith("mailto"):
            continue
        full_url = urljoin(base_url, href)
        links.append(full_url)
    
    print('Please wait 30s to gather links!')
    # gather all aircraft links
    a_to_z = links[8:-2]
    all_three_views = []
    for mainlink in a_to_z:
        response = requests.get(mainlink)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("#") or href.startswith("mailto"):
                continue
            full_url = urljoin(base_url, href)
            all_three_views.append(full_url)
    
    print(f'Links for {len(all_three_views)} aircraft gathered!')
    file_path = "ThreeViewLinks.json"
    with open(file_path, 'w') as file:
        json.dump(all_three_views, file, indent=4) # 'indent=4' makes the file human-readable

################## READ BELOW ##############################
#%% Uncomment to rescrape links if the website got updated
# regather_links()

#%% Run to get a random link!
file_path = 'ThreeViewLinks.json'
with open(file_path, 'r') as file:
    all_three_views = json.load(file)
random_link = random.choice(all_three_views)

print(f"Random aircraft out of {len(all_three_views)}:")
print(random_link)
webbrowser.open_new_tab(random_link)
