import requests, os
import tiktoken
import numpy as np
from datasets import load_dataset
from bs4 import BeautifulSoup
import re
import random


def get_wikipedia_text(url):
    # Send a GET request to the Wikipedia page
    response = requests.get(url)
    if response.status_code != 200:
        return None

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the main content div
    content = soup.find('div', {'id': 'bodyContent'})

    # Remove unwanted sections (e.g., tables, infoboxes, navigation boxes)
    for element in content.find_all(['table', 'div', 'span'], {'class': ['infobox', 'navbox', 'vertical-navbox', 'metadata']}):
        element.decompose()

    # Extract the text from paragraphs
    paragraphs = content.find_all('p')
    text = ' '.join(paragraph.get_text() for paragraph in paragraphs).replace("\n", "")

    return text

def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# List of Wikipedia URLs
urls = [

    # maantieto
    "https://fi.wikipedia.org/wiki/Pariisi",
    "https://fi.wikipedia.org/wiki/Lontoo",
    "https://fi.wikipedia.org/wiki/Shanghai",
    "https://fi.wikipedia.org/wiki/Delhi",
    "https://fi.wikipedia.org/wiki/Tokio",
    "https://fi.wikipedia.org/wiki/S%C3%A3o_Paulo",
    "https://fi.wikipedia.org/wiki/Kairo",
    "https://fi.wikipedia.org/wiki/Beijing",
    "https://fi.wikipedia.org/wiki/New_York_City",
    "https://fi.wikipedia.org/wiki/Moskova",
    "https://fi.wikipedia.org/wiki/Los_Angeles",
    "https://fi.wikipedia.org/wiki/Etel%C3%A4manner",
    "https://fi.wikipedia.org/wiki/Australia",
    "https://fi.wikipedia.org/wiki/Afrikka",
    "https://fi.wikipedia.org/wiki/Aasia",
    "https://fi.wikipedia.org/wiki/Amerikka",
    "https://fi.wikipedia.org/wiki/Eurooppa",
    "https://fi.wikipedia.org/wiki/Turku",
    "https://fi.wikipedia.org/wiki/Helsinki",
    "https://fi.wikipedia.org/wiki/Suomi",
    "https://fi.wikipedia.org/wiki/Espoo",
    "https://fi.wikipedia.org/wiki/Yhdysvallat",
    "https://fi.wikipedia.org/wiki/Kiina",
    "https://fi.wikipedia.org/wiki/Maantiede",


    # historia
    "https://fi.wikipedia.org/wiki/Toinen_maailmansota",
    "https://fi.wikipedia.org/wiki/Talvisota",
    "https://fi.wikipedia.org/wiki/Ensimm%C3%A4inen_maailmansota",
    "https://fi.wikipedia.org/wiki/Suomen_historia",
    "https://fi.wikipedia.org/wiki/Euroopan_historia",
    "https://fi.wikipedia.org/wiki/Renessanssi",
    "https://fi.wikipedia.org/wiki/Antiikin_Kreikka",
    "https://fi.wikipedia.org/wiki/Rooman_valtakunta",
    "https://fi.wikipedia.org/wiki/Uskonpuhdistus",
    "https://fi.wikipedia.org/wiki/Teollinen_vallankumous",
    "https://fi.wikipedia.org/wiki/Amerikan_historia",
    "https://fi.wikipedia.org/wiki/Pohjois-Amerikan_intiaanit",
    "https://fi.wikipedia.org/wiki/Etel%C3%A4-Amerikan_historia",
    "https://fi.wikipedia.org/wiki/Aasian_historia",
    "https://fi.wikipedia.org/wiki/Kiinan_historia",
    "https://fi.wikipedia.org/wiki/Afrikan_historia",
    "https://fi.wikipedia.org/wiki/Kolonialismi",
    "https://fi.wikipedia.org/wiki/Taiteen_historia",


    # henkilöt
    "https://fi.wikipedia.org/wiki/Urho_Kekkonen",
    "https://fi.wikipedia.org/wiki/Elvis_Presley",
    "https://fi.wikipedia.org/wiki/Vincent_van_Gogh",
    "https://fi.wikipedia.org/wiki/Napoleon_I",
    "https://fi.wikipedia.org/wiki/Friedrich_Nietzsche",
    "https://fi.wikipedia.org/wiki/Isaac_Newton",

    # luonnontieteet
    "https://fi.wikipedia.org/wiki/Fysiikka",
    "https://fi.wikipedia.org/wiki/Painovoima",
    "https://fi.wikipedia.org/wiki/S%C3%A4hk%C3%B6magnetismi",
    "https://fi.wikipedia.org/wiki/Termodynamiikka",
    "https://fi.wikipedia.org/wiki/Kvanttimekaniikka",
    "https://fi.wikipedia.org/wiki/Erityinen_suhteellisuusteoria",
    "https://fi.wikipedia.org/wiki/Newtonin_lait",
    "https://fi.wikipedia.org/wiki/S%C3%A4hk%C3%B6varaus",
    "https://fi.wikipedia.org/wiki/Liike-energia",
    "https://fi.wikipedia.org/wiki/Energia",


    "https://fi.wikipedia.org/wiki/Matematiikka",
    "https://fi.wikipedia.org/wiki/Logiikka",
    "https://fi.wikipedia.org/wiki/P%C3%A4%C3%A4ttely",
    "https://fi.wikipedia.org/wiki/Joukko",
    "https://fi.wikipedia.org/wiki/Todenn%C3%A4k%C3%B6isyys",
    "https://fi.wikipedia.org/wiki/Klassinen_todenn%C3%A4k%C3%B6isyyden_m%C3%A4%C3%A4ritelm%C3%A4",
    "https://fi.wikipedia.org/wiki/Todenn%C3%A4k%C3%B6isyysteoria",
    "https://fi.wikipedia.org/wiki/Derivaatta",
    "https://fi.wikipedia.org/wiki/Riemannin_integraali",

    "https://fi.wikipedia.org/wiki/Tietotekniikka",
    "https://fi.wikipedia.org/wiki/Tietokone",
    "https://fi.wikipedia.org/wiki/Ohjelmointi",
    "https://fi.wikipedia.org/wiki/Tietojenk%C3%A4sittelytiede",
    "https://fi.wikipedia.org/wiki/Algoritmi",
    "https://fi.wikipedia.org/wiki/Ohjelmointikieli",
    "https://fi.wikipedia.org/wiki/Teko%C3%A4ly",
    "https://fi.wikipedia.org/wiki/Neuroverkot",
    "https://fi.wikipedia.org/wiki/Syv%C3%A4oppiminen",

    "https://fi.wikipedia.org/wiki/Biologia",
    "https://fi.wikipedia.org/wiki/Maa",
    "https://fi.wikipedia.org/wiki/Avaruus",
    "https://fi.wikipedia.org/wiki/Anatomia",
    "https://fi.wikipedia.org/wiki/Ihmisen_anatomia",
    "https://fi.wikipedia.org/wiki/Ekologia",
    "https://fi.wikipedia.org/wiki/Solubiologia",
    "https://fi.wikipedia.org/wiki/Solu",
    "https://fi.wikipedia.org/wiki/Proteiinisynteesi",
    "https://fi.wikipedia.org/wiki/DNA",
    "https://fi.wikipedia.org/wiki/Mitokondrio",
    "https://fi.wikipedia.org/wiki/Soluhengitys",
    "https://fi.wikipedia.org/wiki/Ribosomi",
    "https://fi.wikipedia.org/wiki/Solunjakautuminen",
    "https://fi.wikipedia.org/wiki/Hermosto",

    "https://fi.wikipedia.org/wiki/Kemia",
    "https://fi.wikipedia.org/wiki/Molekyyli",
    "https://fi.wikipedia.org/wiki/Atomi",
    "https://fi.wikipedia.org/wiki/Alkuaine",
    "https://fi.wikipedia.org/wiki/Protoni",
    "https://fi.wikipedia.org/wiki/Elektroni",

    "https://fi.wikipedia.org/wiki/Taloustiede",
    "https://fi.wikipedia.org/wiki/Raha",
    "https://fi.wikipedia.org/wiki/Valtio",

    "https://fi.wikipedia.org/wiki/Filosofia",
    "https://fi.wikipedia.org/wiki/Estetiikka",

    "https://fi.wikipedia.org/wiki/taide",
    "https://fi.wikipedia.org/wiki/Musiikki",
    "https://fi.wikipedia.org/wiki/Rock",
    "https://fi.wikipedia.org/wiki/Rock_and_roll",
    "https://fi.wikipedia.org/wiki/Iskelm%C3%A4musiikki",
    "https://fi.wikipedia.org/wiki/Blues",



    
]


corpus = ""

# dataset = load_dataset("opus_books", "en-fi")["train"]
# for example in dataset:
#     print(example["translation"]["fi"])
#     corpus += example["translation"]["fi"]


# Iterate over each URL
random.shuffle(urls)
for url in urls:
    text = get_wikipedia_text(url)
    if text:
        # Generate a filename based on the page title
        cleaned_text = re.sub(r'\[\d+\]', '', text)

        corpus += cleaned_text
    else:
        print(f"Failed to retrieve the page at {url}")

# print(corpus)

input_file_path = os.path.join(os.path.dirname(__file__), 'raw_finnish.txt')
if not os.path.exists(input_file_path):
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(corpus)

n = len(corpus)
train_data = corpus[:int(n*0.9)]
val_data = corpus[int(n*0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'finnish_train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'finnish_val.bin'))
