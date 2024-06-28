import requests
import os
import tiktoken
import numpy as np
from bs4 import BeautifulSoup
import re
import random
from collections import deque

def get_wikipedia_links(url):
    """Retrieve all Wikipedia links from the given URL."""
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        links = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/wiki/') and not ':' in href:
                full_url = 'https://fi.wikipedia.org' + href
                links.add(full_url)
        
        return links
    except Exception as e:
        print(f"Failed to retrieve links from {url} due to {e}")
        return []

def get_wikipedia_text(url):
    """Retrieve all text content from the given Wikipedia URL."""
    try:
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
    except Exception as e:
        print(f"Failed to retrieve text from {url} due to {e}")
        return None

def save_text_to_file(text, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

def crawl_wikipedia(start_urls, depth, output_file):
    """Crawl Wikipedia starting from start_urls up to the specified depth and write text to output_file."""
    visited = set()
    queue = deque([(url, 0) for url in start_urls])
    all_texts = ""
    try:    
        while queue:

            current_url, current_depth = queue.popleft()
            
            if current_depth > depth:
                continue
            
            if current_url in visited:
                continue
            
            visited.add(current_url)
            print(f"{current_depth} - {current_url}")

            # Get text content and add to the list
            text_content = get_wikipedia_text(current_url)
            if text_content:
                all_texts += "\n" + text_content
        
            if current_depth < depth:
                links = get_wikipedia_links(current_url)
                if len(links) > 30:
                    links = random.sample(list(links), 30)
                for link in links:
                    if link not in visited:
                        queue.append((link, current_depth + 1))
                        
            if len(visited) % 50 == 0:
                enc = tiktoken.get_encoding("gpt2")
                ids = enc.encode_ordinary(all_texts)
                print(f"Currently {len(ids):,} tokens")
    except KeyboardInterrupt:
        pass
    # Write all collected texts to the output file
    save_text_to_file(all_texts, output_file)

if __name__ == "__main__":
    start_urls = [
        "https://fi.wikipedia.org/wiki/Pariisi",
        "https://fi.wikipedia.org/wiki/Shanghai",
        "https://fi.wikipedia.org/wiki/S%C3%A3o_Paulo",
        "https://fi.wikipedia.org/wiki/Kairo",
        "https://fi.wikipedia.org/wiki/Moskova",
        "https://fi.wikipedia.org/wiki/Etel%C3%A4manner",
        "https://fi.wikipedia.org/wiki/Australia",
        "https://fi.wikipedia.org/wiki/Afrikka",
        "https://fi.wikipedia.org/wiki/Aasia",
        "https://fi.wikipedia.org/wiki/Amerikka",
        "https://fi.wikipedia.org/wiki/Eurooppa",
        "https://fi.wikipedia.org/wiki/Helsinki",
        "https://fi.wikipedia.org/wiki/Suomi",
        "https://fi.wikipedia.org/wiki/Maantiede",
        "https://fi.wikipedia.org/wiki/Toinen_maailmansota",
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
        "https://fi.wikipedia.org/wiki/Kolonialismi",
        "https://fi.wikipedia.org/wiki/Taiteen_historia",
        "https://fi.wikipedia.org/wiki/Urho_Kekkonen",
        "https://fi.wikipedia.org/wiki/Elvis_Presley",
        "https://fi.wikipedia.org/wiki/Vincent_van_Gogh",
        "https://fi.wikipedia.org/wiki/Napoleon_I",
        "https://fi.wikipedia.org/wiki/Friedrich_Nietzsche",
        "https://fi.wikipedia.org/wiki/Isaac_Newton",
        "https://fi.wikipedia.org/wiki/Fysiikka",
        "https://fi.wikipedia.org/wiki/Painovoima",
        "https://fi.wikipedia.org/wiki/S%C3%A4hk%C3%B6magnetismi",
        "https://fi.wikipedia.org/wiki/Termodynamiikka",
        "https://fi.wikipedia.org/wiki/Kvanttimekaniikka",
        "https://fi.wikipedia.org/wiki/Erityinen_suhteellisuusteoria",
        "https://fi.wikipedia.org/wiki/Newtonin_lait",
        "https://fi.wikipedia.org/wiki/S%C3%A4hk%C3%B6varaus",
        "https://fi.wikipedia.org/wiki/Energia",
        "https://fi.wikipedia.org/wiki/Matematiikka",
        "https://fi.wikipedia.org/wiki/Logiikka",
        "https://fi.wikipedia.org/wiki/P%C3%A4%C3%A4ttely",
        "https://fi.wikipedia.org/wiki/Joukko",
        "https://fi.wikipedia.org/wiki/Todenn%C3%A4k%C3%B6isyys",
        "https://fi.wikipedia.org/wiki/Derivaatta",
        "https://fi.wikipedia.org/wiki/Riemannin_integraali",
        "https://fi.wikipedia.org/wiki/Tietotekniikka",
        "https://fi.wikipedia.org/wiki/Tietokone",
        "https://fi.wikipedia.org/wiki/Ohjelmointi",
        "https://fi.wikipedia.org/wiki/Tietojenk%C3%A4sittelytiede",
        "https://fi.wikipedia.org/wiki/Algoritmi",
        "https://fi.wikipedia.org/wiki/Ohjelmointikieli",
        "https://fi.wikipedia.org/wiki/Teko%C3%A4ly",
        "https://fi.wikipedia.org/wiki/Biologia",
        "https://fi.wikipedia.org/wiki/Maa",
        "https://fi.wikipedia.org/wiki/Avaruus",
        "https://fi.wikipedia.org/wiki/Anatomia",
        "https://fi.wikipedia.org/wiki/Ekologia",
        "https://fi.wikipedia.org/wiki/Solubiologia",
        "https://fi.wikipedia.org/wiki/Solu",
        "https://fi.wikipedia.org/wiki/Hermosto",
        "https://fi.wikipedia.org/wiki/Kemia",
        "https://fi.wikipedia.org/wiki/Molekyyli",
        "https://fi.wikipedia.org/wiki/Atomi",
        "https://fi.wikipedia.org/wiki/Taloustiede",
        "https://fi.wikipedia.org/wiki/Raha",
        "https://fi.wikipedia.org/wiki/Valtio",
        "https://fi.wikipedia.org/wiki/Filosofia",
        "https://fi.wikipedia.org/wiki/Estetiikka",
        "https://fi.wikipedia.org/wiki/taide",
        "https://fi.wikipedia.org/wiki/Musiikki",
        "https://fi.wikipedia.org/wiki/Rock",
        "https://fi.wikipedia.org/wiki/Blues",
    ]

    random.shuffle(start_urls)
    depth = 3  # Adjust depth as needed
    output_file = 'raw_finnish.txt'
    
    crawl_wikipedia(start_urls, depth, output_file)

    # Read the corpus
    with open(output_file, 'r', encoding='utf-8') as file:
        corpus = file.read()

    # Clean the corpus
    cleaned_corpus = re.sub(r'\[\d+\]', '', corpus)

    # Save cleaned corpus
    input_file_path = os.path.join(os.path.dirname(__file__), 'cleaned_finnish.txt')
    if not os.path.exists(input_file_path):
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_corpus)

    # Split the data into training and validation sets
    n = len(cleaned_corpus)
    train_data = cleaned_corpus[:int(n*0.9)]
    val_data = cleaned_corpus[int(n*0.9):]

    # Encode the text
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Export to binary files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'finnish_train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'finnish_val.bin'))
