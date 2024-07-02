import requests
import os
import tiktoken
import numpy as np
from bs4 import BeautifulSoup
import re
import random
from collections import deque

visited_file_path = 'data/visited.txt'

def load_visited_urls(file_path):
    """Load visited URLs from a file."""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            visited = set(file.read().splitlines())
    else:
        visited = set()
    return visited

def save_visited_urls(visited, file_path):
    """Save visited URLs to a file."""
    with open(file_path, 'a', encoding='utf-8') as file:
        for url in visited:
            file.write(f"{url}\n")

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
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(text)

def crawl_wikipedia(start_urls, depth, output_file):
    """Crawl Wikipedia starting from start_urls up to the specified depth and write text to output_file."""
    visited = load_visited_urls(visited_file_path)
    start_urls = [url for url in start_urls if url not in visited]
    queue = deque([(url, 0) for url in start_urls if url not in visited])
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
    save_visited_urls(visited, visited_file_path)  # Save visited URLs at the end


if __name__ == "__main__":
    start_urls = [
        "https://fi.wikipedia.org/wiki/Komedia",
        "https://fi.wikipedia.org/wiki/Makaroni",
        "https://fi.wikipedia.org/wiki/Ruoka",
        "https://fi.wikipedia.org/wiki/Hunaja",
        "https://fi.wikipedia.org/wiki/Kasvit",
        "https://fi.wikipedia.org/wiki/J%C3%A4nisel%C3%A4imet",
        "https://fi.wikipedia.org/wiki/Radio",
        "https://fi.wikipedia.org/wiki/Musta",
        "https://fi.wikipedia.org/wiki/Koira",
        "https://fi.wikipedia.org/wiki/Nappi",
        "https://fi.wikipedia.org/wiki/Pluto",
        "https://fi.wikipedia.org/wiki/Harry_Potter",
        "https://fi.wikipedia.org/wiki/Sofi_Oksanen",
        "https://fi.wikipedia.org/wiki/Aleksis_Kivi",
    ]

    random.shuffle(start_urls)
    depth = 3  # Adjust depth as needed
    output_file = 'data/raw_finnish.txt'
    
    crawl_wikipedia(start_urls, depth, output_file)

    # Read the corpus
    with open(output_file, 'r', encoding='utf-8') as file:
        corpus = file.read()

    # Clean the corpus
    cleaned_corpus = re.sub(r'\[\d+\]', '', corpus)

    # Save cleaned corpus
    input_file_path = os.path.join(os.path.dirname(__file__), 'data/cleaned_finnish.txt')
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
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'data/finnish_train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'data/finnish_val.bin'))
