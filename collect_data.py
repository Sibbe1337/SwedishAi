import requests
from bs4 import BeautifulSoup
import logging
import os
import json
import pandas as pd
import feedparser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_riksdagen_data():
    """Hämta data från Riksdagens öppna data"""
    try:
        # API endpoint för Riksdagens dokument
        url = "https://data.riksdagen.se/api/dokumentlista/bet?utformat=json&limit=100"
        response = requests.get(url)
        data = response.json()
        
        texts = []
        for dok in data.get('dokumentlista', {}).get('dokument', []):
            if 'titel' in dok and 'summary' in dok:
                texts.append(f"{dok['titel']}\n{dok['summary']}")
        
        return texts
    except Exception as e:
        logger.error(f"Fel vid hämtning av Riksdagsdata: {str(e)}")
        return []

def fetch_kb_data():
    """Hämta data från Kungliga Biblioteket"""
    try:
        # KB:s öppna data API
        url = "https://libris.kb.se/api/search?q=language:swe&format=json&limit=100"
        response = requests.get(url)
        data = response.json()
        
        texts = []
        for item in data.get('items', []):
            if 'title' in item and 'description' in item:
                texts.append(f"{item['title']}\n{item['description']}")
        
        return texts
    except Exception as e:
        logger.error(f"Fel vid hämtning av KB-data: {str(e)}")
        return []

def fetch_swedish_news():
    """Hämta nyheter från svenska nyhetskällor"""
    try:
        urls = [
            "https://www.svt.se/nyheter/rss.xml",
            "https://www.dn.se/rss/",
            "https://www.svd.se/feed/articles.rss",
            "https://www.aftonbladet.se/rss.xml",
            "https://feeds.expressen.se/nyheter",
            "https://www.sydsvenskan.se/rss.xml"
        ]
        
        texts = []
        for url in urls:
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:20]:  # Ta de 20 senaste nyheterna
                title = entry.get('title', '')
                desc = entry.get('description', '')
                content = entry.get('content', [{'value': ''}])[0]['value']
                
                text = f"{title}\n{desc}\n{content}".strip()
                if text:
                    texts.append(text)
        
        return texts
    except Exception as e:
        logger.error(f"Fel vid hämtning av nyheter: {str(e)}")
        return []

def fetch_swedish_literature():
    """Hämta svensk litteratur från Project Runeberg"""
    try:
        url = "https://runeberg.org/authors/swe.html"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        texts = []
        for link in soup.find_all('a', href=True):
            if '/txt/' in link['href']:
                text_url = f"https://runeberg.org{link['href']}"
                text_response = requests.get(text_url)
                texts.append(text_response.text)
                
                if len(texts) >= 50:  # Begränsa antal texter
                    break
        
        return texts
    except Exception as e:
        logger.error(f"Fel vid hämtning av litteratur: {str(e)}")
        return []

def scrape_swedish_text(urls):
    """Skrapa svensk text från givna URLs"""
    texts = []
    
    for url in urls:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Ta bort script och style-element
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Hämta text
            text = soup.get_text()
            
            # Rensa och formatera
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            texts.append(text)
            
        except Exception as e:
            logger.error(f"Fel vid skrapning av {url}: {str(e)}")
    
    return texts

def save_training_data(texts, output_file="data/training_data.txt"):
    """Spara texter till fil"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + "\n\n")
        logger.info(f"Data sparad till {output_file}")
    except Exception as e:
        logger.error(f"Fel vid sparande av data: {str(e)}")
        raise

if __name__ == "__main__":
    # Konfigurera requests för att hantera timeouts
    requests.adapters.DEFAULT_RETRIES = 5
    
    # Skapa data-mappen om den inte finns
    os.makedirs("data", exist_ok=True)
    
    all_texts = []
    
    logger.info("Startar datainsamling...")
    
    # Wikipedia-artiklar
    logger.info("Hämtar Wikipedia-artiklar...")
    wiki_texts = scrape_swedish_text([
        "https://sv.wikipedia.org/wiki/Sverige",
        "https://sv.wikipedia.org/wiki/Svenska",
        "https://sv.wikipedia.org/wiki/Stockholm",
        "https://sv.wikipedia.org/wiki/Svensk_kultur",
        "https://sv.wikipedia.org/wiki/Svenska_traditioner",
        "https://sv.wikipedia.org/wiki/Svensk_litteratur",
        "https://sv.wikipedia.org/wiki/Sveriges_historia",
        "https://sv.wikipedia.org/wiki/Svenska_språket",
        "https://sv.wikipedia.org/wiki/Sveriges_geografi",
        "https://sv.wikipedia.org/wiki/Svenska_högtider",
        "https://sv.wikipedia.org/wiki/Svensk_mat",
        "https://sv.wikipedia.org/wiki/Svenska_uppfinningar"
    ])
    all_texts.extend(wiki_texts)
    logger.info(f"Hämtade {len(wiki_texts)} Wikipedia-texter")
    
    # Riksdagsdata
    logger.info("Hämtar Riksdagsdata...")
    riksdag_texts = fetch_riksdagen_data()
    all_texts.extend(riksdag_texts)
    logger.info(f"Hämtade {len(riksdag_texts)} Riksdagstexter")
    
    # KB-data
    logger.info("Hämtar KB-data...")
    kb_texts = fetch_kb_data()
    all_texts.extend(kb_texts)
    logger.info(f"Hämtade {len(kb_texts)} KB-texter")
    
    # Nyheter
    logger.info("Hämtar nyheter...")
    news_texts = fetch_swedish_news()
    all_texts.extend(news_texts)
    logger.info(f"Hämtade {len(news_texts)} nyhetstexter")
    
    # Litteratur
    logger.info("Hämtar litteratur...")
    lit_texts = fetch_swedish_literature()
    all_texts.extend(lit_texts)
    logger.info(f"Hämtade {len(lit_texts)} litteraturtexter")
    
    # Spara all data
    logger.info(f"Totalt antal texter: {len(all_texts)}")
    save_training_data(all_texts) 