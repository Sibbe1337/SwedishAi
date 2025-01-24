import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import List, Dict
import logging

class DataCollector:
    def __init__(self, output_file: str = "data/training_data.txt"):
        self.output_file = output_file
        self.logger = logging.getLogger(__name__)
    
    def collect_from_wikipedia(self, categories: List[str]) -> List[str]:
        """Samla text från svenska Wikipedia-artiklar"""
        texts = []
        base_url = "https://sv.wikipedia.org/w/api.php"
        
        for category in categories:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": "500"
            }
            
            try:
                response = requests.get(base_url, params=params)
                data = response.json()
                
                for page in data["query"]["categorymembers"]:
                    page_text = self._get_page_content(page["pageid"])
                    if page_text:
                        texts.append(page_text)
                        
            except Exception as e:
                self.logger.error(f"Fel vid hämtning av {category}: {str(e)}")
        
        return texts
    
    def collect_from_news(self, sources: List[str]) -> List[str]:
        """Samla text från svenska nyhetskällor"""
        # Implementera nyhetsinsamling här
        pass

    def save_texts(self, texts: List[str]):
        """Spara texter till fil"""
        with open(self.output_file, "a", encoding="utf-8") as f:
            for text in texts:
                f.write(text + "\n\n") 