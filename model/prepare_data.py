from datasets import load_dataset
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_swedish_data():
    """Ladda ner och förbereda svensk textdata"""
    try:
        # Skapa datamapp om den inte finns
        os.makedirs("model/data", exist_ok=True)
        
        logger.info("Laddar ner svensk data från OSCAR...")
        dataset = load_dataset(
            "oscar",
            "unshuffled_deduplicated_sv",
            split="train",
            streaming=True  # För att hantera stor datamängd
        )
        
        # Spara data i chunks för att hantera minnet effektivt
        with open("model/data/swedish_text.txt", "w", encoding="utf-8") as f:
            count = 0
            for item in dataset:
                if count >= 100000:  # Begränsa till 100k exempel för MVP
                    break
                    
                text = item['text'].replace('\n', ' ').strip()
                if len(text.split()) > 50:  # Filtrera korta texter
                    f.write(text + "\n")
                    count += 1
                    
                if count % 1000 == 0:
                    logger.info(f"Bearbetat {count} texter...")
        
        logger.info(f"Sparade {count} texter till swedish_text.txt")
        return "model/data/swedish_text.txt"
        
    except Exception as e:
        logger.error(f"Fel vid datanedladdning: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        data_file = download_swedish_data()
        logger.info(f"Data sparad i: {data_file}")
    except Exception as e:
        logger.error(f"Skript misslyckades: {str(e)}") 