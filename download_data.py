
import requests
import os

def download_training_data():
    url = "https://storage.googleapis.com/swedish-ai-data/swedish_text.txt"
    os.makedirs("model/data", exist_ok=True)
    with open("model/data/swedish_text.txt", "wb") as f:
        response = requests.get(url)
        f.write(response.content)

if __name__ == "__main__":
    download_training_data()

