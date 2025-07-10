import requests

def fetch_and_save_text(url: str, save_path: str):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        with open(save_path, 'w', encoding='utf-8') as file:
            file.write(response.text)

        print(f"Text successfully saved to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
    except IOError as e:
        print(f"Error writing to file: {e}")

URL = "https://raw.githubusercontent.com/MIT-JClinic/JClinic-2025-Summer-Program-Data/main/hackathon_data.csv"
SAVE_PATH = "hackathon_data.csv"

if __name__ == "__main__":
    fetch_and_save_text(URL, SAVE_PATH)