import requests
import os

api_key = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
headers = {
    "Authorization": f"Bearer {api_key}"
}

try:
    response = requests.get("https://api.openai.com/v1/models", headers=headers)
    response.raise_for_status()
    print("API Key is working. Available models:", response.json())
except requests.exceptions.HTTPError as err:
    print("Error:", err)
