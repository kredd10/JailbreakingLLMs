import requests
import os
import json

# Set up your OpenAI API key
api_key = os.getenv("OPENAI_API_KEY") or "YOUR_API_KEY"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Define the prompt and model
data = {
    "model": "gpt-4o-mini",  # Use "gpt-4" if you have access
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the importance of data privacy in a digital world."}
    ],
    "max_tokens": 100
}

# Send the request
try:
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data))
    response.raise_for_status()
    result = response.json()
    print("Model Response:", result["choices"][0]["message"]["content"].strip())
except requests.exceptions.HTTPError as err:
    print("Error:", err)
