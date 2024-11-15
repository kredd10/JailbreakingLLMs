import requests
import json
# not working yet
# Define the API key and URL
api_key = "GEMINI_API_KEY"
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

# Define the headers and the payload
headers = {
    "Content-Type": "application/json"
}

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Explain how AI works"
                }
            ]
        }
    ]
}

# Make the POST request
try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    # Check if the request was successful
    if response.status_code == 200:
        print("Response:", response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print("An error occurred:", e)
