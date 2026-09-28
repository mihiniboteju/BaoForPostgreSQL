import os
import requests

api_key = os.environ["GEMINI_API_KEY"]

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent"

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": api_key,
}

data = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Reply with exactly: GEMINI_TEST_OK"
                }
            ]
        }
    ]
}

response = requests.post(
    url,
    headers=headers,
    json=data,
    timeout=60
)

print("HTTP status:", response.status_code)
print("Response:", response.text)
