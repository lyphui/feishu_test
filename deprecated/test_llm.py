import json
import requests
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

COZE_API_KEY = os.getenv("COZE_API_KEY", "")

url = "https://99p6x2gyv9.coze.site/run"
headers = {
  "Authorization": f"Bearer {COZE_API_KEY}",
  "Content-Type": "application/json",
}
payload = json.loads(r'''{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的AI助手。"},
    {"role": "user", "content": "你好！"}
  ],
  "model": "doubao-pro",
  "temperature": 0.7,
  "max_tokens": 32768,
  "top_p": 0.9
}''')

response = requests.post(url, headers=headers, json=payload)
print(response.status_code)
print(response.text)