import uvicorn
from f1pred.web import app
from f1pred.config import load_config
import f1pred.web
import threading

def start_server():
    f1pred.web._config = load_config('config.yaml')
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")

t = threading.Thread(target=start_server, daemon=True)
t.start()

import time
time.sleep(2)
import requests

try:
    print(requests.get("http://127.0.0.1:8000/api/event-status/2024/1").text)
    print(requests.get("http://127.0.0.1:8000/api/schedule/2024").text[:100])
except Exception as e:
    print(e)
