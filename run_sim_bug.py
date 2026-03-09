# If an exception is raised inside FastAPI, it might return a 500 with plain text or HTML
# but FastAPI usually returns {"detail": "Internal Server Error"} as JSON.
# Wait! Look at f1pred/web.py get_event_status.
import requests
from fastapi.testclient import TestClient
from f1pred.config import load_config
import f1pred.web

f1pred.web._config = load_config('config.yaml')
client = TestClient(f1pred.web.app)

response = client.get("/api/event-status/2024/99")
print(response.status_code)
print(response.text)
print(response.headers.get("content-type"))
