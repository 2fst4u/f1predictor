from fastapi.testclient import TestClient
from f1pred.config import load_config
import f1pred.web

f1pred.web._config = load_config('config.yaml')
client = TestClient(f1pred.web.app)

# What if season is None or empty?
response = client.get("/api/event-status/None/1")
print("status:", response.status_code)
print("content:", response.text)
print("type:", response.headers.get("content-type"))
