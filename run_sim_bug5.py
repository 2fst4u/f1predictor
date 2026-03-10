from fastapi.testclient import TestClient
from f1pred.config import load_config
import f1pred.web

f1pred.web._config = load_config('config.yaml')
client = TestClient(f1pred.web.app)

# What if it's a 404? (e.g. invalid endpoint format)
response = client.get("/api/event-status/2024/")
print("status:", response.status_code)
print("content:", response.text)
print("type:", response.headers.get("content-type"))
