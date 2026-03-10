from fastapi.testclient import TestClient
from f1pred.config import load_config
import f1pred.web

f1pred.web._config = load_config('config.yaml')
client = TestClient(f1pred.web.app)

response = client.get("/api/schedule/abc")
print(response.status_code)
print(response.text)
print(response.headers.get("content-type"))
