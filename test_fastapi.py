from fastapi.testclient import TestClient
from f1pred.config import load_config
import f1pred.web

f1pred.web._config = load_config('config.yaml')
client = TestClient(f1pred.web.app)

def run():
    response = client.get("/api/event-status/2024/1")
    print(response.status_code)
    print(response.text)

if __name__ == "__main__":
    run()
