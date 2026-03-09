from fastapi.testclient import TestClient
from f1pred.config import load_config
import f1pred.web

f1pred.web._config = load_config('config.yaml')
client = TestClient(f1pred.web.app)

# The user is probably getting an error parsing JSON somewhere.
# Wait, look at this error: "JSON.parse: unexpected character at line 1 column 1 of the JSON data"
# This typically happens when fetch returns HTML (like a 500 error page or a 404 page) instead of JSON.
# But FastAPI returns {"detail": "Internal server error"} which IS valid JSON!

response = client.get("/api/event-status/2024/1")
print(response.headers.get("content-type"))
