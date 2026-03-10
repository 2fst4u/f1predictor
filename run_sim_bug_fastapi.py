from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/test")
async def test():
    raise HTTPException(status_code=500, detail="Internal server error")

client = TestClient(app)
print(client.get("/test").text)
print(client.get("/test").headers)
