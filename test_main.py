from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment Analysis API is running!"}

def test_predict_positive():
    payload = {"text": "Отличный день, всё работает замечательно!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert data["label"] == "POSITIVE"
    assert data["score"] > 0.5

def test_predict_negative():
    payload = {"text": "Ужасный сервис, все сломалось и не работает."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "NEGATIVE"
    assert data["score"] > 0.5

def test_predict_empty_invalid_payload():
    response = client.post("/predict", json={"invalid_field": 123})
    assert response.status_code == 422