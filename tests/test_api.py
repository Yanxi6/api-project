from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict():
    payload = {
        "Pclass": 3,
        "Name": "Doe, Mr. John",
        "Sex": "male",
        "Age": 22,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 7.25,
        "Cabin": None,
        "Embarked": "S"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "survived" in body and body["survived"] in [0, 1]
    assert "confidence" in body and 0.0 <= body["confidence"] <= 1.0
