import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from serving.fast_api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health_check")
    assert response.status_code == 200
    assert response.json() == "ok"


def test_predict():
    response = client.post("/predict", json={"text": ["this is test"]})
    assert response.status_code == 200
    assert len(response.json()["probs"][0]) == 2
