import logging

from fastapi.testclient import TestClient

from ai_service.main import app
from ai_service.services.ai_service import AIService

client = TestClient(app)


def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "AI Service is running"
    assert response.json()["version"] == "1.0.0"


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_password_classification() -> None:
    response = client.post("/classify", params={"text": "I forgot my password"})
    assert response.status_code == 200
    assert response.json()["category"] == "account"


def test_fee_classification() -> None:
    response = client.post("/classify", params={"text": "I have a fee problem"})
    assert response.status_code == 200
    assert response.json()["category"] == "finance"


def test_technical_classification() -> None:
    response = client.post("/classify", params={"text": "The server is down"})
    assert response.status_code == 200
    assert response.json()["category"] == "technical"


def test_unknown_classification() -> None:
    response = client.post("/classify", params={"text": "I need information"})
    assert response.status_code == 200
    assert response.json()["category"] == "other"


def test_empty_text_returns_400() -> None:
    response = client.post("/classify", params={"text": ""})
    assert response.status_code == 400


def test_service_directly() -> None:
    service = AIService(logging.getLogger("test"))
    result = service.classify("I forgot my password")
    assert result == {
        "text": "I forgot my password",
        "category": "account",
    }


def test_facilities_classification_lab() -> None:
    response = client.post("/classify", params={"text": "The computer lab is closed"})
    assert response.status_code == 200
    assert response.json()["category"] == "facilities"


def test_facilities_classification_classroom() -> None:
    response = client.post("/classify", params={"text": "The classroom needs repair"})
    assert response.status_code == 200
    assert response.json()["category"] == "facilities"


def test_version_endpoint() -> None:
    response = client.get("/version")
    assert response.status_code == 200
    assert response.json() == {"version": "1.0.0"}
