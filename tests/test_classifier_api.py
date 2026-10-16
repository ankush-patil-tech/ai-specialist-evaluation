
import pytest
from fastapi.testclient import TestClient
from scenario1_classifier.predict import predict_ticket
from scenario1_classifier.api import app


@pytest.fixture
def client():
    # Runs FastAPI startup and loads the trained model
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_ticket_message(client):
    response = client.post(
        "/tickets/classify",
        json={
            "message": "I was charged twice for my subscription"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["category"] in {
        "billing",
        "technical_issue",
        "general_inquiry",
    }
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["needs_human_review"], bool)


def test_empty_message(client):
    response = client.post(
        "/tickets/classify",
        json={"message": ""},
    )

    # Rejected by Pydantic validation
    assert response.status_code == 422


def test_whitespace_only_message(client):
    response = client.post(
        "/tickets/classify",
        json={"message": "   "},
    )

    assert response.status_code == 422
    assert "whitespace only" in response.text.lower()


def test_oversized_message(client):
    response = client.post(
        "/tickets/classify",
        json={"message": "a" * 5001},
    )

    assert response.status_code == 422


def test_missing_message_field(client):
    response = client.post(
        "/tickets/classify",
        json={},
    )

    assert response.status_code == 422


def test_invalid_json_body(client):
    response = client.post(
        "/tickets/classify",
        json=["this", "is", "not", "an", "object"],
    )

    assert response.status_code == 422


class MockClassifier:
    def __init__(self, probabilities, category="billing"):
        self.probabilities = probabilities
        self.category = category

    def predict(self, messages):
        return [self.category]

    def predict_proba(self, messages):
        return [self.probabilities]


def test_low_confidence_requires_human_review():
    model = MockClassifier([0.45, 0.35, 0.20])

    result = predict_ticket("Please check my ticket", model=model)

    assert result["category"] == "billing"
    assert result["confidence"] == 0.45
    assert result["needs_human_review"] is True


def test_high_confidence_does_not_trigger_review():
    model = MockClassifier([0.85, 0.10, 0.05])

    result = predict_ticket("Please check my ticket", model=model)

    assert result["confidence"] == 0.85
    assert result["needs_human_review"] is False
