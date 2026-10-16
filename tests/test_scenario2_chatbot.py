
from unittest import result

from fastapi.testclient import TestClient

from scenario2_chatbot import api, chatbot

client = TestClient(api.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_endpoint_returns_answer(monkeypatch):
    expected = {
        "answer": "Check the endpoint and network connectivity.",
        "sources": [
            {
                "source": "engineering_guide.txt",
                "chunk_id": "chunk_1",
                "page": None,
                "similarity": 0.662,
            }
        ],
        "abstained": False,
    }

    monkeypatch.setattr(api, "answer_question", lambda question: expected)

    response = client.post(
        "/chat",
        json={"question": "What should I check for a database failure?"},
    )

    assert response.status_code == 200
    assert response.json() == expected


def test_chat_endpoint_strips_question(monkeypatch):
    captured = {}

    def fake_answer(question):
        captured["question"] = question
        return {"answer": "Test answer", "sources": [], "abstained": False}

    monkeypatch.setattr(api, "answer_question", fake_answer)

    response = client.post(
        "/chat",
        json={"question": "  How do I check service health?  "},
    )

    assert response.status_code == 200
    assert captured["question"] == "How do I check service health?"


def test_chat_rejects_empty_question():
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422


def test_chat_rejects_whitespace_question():
    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 422


def test_chat_rejects_question_over_max_length():
    response = client.post("/chat", json={"question": "a" * 1001})

    assert response.status_code == 422


def test_chat_rejects_missing_question():
    response = client.post("/chat", json={})

    assert response.status_code == 422


def test_chat_returns_503_when_service_fails(monkeypatch):
    def fake_answer(question):
        raise RuntimeError("Simulated model failure")

    monkeypatch.setattr(api, "answer_question", fake_answer)

    response = client.post(
        "/chat",
        json={"question": "How do I restart a service?"},
    )

    assert response.status_code == 503
    assert "could not process" in response.json()["detail"].lower()


def test_chatbot_abstains_when_retrieval_is_empty(monkeypatch):
    monkeypatch.setattr(chatbot, "search", lambda query, top_k=3: [])

    result = chatbot.answer_question("What is the leave policy?")

    assert result["abstained"] is True
    assert result["sources"] == []
    assert "sufficient evidence" in result["answer"].lower()

def test_chatbot_abstains_when_similarity_is_too_low(monkeypatch):
    monkeypatch.setattr(
        chatbot,
        "search",
        lambda query, top_k=3: [
            {
                "text": "Unrelated engineering content",
                "source": "engineering_guide.txt",
                "chunk_id": "chunk_1",
                "page": None,
                "score": 0.10,
            }
        ],
    )

    result = chatbot.answer_question("What is the leave policy?")

    assert result["abstained"] is True
    assert result["sources"] == []


def test_chatbot_rejects_empty_question():
    import pytest

    with pytest.raises(ValueError, match="cannot be empty"):
        chatbot.answer_question("   ")
