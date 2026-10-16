
import requests

from scenario2_chatbot.prompts import build_prompt
from scenario2_chatbot.retrieval import search


OLLAMA_URL = "http://localhost:11434"
CHAT_MODEL = "llama3:latest"

# Initial demo setting; tune with relevant and irrelevant queries.
MIN_SIMILARITY = 0.35


def answer_question(question: str) -> dict:
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    retrieved_chunks = search(question, top_k=3)

    if not retrieved_chunks:
        return {
            "answer": (
                "I could not find sufficient evidence in the "
                "available engineering documentation."
            ),
            "sources": [],
            "abstained": True,
        }

    best_score = max(chunk["score"] for chunk in retrieved_chunks)

    if best_score < MIN_SIMILARITY:
        return {
            "answer": (
                "I could not find sufficient evidence in the "
                "available engineering documentation."
            ),
            "sources": [],
            "abstained": True,
        }

    prompt = build_prompt(question, retrieved_chunks)

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": CHAT_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=180,
    )
    response.raise_for_status()

    answer = response.json()["response"].strip()

    sources = []
    for chunk in retrieved_chunks:
        source = {
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "page": chunk.get("page"),
            "similarity": round(chunk["score"], 3),
        }

        if source not in sources:
            sources.append(source)

    return {
        "answer": answer,
        "sources": sources,
        "abstained": False,
    }


if __name__ == "__main__":
    question = "What should I check if a database connection fails?"

    result = answer_question(question)

    print("\nQuestion:", question)
    print("\nAnswer:", result["answer"])
    print("\nSources:")

    for source in result["sources"]:
        print(source)

    print("\nAbstained:", result["abstained"])
