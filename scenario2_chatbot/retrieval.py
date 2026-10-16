
from pathlib import Path

import faiss
import numpy as np
import requests

from scenario2_chatbot.ingestion import (
    chunk_documents,
    extract_documents,
)


OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text:latest"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
INDEX_DIR = ROOT_DIR / "vector_index"


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate normalized embeddings using Ollama."""
    vectors = []

    for text in texts:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": text,
            },
            timeout=120,
        )
        response.raise_for_status()

        embeddings = response.json()["embeddings"]
        vectors.append(embeddings[0])

    matrix = np.asarray(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)

    return matrix


def build_index():
    documents = extract_documents(DATA_DIR)
    chunks = chunk_documents(documents)

    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "engineering.faiss"))

    # Store metadata in the same order as vectors in the index
    import json

    with (INDEX_DIR / "chunks.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"Indexed chunks: {index.ntotal}")
    print(f"Vector dimensions: {vectors.shape[1]}")
    print(f"Index saved to: {INDEX_DIR}")


def search(query: str, top_k: int = 3) -> list[dict]:
    """Retrieve the most similar document chunks."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    import json

    index_path = INDEX_DIR / "engineering.faiss"
    metadata_path = INDEX_DIR / "chunks.json"

    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run build_index() first."
        )

    index = faiss.read_index(str(index_path))

    with metadata_path.open(encoding="utf-8") as file:
        chunks = json.load(file)

    query_vector = embed_texts([query])

    k = min(max(1, top_k), index.ntotal)
    scores, indices = index.search(query_vector, k)

    results = []

    for score, position in zip(scores[0], indices[0]):
        if position < 0:
            continue

        result = chunks[int(position)].copy()
        result["score"] = float(score)
        results.append(result)

    return results


if __name__ == "__main__":
    build_index()

    query = "What should I check if a database connection fails?"
    results = search(query, top_k=2)

    print(f"\nQuestion: {query}")

    for result in results:
        print(f"\nSource: {result['source']}")
        print(f"Similarity: {result['score']:.3f}")
        print(result["text"])
