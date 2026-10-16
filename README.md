
# AI Specialist Candidate Evaluation

This repository contains two AI/ML solutions developed as part of the AI Specialist Candidate Evaluation.

## Scenario 1: Customer Support Ticket Classifier

A machine learning application that classifies customer support messages into three categories: billing, technical issues, and general inquiries.

**Key features:**
- Text classification using TF-IDF and Logistic Regression.
- Confidence-based predictions and human-review recommendations.
- FastAPI REST API.
- Model evaluation and automated tests.

**Commands to run Scenario 1**

Train the model:

```powershell
uv run python scenario1_classifier/train.py
```

Run model evaluation:

```powershell
uv run python scenario1_classifier/evaluation.py
```

Start the API:

```powershell
uv run uvicorn scenario1_classifier.api:app --reload
```

Open Swagger UI: http://127.0.0.1:8000/docs

Documentation: [Scenario 1 README](scenario1_classifier/README.md)

## Scenario 2: Engineering Knowledge Chatbot (RAG)

A local Retrieval-Augmented Generation (RAG) application that answers engineering questions using technical documentation.

**Key features:**
- TXT and PDF document ingestion.
- Embedding generation using Ollama.
- Semantic search using FAISS.
- Answer generation using Llama 3.
- Source attribution and insufficient-evidence handling.
- FastAPI REST API.

**Prerequisites:** Ollama must be running, with the required models available:

```powershell
ollama pull llama3:latest
ollama pull nomic-embed-text:latest
```

Build the vector index:

```powershell
uv run python -m scenario2_chatbot.retrieval
```

Run the chatbot in the terminal:

```powershell
uv run python -m scenario2_chatbot.chatbot
```

Start the API:

```powershell
uv run uvicorn scenario2_chatbot.api:app --reload
```

Open Swagger UI: http://127.0.0.1:8000/docs

Documentation: [Scenario 2 README](scenario2_chatbot/README.md)

## Technology Stack

- **Language:** Python
- **Machine Learning:** scikit-learn
- **API:** FastAPI
- **LLM Runtime:** Ollama
- **Embedding Model:** Nomic Embed Text
- **Vector Search:** FAISS
- **Testing:** pytest

## Automated Tests

Run all tests from the repository root:

```powershell
uv run python -m pytest tests -v
```

Both scenarios include dedicated implementation documentation and architecture descriptions.
