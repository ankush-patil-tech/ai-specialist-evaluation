
# Scenario 2: Engineering Knowledge Chatbot (RAG)

## 1. Overview

The Engineering Knowledge Chatbot is a Retrieval-Augmented Generation (RAG) application designed to answer engineering-related questions using a collection of technical documents.

Instead of relying only on the language model's internal knowledge, the application retrieves relevant information from the available documentation and provides it as context to the model. The model then generates an answer based on that retrieved information.

The application is designed to demonstrate a practical, locally running RAG pipeline with document ingestion, text chunking, embedding generation, vector search, answer generation, source attribution, and API integration.

### Problem Statement

Engineering teams often need to search technical documentation to troubleshoot service failures, understand operational procedures, and find relevant engineering guidelines.

Searching documents manually can be time-consuming. A knowledge chatbot can simplify this process by allowing users to ask questions in natural language.

### Objectives

- Ingest technical documentation in TXT and PDF formats.
- Split documents into smaller chunks for retrieval.
- Generate vector embeddings using a local embedding model.
- Store and search embeddings using FAISS.
- Retrieve relevant context for user questions.
- Generate answers using a locally hosted language model.
- Return source references for traceability.
- Abstain from answering when retrieval evidence appears insufficient.
- Expose the application through a REST API.
- Validate application behavior with automated tests.

## 2. Key Features

### Document Ingestion

The application reads supported text and PDF documents from the configured data directory.

Each extracted document includes its text, source filename, and page information when available.

### Text Chunking

Documents are divided into smaller text segments so that retrieval can identify relevant portions instead of passing entire documents to the language model.

Chunk metadata includes the source filename, page number where applicable, and chunk identifier.

### Embedding Generation

The application uses the `nomic-embed-text:latest` model through Ollama to convert document chunks and user questions into numerical vector representations.

These vectors capture semantic features that help retrieve relevant text even when the question does not use exactly the same words as the source document.

### Vector Search

FAISS stores the document embeddings and performs similarity search.

The current implementation normalizes embeddings and uses an inner-product index. With normalized vectors, inner-product similarity corresponds to cosine similarity, subject to normal floating-point precision.

### Answer Generation

The application sends the user's question and retrieved document context to `llama3:latest` through Ollama.

The prompt instructs the model to answer using the supplied context, avoid inventing procedures, and identify when evidence is insufficient.

### Source Attribution

The response includes source metadata for retrieved chunks, including filenames, chunk identifiers, optional page numbers, and similarity scores.

These references help users identify the documentation used during answer generation.

### Insufficient-Evidence Handling

The chatbot abstains when no chunks are retrieved or when the best retrieval score falls below the configured threshold.

This is a heuristic, not a guarantee that an answer is correct or that every unanswerable question will be rejected.

### REST API

FastAPI exposes endpoints for health checks and question answering. Interactive API documentation is available through Swagger UI.

## 3. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python 3.11 | Application development |
| API framework | FastAPI | REST API |
| API server | Uvicorn | Runs the FastAPI application |
| Local model runtime | Ollama | Hosts local language and embedding models |
| Generation model | `llama3:latest` | Generates answers |
| Embedding model | `nomic-embed-text:latest` | Creates document and query embeddings |
| Vector database/index | FAISS | Similarity search |
| PDF extraction | PyMuPDF | Extracts PDF text |
| HTTP client | Requests | Communicates with Ollama |
| Numerical operations | NumPy | Handles embedding vectors |
| Testing | pytest and HTTPX | Automated API and behavior tests |

## 4. Project Structure

```text
scenario2_chatbot/
├── __init__.py
├── README.md
├── architecture.md
├── api.py
├── chatbot.py
├── ingestion.py
├── prompts.py
├── retrieval.py
└── data/
    └── engineering_guide.txt

vector_index/
├── engineering.faiss
└── chunks.json

tests/
└── test_scenario2_chatbot.py
```

The `vector_index/` directory is located at the repository root, rather than inside `scenario2_chatbot/`.

The index and metadata files are generated artifacts. They can be recreated from the source documents using the ingestion and indexing pipeline.

## 5. Prerequisites

Install or configure the following:

- Python 3.11
- `uv`
- Ollama
- The required Ollama models
- Project dependencies

Verify Python and uv:

```powershell
python --version
uv --version
```

Verify Ollama:

```powershell
ollama --version
ollama list
```

Pull the required models if they are not already available:

```powershell
ollama pull llama3:latest
ollama pull nomic-embed-text:latest
```

Ensure the Ollama service is running before building the index or generating answers.

## 6. Installation

Run commands from the repository root.

If the project is configured with `pyproject.toml` and a uv lockfile, synchronize the environment:

```powershell
uv sync
```

If dependencies are managed through `requirements.txt`, create the environment and install dependencies:

```powershell
uv venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

If the virtual environment already exists and dependencies are installed, you can proceed to the next step.

## 7. Document Ingestion and Indexing

The sample engineering document is located at:

`scenario2_chatbot/data/engineering_guide.txt`

It contains example guidance covering:

- Service restarts
- HTTP 500 and HTTP 503 errors
- Database connectivity troubleshooting
- Incident escalation

The content is synthetic demonstration data.

Build or rebuild the vector index from the repository root:

```powershell
uv run python -m scenario2_chatbot.retrieval
```

The indexing pipeline performs these steps:

1. Extract text from supported documents.
2. Divide the extracted text into chunks.
3. Generate embeddings for the chunks.
4. Normalize the vectors.
5. Build a FAISS index.
6. Save the index and chunk metadata to `vector_index/`.

Rebuild the index whenever source documents or chunking configuration changes.

## 8. Running the Chatbot

### Command-Line Demonstration

Run:

```powershell
uv run python -m scenario2_chatbot.chatbot
```

The demonstration sends a sample engineering question through the retrieval and generation pipeline and prints the answer and its sources.

### Start the REST API

Run:

```powershell
uv run uvicorn scenario2_chatbot.api:app --reload
```

The API runs locally at:

`http://127.0.0.1:8000`

Interactive Swagger documentation:

`http://127.0.0.1:8000/docs`

Stop the server with `Ctrl+C`.

If another application is already using port 8000, stop that server or configure a different port.

## 9. API Documentation

### GET /health

Checks whether the API process is running.

Example:

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "service": "engineering-knowledge-chatbot"
}
```

This endpoint checks API availability; it does not by itself verify that Ollama and the vector index are ready.

### POST /chat

Accepts an engineering question and returns a generated response.

Example request:

```json
{
  "question": "What should I check if a database connection fails?"
}
```

Example response structure:

```json
{
  "answer": "Verify the database endpoint, network connectivity, credentials, and whether the database service is running. [Source: engineering_guide.txt, chunk chunk_1]",
  "sources": [
    {
      "source": "engineering_guide.txt",
      "chunk_id": "chunk_1",
      "page": null,
      "similarity": 0.662
    }
  ],
  "abstained": false
}
```

The response above illustrates the response structure; actual wording and scores may vary.

#### Response Fields

| Field | Description |
|---|---|
| `answer` | Generated answer or an insufficient-evidence message |
| `sources` | Retrieved source references and similarity scores |
| `source` | Source filename |
| `chunk_id` | Identifier of the retrieved chunk |
| `page` | PDF page number, or null for text files |
| `similarity` | Similarity score returned by retrieval |
| `abstained` | Indicates whether the chatbot declined to generate an answer because retrieval evidence was insufficient |

### Input Validation

The API validates incoming questions.

- Empty questions are rejected.
- Whitespace-only questions are rejected.
- Questions exceeding the configured length limit are rejected.
- Missing required fields are rejected.

FastAPI returns validation errors for invalid request bodies.

### Error Handling

If chatbot processing fails, the API returns a service error rather than exposing the underlying exception details to the caller.

For troubleshooting, verify that Ollama is running, the required models are installed, and the FAISS index exists.

## 10. Retrieval and Answer Generation

The current pipeline uses the following approach:

1. Embed the incoming question.
2. Search the FAISS index for the top three chunks.
3. Inspect the highest returned similarity score.
4. Abstain if no chunks are returned or the score is below the configured minimum.
5. Construct a prompt containing the question and retrieved context.
6. Generate an answer using Llama 3 with temperature set to zero.
7. Return the answer, retrieved source metadata, and abstention status.

The minimum similarity score is configured in `chatbot.py`.

A similarity score measures vector similarity, not the probability that the answer is correct. Thresholds must be evaluated against representative questions and labelled evidence before being used as a reliability control.

## 11. Prompt Design and Grounding

The prompt in `prompts.py` instructs the language model to:

- Answer using only the supplied document context.
- Avoid inventing technical procedures or specifications.
- Treat retrieved documents as reference data rather than instructions that override system rules.
- Identify insufficient evidence.
- Cite supporting sources.
- Avoid revealing secrets or credentials.

These instructions improve the intended behavior but do not provide a formal guarantee against hallucinations, prompt injection, or incorrect citations.

A production implementation should add independent citation checks and answer-quality evaluation.

## 12. Automated Testing

Run all repository tests:

```powershell
uv run python -m pytest tests -v
```

Run only Scenario 2 tests:

```powershell
uv run python -m pytest tests/test_scenario2_chatbot.py -v
```

The current test suite covers:

- Health endpoint behavior
- Successful chat API responses
- Question trimming
- Empty and whitespace-only questions
- Maximum question length
- Missing request fields
- API error handling
- Abstention when retrieval returns no chunks
- Abstention when similarity is below the configured threshold
- Empty-question handling in the chatbot

The current combined repository test suite contains 20 passing tests. Tests mock model-facing behavior where appropriate, so a passing test suite does not prove that live LLM answers are always correct.

## 13. Evaluation Strategy

A complete RAG evaluation should assess retrieval and generation separately.

### Retrieval Evaluation

Create a labelled dataset containing questions and their expected supporting document chunks.

Useful metrics include:

- Recall@k
- Precision@k
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG)

### Generation Evaluation

Evaluate whether generated answers:

- Are supported by retrieved context
- Correctly answer the question
- Include accurate citations
- Avoid unsupported claims
- Abstain when evidence is insufficient

A manually reviewed evaluation dataset is recommended before using automated LLM-as-judge metrics.

### Current Evaluation Status

The application has been functionally tested with a sample database-connectivity question. A formal benchmark with labelled retrieval and answer-quality metrics has not yet been established.

## 14. Security and Reliability Considerations

The current implementation is a local demonstration rather than a production service.

Before production use, consider:

- Authentication and authorization
- Rate limiting and request-size limits
- Secure file upload validation
- Protection against path traversal and malicious documents
- Prompt-injection defenses
- Logging that excludes passwords, secrets, and sensitive document content
- Monitoring of model availability, latency, and errors
- Evaluation of retrieval thresholds
- Independent validation of citations and answers
- Controlled document access and data retention policies

## 15. Known Limitations

- The supplied engineering guide is small and synthetic.
- Character-based chunking may split sentences or sections.
- Retrieval quality depends on the corpus, chunk boundaries, and embedding model.
- A similarity threshold is only a heuristic for insufficient evidence.
- Temperature zero does not make model outputs deterministic in every runtime configuration.
- Prompt instructions cannot completely prevent hallucinations.
- Source metadata identifies retrieved chunks but does not independently prove that every claim is supported.
- The current application does not include a production document-management interface or full security controls.

## 16. Future Improvements

Potential improvements include:

- Improve chunking with section-aware or sentence-aware boundaries.
- Make chunk size, overlap, top-k, and similarity threshold configurable.
- Add a labelled retrieval evaluation dataset.
- Add answer faithfulness, relevance, and citation correctness evaluation.
- Implement a secure document-upload and reindexing workflow.
- Add a readiness endpoint for Ollama, required models, and the index.
- Add structured logging and latency measurements.
- Add document versioning and incremental index updates.
- Add authentication and access controls.
- Containerize the application for repeatable deployment.

## 17. Conclusion

This project demonstrates an end-to-end local RAG workflow, from document ingestion and embedding generation to semantic retrieval, context-based answer generation, source attribution, and API testing.

It provides a foundation for exploring engineering knowledge assistants while clearly identifying the evaluation and security work required for production readiness.
