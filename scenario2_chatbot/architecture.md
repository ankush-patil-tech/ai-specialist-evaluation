
# Scenario 2 Architecture: Engineering Knowledge Chatbot (RAG)

## 1. Purpose

This document describes the architecture, processing flow, components, data structures, design decisions, and limitations of the Engineering Knowledge Chatbot.

The application uses Retrieval-Augmented Generation (RAG) to answer engineering questions using a local document collection.

The primary design goal is to retrieve relevant source material before generating an answer, allowing the application to return source references and attempt to avoid unsupported responses.

## 2. High-Level Architecture

The system consists of five logical layers:

1. Document ingestion
2. Embedding generation and indexing
3. Semantic retrieval
4. Prompt construction and answer generation
5. API and response handling

### High-Level Component Diagram

```text
                  DOCUMENT INGESTION
                           |
                +----------+----------+
                |                     |
             TXT files             PDF files
                |                     |
                +----------+----------+
                           |
                           v
                   Text Extraction
                           |
                           v
                      Chunking
                           |
                           v
                 Embedding Generation
                   Ollama / Nomic
                           |
                           v
                    Vector Normalization
                           |
                           v
                      FAISS Index
                           |
                    +------+------+
                    |             |
                    v             |
               User Question      |
                    |             |
                    v             |
             Query Embedding      |
                    |             |
                    v             |
               FAISS Search ------+
                    |
                    v
             Retrieved Chunks
                    |
                    v
             Evidence Threshold
                /         \
        Insufficient      Sufficient
             |                 |
             v                 v
          Abstain         Prompt Builder
                                |
                                v
                          Ollama / Llama 3
                                |
                                v
                         Generated Answer
                                |
                                v
                     Answer + Source Metadata
                                |
                                v
                          FastAPI Response
```

## 3. Technology Components

| Component | Responsibility |
|---|---|
| FastAPI | Exposes health and chat endpoints |
| Pydantic | Validates incoming API request data |
| Uvicorn | Runs the API server |
| Ingestion module | Extracts text from supported source files |
| Chunking function | Splits extracted text and attaches metadata |
| Ollama | Runs the local embedding and generation models |
| Nomic embedding model | Encodes documents and questions as vectors |
| NumPy | Stores and prepares embedding matrices |
| FAISS | Builds the vector index and performs similarity search |
| Prompt builder | Combines model instructions, context, and question |
| Llama 3 | Generates the final natural-language answer |
| pytest and HTTPX | Test API and chatbot behavior |

## 4. Module Responsibilities

### `ingestion.py`

Responsible for document extraction and chunking.

Main responsibilities:

- Discover supported files in the configured data directory.
- Extract text from TXT files.
- Extract text and page information from PDF files.
- Split extracted text into chunks.
- Attach source metadata to each chunk.
- Return structured chunks for embedding generation.

Typical chunk structure:

```python
{
    "text": "Verify the database endpoint and network connectivity.",
    "source": "engineering_guide.txt",
    "page": None,
    "chunk_id": "chunk_1"
}
```

The exact chunk contents depend on the source document and chunking configuration.

### `retrieval.py`

Responsible for embedding generation, vector indexing, and semantic retrieval.

Main responsibilities:

- Send document text to Ollama's embedding endpoint.
- Convert embeddings into a NumPy float32 matrix.
- Normalize embedding vectors.
- Build a FAISS inner-product index.
- Persist the index and corresponding chunk metadata.
- Embed incoming questions.
- Retrieve the highest-scoring chunks.

The current implementation uses:

- Embedding model: `nomic-embed-text:latest`
- Index type: `faiss.IndexFlatIP`
- Default retrieval count: top three chunks

### `prompts.py`

Responsible for constructing the model prompt.

The prompt contains:

- System-level instructions
- Retrieved document context
- Source references
- The user's question
- Instructions for evidence-based answering and citations

Retrieved text is treated as untrusted reference content. Prompt wording helps guide model behavior but is not a complete security boundary.

### `chatbot.py`

Coordinates the question-answering workflow.

Main responsibilities:

1. Validate that the question is not empty.
2. Retrieve matching document chunks.
3. Handle empty retrieval results.
4. Compare the highest retrieval score against the configured threshold.
5. Abstain when retrieval evidence is considered insufficient.
6. Construct the prompt.
7. Call the local generation model through Ollama.
8. Collect source metadata.
9. Return the answer and abstention status.

### `api.py`

Exposes the chatbot to external clients through FastAPI.

Current endpoints:

- `GET /health`
- `POST /chat`

The chat endpoint validates the request, calls the chatbot service, and returns structured JSON. Processing failures are translated into service errors.

## 5. Document Ingestion Flow

The ingestion pipeline starts from the configured document directory:

`scenario2_chatbot/data/`

### Processing Steps

1. Discover supported TXT and PDF files.
2. Extract the text from each file.
3. Preserve source filenames.
4. Preserve PDF page information when available.
5. Split document text into smaller chunks.
6. Assign chunk identifiers.
7. Pass chunk text to the embedding model.
8. Associate each embedding with its chunk metadata.

### Chunking Strategy

The current implementation uses character-based chunking with configurable chunk size and overlap.

The initial configuration uses:

- Chunk size: 800 characters
- Chunk overlap: 150 characters

Overlap helps preserve context around adjacent chunk boundaries.

Character-based chunking is simple to implement, but it may split sentences, headings, or technical instructions. Section-aware or sentence-aware chunking may improve retrieval quality.

## 6. Embedding and Index Creation

Each document chunk is converted into a vector using the local Nomic embedding model.

Let a chunk embedding be represented by vector \(v\). The application normalizes the vector before adding it to FAISS:

\[
\hat{v} = \frac{v}{\|v\|_2}
\]

The index uses inner-product search. For normalized vectors, the inner product between two vectors corresponds to cosine similarity:

\[
\operatorname{similarity}(q,d)=\hat{q}^{T}\hat{d}
\]

where:

- \(q\) is the question embedding
- \(d\) is a document-chunk embedding

This makes the returned score useful for ranking chunks by vector similarity. It does not make the score a probability of answer correctness.

### Persistent Artifacts

The current implementation saves the index and chunk metadata under the repository-level directory:

```text
vector_index/
├── engineering.faiss
└── chunks.json
```

The FAISS index stores the vectors. The JSON file stores the text and metadata needed to map returned index positions back to their source chunks.

These artifacts must remain consistent. If the source corpus or chunking configuration changes, rebuild the index and metadata together.

## 7. Query Retrieval Flow

When a user submits a question:

1. The question is validated.
2. The embedding model converts the question into a vector.
3. The vector is normalized.
4. FAISS searches the index.
5. The top matching chunk positions and scores are returned.
6. The application maps those positions to stored chunk metadata.
7. The retrieved chunks are passed to the chatbot orchestration layer.

The current default is top-k retrieval with \(k=3\), bounded by the number of indexed chunks.

The retrieved chunks may overlap or contain redundant content. Retrieval quality should be evaluated on a representative set of questions rather than inferred from a few successful examples.

## 8. Evidence Threshold and Abstention

The chatbot uses a configurable minimum similarity threshold.

Current configuration:

```python
MIN_SIMILARITY = 0.35
```

The decision is conceptually:

```text
No retrieved chunks?
    |
    +-- Yes --> Return an abstention response
    |
    +-- No --> Is the highest score below the threshold?
                    |
                    +-- Yes --> Return an abstention response
                    |
                    +-- No --> Construct prompt and generate answer
```

This is a simple retrieval-based heuristic.

A high similarity score does not guarantee that the retrieved chunks contain enough information to answer a question. Conversely, a relevant chunk may receive a low score due to wording, domain terminology, or embedding behavior.

Threshold selection should be based on labelled evaluation data, including answerable and unanswerable questions.

## 9. Prompt Construction and Generation

The prompt builder combines:

1. System instructions
2. Retrieved document context
3. Source filename and chunk identifiers
4. The user question
5. Explicit instructions to answer from the context and acknowledge insufficient evidence

The chatbot calls Ollama's generation endpoint using `llama3:latest`. The current configuration uses temperature zero to reduce sampling variability.

The generated answer is returned with the retrieved source metadata.

### Grounding Limitations

The model may still:

- Introduce unsupported details
- Misinterpret retrieved text
- Cite a retrieved source that does not support every claim
- Fail to abstain when evidence is insufficient

Prompt instructions alone cannot guarantee faithful answers. Independent answer and citation evaluation is needed for stronger reliability.

## 10. API Architecture

### Health Endpoint

`GET /health`

Returns a simple service-health response.

This endpoint confirms that the API process responds. It does not verify that the embedding model, generation model, or vector index is available.

### Chat Endpoint

`POST /chat`

Expected request:

```json
{
  "question": "What should I check if a database connection fails?"
}
```

The API validates the question and calls the chatbot orchestration function.

The response contains:

- Generated answer
- Source metadata
- Similarity scores
- Abstention status

### Error Handling

The API rejects invalid request data through Pydantic validation.

Errors during chatbot processing are converted into a service-unavailable response. The response avoids returning raw exception details to the client.

Operational logging should be added to help distinguish between unavailable Ollama services, missing index artifacts, and other processing failures.

## 11. Testing Architecture

The automated tests use pytest and FastAPI's TestClient.

The test suite separates API contract tests from tests of chatbot decision branches.

### API Tests

Verify:

- Health endpoint response
- Valid chat request handling
- Whitespace trimming
- Empty question rejection
- Maximum question length
- Missing request fields
- Service failure handling

### Chatbot Tests

Verify:

- Abstention when retrieval returns no chunks
- Abstention when the best similarity score is below the configured threshold
- Rejection of empty questions

The model-facing dependencies are mocked where appropriate. This keeps the tests fast and avoids requiring live model inference for every test.

Passing these tests verifies the tested software behavior, not the factual accuracy of all model-generated answers.

## 12. Evaluation Architecture

A robust RAG evaluation should include separate retrieval and generation evaluation.

### Retrieval Metrics

Use labelled questions with expected supporting chunks to measure:

- Recall@k
- Precision@k
- Mean Reciprocal Rank
- Normalized Discounted Cumulative Gain

### Generation Metrics

Measure:

- Answer correctness
- Faithfulness to retrieved context
- Answer relevance
- Citation correctness
- Appropriate abstention
- Unsupported-claim rate

Human-reviewed examples should be used to validate automated evaluation results.

The current project demonstrates functional retrieval and answer generation but does not yet provide a formal benchmark of these metrics.

## 13. Security Considerations

The application is a local demonstration and is not production-hardened.

Important considerations include:

- Retrieved documents may contain prompt-injection attempts.
- Uploaded files, if upload support is added, must be validated and size-limited.
- File paths must be handled safely.
- Sensitive information must not be written to diagnostic logs.
- API endpoints need authentication and rate limiting before being exposed beyond a trusted local environment.
- Document access permissions must be enforced if the corpus contains restricted information.

A prompt telling the model to ignore malicious document instructions is helpful but does not replace application-level security controls.

## 14. Performance Considerations

The application runs the models locally through Ollama.

Performance depends on:

- Available RAM and GPU resources
- Model size
- Document count and chunk count
- Embedding generation time
- Number of retrieved chunks
- Prompt length
- Generation length

The current flat FAISS index performs exact similarity search. This is suitable for a small demonstration corpus. Larger corpora may require a more scalable index, batching, caching, or a dedicated vector database.

## 15. Failure Scenarios

| Failure | Expected behavior | Recommended operational check |
|---|---|---|
| Empty question | Reject input | Check request validation |
| No matching chunks returned | Abstain | Verify index contents and retrieval behavior |
| Similarity below threshold | Abstain | Evaluate threshold against labelled examples |
| Ollama unavailable | Return service error | Verify Ollama service status |
| Required model unavailable | Return service error | Check installed model names |
| Index or metadata missing | Return service error | Rebuild the index |
| Poor answer quality | May return a weak answer | Inspect retrieved context and evaluate the model response |

## 16. Future Architecture Improvements

Potential extensions include:

- Sentence-aware or section-aware chunking
- Configurable retrieval and generation parameters
- Hybrid keyword and vector retrieval
- Reranking retrieved chunks
- A labelled RAG evaluation pipeline
- Independent citation validation
- Secure upload and reindexing endpoints
- Health and readiness checks for model and index dependencies
- Structured logs and latency metrics
- Model and index versioning
- Authentication, authorization, and request limits
- Containerized deployment

## 17. Summary

The architecture demonstrates an end-to-end local RAG pipeline: document extraction, chunking, embedding generation, vector indexing, semantic retrieval, prompt construction, local language-model inference, source attribution, and REST API exposure.

The design prioritizes local execution, traceability, and a clear separation of responsibilities. The current implementation is suitable for demonstrating RAG concepts and software engineering practices, while formal answer-quality evaluation and production security remain areas for improvement.
