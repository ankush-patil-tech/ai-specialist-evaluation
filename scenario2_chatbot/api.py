
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from scenario2_chatbot.chatbot import answer_question

app = FastAPI(
    title="Engineering Knowledge Chatbot",
    description="A local RAG chatbot powered by Ollama and FAISS.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Question about the engineering documentation",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be empty or whitespace.")
        return value


@app.get("/health")
def health():
    return {"status": "healthy", "service": "engineering-knowledge-chatbot"}


@app.post("/chat")
def chat(request: ChatRequest):
    try:
        return answer_question(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The chatbot could not process the question. "
                   "Check that Ollama is running and the vector index exists.",
        ) from exc
