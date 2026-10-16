
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from scenario1_classifier.predict import load_model, predict_ticket


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    yield


app = FastAPI(
    title="Customer Support Classifier",
    description="Classify customer messages into support categories.",
    version="1.0.0",
    lifespan=lifespan,
)


from pydantic import BaseModel, Field, field_validator


class TicketRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Message cannot be empty or whitespace only.")

        return value


class TicketResponse(BaseModel):
    category: str
    confidence: float
    needs_human_review: bool


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/tickets/classify", response_model=TicketResponse)
def classify_ticket(request: TicketRequest):
    try:
        return predict_ticket(
            request.message,
            model=app.state.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
