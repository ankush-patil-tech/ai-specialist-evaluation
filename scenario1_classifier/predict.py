
from pathlib import Path

import joblib


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "ticket_classifier.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run scenario1_classifier/train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_ticket(message: str, model=None) -> dict:
    message = message.strip()

    if not message:
        raise ValueError("Message cannot be empty.")

    if model is None:
        model = load_model()

    # Predict category and class probabilities
    category = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    confidence = float(max(probabilities))

    # Initial demo threshold; tune this using validation data
    review_threshold = 0.60

    return {
        "category": category,
        "confidence": round(confidence, 4),
        "needs_human_review": confidence < review_threshold,
    }


if __name__ == "__main__":
    messages = [
        "I was charged twice for my subscription",
        "My application crashes whenever I open it",
        "What are your customer support hours?",
    ]

    model = load_model()

    for message in messages:
        result = predict_ticket(message, model)

        print(f"\nMessage: {message}")
        print(f"Prediction: {result}")
