
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "sample_data" / "support_tickets.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "ticket_classifier.joblib"


def main():
    # 1. Load the dataset
    df = pd.read_csv(DATA_PATH)

    # 2. Validate required columns and remove invalid rows
    df = df.dropna(subset=["message", "category"])
    df["message"] = df["message"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[df["message"] != ""]

    if df["category"].nunique() < 2:
        raise ValueError("At least two categories are required.")

    if df["category"].value_counts().min() < 2:
        raise ValueError("Each category needs at least two samples.")

    X = df["message"]
    y = df["category"]

    # 3. Split data while preserving category proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=y,
    )

    # 4. Build the ML pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    # 5. Train the model
    model.fit(X_train, y_train)

    # 6. Evaluate on held-out data
    predictions = model.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # 7. Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
