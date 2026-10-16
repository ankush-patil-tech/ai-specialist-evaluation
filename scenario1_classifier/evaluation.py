
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "sample_data" / "support_tickets.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["message", "category"]).copy()
    df["message"] = df["message"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["message"] != "") & (df["category"] != "")]

    print(f"Total samples: {len(df)}")
    print("\nSamples per category:")
    print(df["category"].value_counts())

    counts = df["category"].value_counts()

    if len(counts) < 2 or counts.min() < 5:
        raise ValueError(
            "Each category needs at least 5 examples "
            "for five-fold cross-validation."
        )

    X = df["message"]
    y = df["category"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = {
        "Majority baseline": DummyClassifier(
            strategy="most_frequent"
        ),
        "TF-IDF + Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]),
    }

    for name, model in models.items():
        predictions = cross_val_predict(
            model, X, y, cv=cv
        )

        print(f"\n{'=' * 55}")
        print(name)
        print("=" * 55)

        print(f"Accuracy: {accuracy_score(y, predictions):.3f}")
        print(
            f"Macro F1: "
            f"{f1_score(y, predictions, average='macro'):.3f}"
        )
        print(
            f"Macro precision: "
            f"{precision_score(y, predictions, average='macro', zero_division=0):.3f}"
        )
        print(
            f"Macro recall: "
            f"{recall_score(y, predictions, average='macro', zero_division=0):.3f}"
        )

        print("\nClassification report:")
        print(classification_report(y, predictions, zero_division=0))

        labels = sorted(y.unique())
        matrix = confusion_matrix(y, predictions, labels=labels)

        print("Confusion matrix (rows=actual, columns=predicted):")
        print(pd.DataFrame(matrix, index=labels, columns=labels))


if __name__ == "__main__":
    main()
