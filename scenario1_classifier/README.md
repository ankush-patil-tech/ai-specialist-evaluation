
# AI Specialist Evaluation — Customer Support Ticket Classifier

## 1. Overview

This project implements a machine learning system that automatically classifies customer support messages into three categories:

- `billing`
- `technical_issue`
- `general_inquiry`

The application exposes predictions through a FastAPI REST API and flags low-confidence predictions for human review.

## 2. Architecture

```text
Support Message
      |
      v
FastAPI Request Validation
      |
      v
TF-IDF Text Vectorization
      |
      v
Logistic Regression Classifier
      |
      v
Predicted Category + Confidence
      |
      v
Human Review Decision
      |
      v
JSON Response
```

See [architecture.md](architecture.md) for the detailed architecture and design decisions.

## 3. Technology Stack

- Python 3.11
- scikit-learn
- pandas
- FastAPI
- Pydantic
- Joblib
- pytest
- HTTPX
- uv for environment and dependency management

## 4. Project Structure

```text
ai-specialist-evaluation/
├── sample_data/
│   └── support_tickets.csv
├── models/
│   └── ticket_classifier.joblib
├── scenario1_classifier/
│   ├── __init__.py
│   ├── README.md
│   ├── architecture.md
│   ├── api.py
│   ├── train.py
│   ├── predict.py
│   └── evaluation.py
└── tests/
    └── test_classifier_api.py
```

The model artifact is generated during training and may be excluded from version control. The directory structure above shows the relevant files for this scenario, not the complete repository.

## 5. Dataset

The demonstration dataset is stored at:

`sample_data/support_tickets.csv`

It contains 60 synthetic messages, with 20 examples per category. These examples were created for evaluation purposes and do not represent real customer data.

Synthetic data is useful for demonstrating the training, inference, and evaluation workflow, but it does not establish real-world model performance.

## 6. Model Design

The classifier uses a scikit-learn Pipeline containing:

1. `TfidfVectorizer` with unigram and bigram features.
2. `LogisticRegression` for multiclass classification.

TF-IDF converts text into numerical features based on term importance. Logistic Regression uses these features to predict the message category.

The trained pipeline is saved to:

`models/ticket_classifier.joblib`

The prediction component loads the saved pipeline and returns the predicted category, confidence information, and human-review recommendation.

### Human Review Logic

The current demonstration threshold is `0.60`.

- Maximum predicted class probability below `0.60`: recommend human review.
- Maximum predicted class probability at or above `0.60`: no review is recommended by this rule.

This threshold is configurable in the implementation. The predicted probability is not a guarantee of correctness, and the threshold should be validated using representative data before production use.

## 7. Setup

### Prerequisites

- Python 3.11
- `uv` installed

From the repository root, create and activate the environment in Windows PowerShell:

```powershell
uv venv .venv --python 3.11
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
uv pip install -r requirements.txt
```

If your project uses a configured `pyproject.toml` and lockfile, you can instead use `uv sync`.

## 8. Train the Model

Run from the repository root:

```powershell
uv run python scenario1_classifier/train.py
```

The training script:

1. Loads the labelled dataset.
2. Splits the data into stratified training and test sets.
3. Fits the TF-IDF and Logistic Regression pipeline.
4. Reports classification metrics on the held-out test set.
5. Saves the fitted pipeline to the `models/` directory.

The current configuration uses a stratified 60/40 train-test split.

## 9. Run Evaluation

```powershell
uv run python scenario1_classifier/evaluation.py
```

The evaluation script compares a most-frequent-class baseline against TF-IDF + Logistic Regression using five-fold stratified cross-validation.

Metrics include:

- Accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class precision, recall, and F1
- Confusion matrix

The baseline provides a reference point for assessing whether the trained model learns useful patterns beyond predicting the most common class.

## 10. Start the API

Run from the repository root:

```powershell
uv run uvicorn scenario1_classifier.api:app --reload
```

The API runs locally at:

`http://127.0.0.1:8000`

Interactive API documentation:

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/tickets/classify` | Classify a support message |

### Example Request

```json
{
  "message": "I was charged twice for my subscription"
}
```

### Example Response

```json
{
  "category": "billing",
  "confidence": 0.85,
  "needs_human_review": false
}
```

The response is illustrative; the actual prediction and confidence depend on the trained model and input message. Refer to the API schema for the exact response fields.

### Input Validation

The API validates incoming messages and rejects invalid inputs, including:

- Empty messages
- Whitespace-only messages
- Messages exceeding the configured length limit
- Missing required fields
- Invalid request bodies

Invalid requests receive appropriate HTTP validation errors.

## 11. Automated Tests

Run the Scenario 1 tests:

```powershell
uv run python -m pytest tests/test_classifier_api.py -v
```

Run all repository tests:

```powershell
uv run python -m pytest tests -v
```

The Scenario 1 tests cover:

- Health endpoint behavior
- Valid ticket classification
- Empty and whitespace-only messages
- Oversized messages
- Missing message fields
- Invalid request bodies
- Low-confidence predictions requiring human review
- High-confidence predictions not triggering review

The combined repository test suite currently contains 20 passing tests across both scenarios.

## 12. Evaluation Results

The dataset contains 60 synthetic examples, evenly distributed across the three categories.

### Five-Fold Stratified Cross-Validation

| Metric | Most-Frequent Baseline | TF-IDF + Logistic Regression |
|---|---:|---:|
| Accuracy | 0.333 | 0.867 |
| Macro Precision | 0.111 | 0.866 |
| Macro Recall | 0.333 | 0.867 |
| Macro F1 | 0.167 | 0.866 |

### Separate Holdout Evaluation

A stratified 60/40 train-test split produced:

- Test examples: 24
- Accuracy: 0.917

These results come from different evaluation procedures and should not be compared as though they were measured on the same test examples.

The results demonstrate the model's behavior on this small synthetic dataset. They do not establish expected performance on real customer support tickets.

## 13. Risks and Limitations

- Synthetic data may not represent real customer language or ticket distributions.
- Ambiguous messages may belong to multiple categories.
- New vocabulary and changes in ticket patterns may reduce performance.
- Low confidence does not necessarily mean a prediction is wrong.
- High confidence does not guarantee a correct prediction.
- The current `0.60` review threshold is a demonstration setting.
- Probability calibration has not been established.
- There is no production monitoring or automated retraining workflow.
- The API does not currently implement authentication or rate limiting.

## 14. Future Improvements

- Evaluate with a larger, representative, human-labelled dataset.
- Calibrate predicted probabilities using validation data.
- Tune the human-review threshold against operational costs.
- Monitor category distribution and model drift.
- Compare against transformer-based classification models.
- Add model versioning and a controlled retraining workflow.
- Add authentication, rate limiting, structured logging, and deployment configuration.
- Track model performance using real-world feedback and reviewed predictions.

## 15. Conclusion

This scenario demonstrates a complete baseline text-classification workflow, from labelled data and model training to evaluation, API-based inference, confidence-based review decisions, and automated testing.

It provides a foundation for a customer support classification service while identifying the additional data, calibration, monitoring, and security work needed before production deployment.
