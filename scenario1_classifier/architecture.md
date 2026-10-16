
# Customer Support Classifier — Architecture

## Components

1. **API layer:** FastAPI accepts incoming ticket messages.
2. **Validation layer:** Pydantic validates message type, length,
   and whitespace.
3. **Feature extraction:** TF-IDF converts text into numerical
   unigram and bigram features.
4. **Classification:** Logistic Regression predicts one of the
   supported ticket categories.
5. **Confidence handling:** The maximum predicted class probability
   is compared with the initial threshold of 0.60.
6. **Human review:** Predictions below the threshold are flagged
   for review. The current prototype returns a flag; it does not
   implement a persistent review queue.
7. **Response layer:** FastAPI returns the category, confidence,
   and review flag as JSON.

## Request Flow

```text
Client
  |
  v
POST /tickets/classify
  |
  v
Pydantic Validation
  |
  v
TF-IDF + Logistic Regression
  |
  v
Confidence Threshold Check
  |
  +---- Confidence < 0.60 ----> needs_human_review = true
  |
  +---- Confidence >= 0.60 ---> needs_human_review = false
  |
  v
JSON Response
```

## Model Lifecycle

The training script loads the labeled CSV, splits the data,
trains the pipeline, evaluates the held-out split, and saves
the artifact using Joblib.

The API loads the saved artifact during application startup.

The evaluation script separately compares the classifier against
a majority-class baseline using five-fold stratified
cross-validation.

## Production Considerations

Before production deployment, validate performance on representative
real-world data, calibrate probabilities, tune the review threshold,
add authentication and rate limiting, and monitor model drift.

Human review is represented by a response flag in this prototype;
a real deployment would need a review workflow and audit trail.
