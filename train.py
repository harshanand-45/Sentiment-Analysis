"""
train.py
--------
End-to-end training pipeline for the Sentiment Analysis project.

Steps performed:
1. Load dataset (data/dataset.csv)
2. Validate dataset (columns, missing values, class labels)
3. Clean/preprocess text using src/preprocessing.py
4. Split into train/test sets
5. Convert text to TF-IDF features
6. Train a Logistic Regression classifier
7. Evaluate the model (accuracy, precision, recall, F1, confusion matrix)
8. Save the trained model, TF-IDF vectorizer, and evaluation report

Run with:  python train.py
"""

import os
import sys
import json
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works without a display
import matplotlib.pyplot as plt

from src.preprocessing import clean_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42          # for reproducibility
TEST_SIZE = 0.2            # 20% of data held out for testing
VALID_LABELS = {"Positive", "Neutral", "Negative"}

DATA_PATH = os.path.join("data", "dataset.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_report.json")
CONFUSION_MATRIX_IMG = os.path.join(MODEL_DIR, "confusion_matrix.png")


def load_dataset(path: str) -> pd.DataFrame:
    """Load the dataset CSV and return it as a DataFrame, with error handling."""
    if not os.path.exists(path):
        print(f"ERROR: Dataset not found at '{path}'.")
        print("Run 'python data/generate_dataset.py' first, or place your own")
        print("dataset.csv (with 'text' and 'label' columns) in the data/ folder.")
        sys.exit(1)

    df = pd.read_csv(path)
    return df


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the dataset has the required structure:
    - Must contain 'text' and 'label' columns
    - Labels must be one of Positive / Neutral / Negative
    - Drops rows with missing/empty text
    """
    required_columns = {"text", "label"}
    if not required_columns.issubset(set(df.columns)):
        print(f"ERROR: Dataset must contain columns {required_columns}.")
        print(f"Found columns: {list(df.columns)}")
        sys.exit(1)

    # Drop rows with missing text or label
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    after = len(df)
    if after < before:
        print(f"Dropped {before - after} rows with missing/empty text or label.")

    # Normalize label capitalization (e.g. "positive" -> "Positive")
    df["label"] = df["label"].astype(str).str.strip().str.capitalize()

    # Check for unexpected labels
    unexpected = set(df["label"].unique()) - VALID_LABELS
    if unexpected:
        print(f"ERROR: Found unexpected labels in dataset: {unexpected}")
        print(f"Expected only: {VALID_LABELS}")
        sys.exit(1)

    if len(df) < 50:
        print(f"WARNING: Dataset only has {len(df)} rows. "
              f"The project spec recommends at least 2,000 samples.")

    return df.reset_index(drop=True)


def main():
    print("=" * 60)
    print("STEP 1: Loading dataset")
    print("=" * 60)
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    print("\n" + "=" * 60)
    print("STEP 2: Validating dataset")
    print("=" * 60)
    df = validate_dataset(df)
    print("Class distribution:")
    print(df["label"].value_counts())

    print("\n" + "=" * 60)
    print("STEP 3: Cleaning / preprocessing text")
    print("=" * 60)
    df["cleaned_text"] = df["text"].apply(clean_text)
    # Remove rows that became empty after cleaning (e.g. text was only URLs/symbols)
    empty_after_clean = (df["cleaned_text"].str.strip() == "").sum()
    if empty_after_clean > 0:
        print(f"Dropping {empty_after_clean} rows that became empty after cleaning.")
        df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    print("Sample cleaned text:")
    print(df[["text", "cleaned_text"]].head(3).to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 4: Splitting into train/test sets")
    print("=" * 60)
    X = df["cleaned_text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # keep class balance consistent across train/test
    )
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    print("\n" + "=" * 60)
    print("STEP 5: TF-IDF Vectorization")
    print("=" * 60)
    # TF-IDF parameter choices explained:
    # - max_features=5000: caps vocabulary size to keep the model small/fast
    #   while still capturing the most informative words.
    # - ngram_range=(1, 2): includes both single words and two-word phrases
    #   (e.g. "not good"), which helps capture negation and short phrases.
    # - min_df=2: ignores words that appear in fewer than 2 documents (noise/typos).
    # - max_df=0.9: ignores words that appear in more than 90% of documents
    #   (too common to be useful, similar to stopwords).
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")

    print("\n" + "=" * 60)
    print("STEP 6: Training Logistic Regression model")
    print("=" * 60)
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        class_weight="balanced",  # helps if classes are slightly imbalanced
    )
    model.fit(X_train_tfidf, y_train)
    print("Model training complete.")

    print("\n" + "=" * 60)
    print("STEP 7: Evaluating the model")
    print("=" * 60)
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    report_text = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(VALID_LABELS))

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall   : {recall:.4f} (weighted)")
    print(f"F1-score : {f1:.4f} (weighted)")
    print("\nFull classification report:")
    print(report_text)
    print("Confusion matrix (rows=actual, cols=predicted):")
    labels_sorted = sorted(VALID_LABELS)
    print("Labels order:", labels_sorted)
    print(cm)

    print("\n" + "=" * 60)
    print("STEP 8: Saving model, vectorizer, and evaluation report")
    print("=" * 60)
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Saved TF-IDF vectorizer to {VECTORIZER_PATH}")

    evaluation_summary = {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels_sorted,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=2)
    print(f"Saved evaluation report to {METRICS_PATH}")

    # Save a confusion matrix plot image for the README / demo
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted)
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion Matrix")
    for i in range(len(labels_sorted)):
        for j in range(len(labels_sorted)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_IMG)
    print(f"Saved confusion matrix image to {CONFUSION_MATRIX_IMG}")

    print("\nTraining pipeline finished successfully!")


if __name__ == "__main__":
    main()
