"""
train_model.py

Trains a TF-IDF + Logistic Regression fake-news classifier from
dataset/news.csv and saves the trained vectorizer + model to
model/fake_news_model.pkl using joblib.

Run with:
    python train_model.py

Required dataset format (dataset/news.csv):
    title,text,label

    - title : the news headline (optional per-row, but column should exist)
    - text  : the full article content
    - label : REAL or FAKE (case-insensitive; 1/0 and TRUE/FALSE are also
              tolerated, see `normalize_label`)

This script performs NO fabrication of results: accuracy, precision,
recall, F1-score and the confusion matrix are all computed from the
actual train/test split of the dataset you provide.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from model import preprocess_text

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "news.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"

# Column name variations we will try to accommodate automatically.
TITLE_COLUMNS = ["title", "headline", "news_title"]
TEXT_COLUMNS = ["text", "content", "article", "news_text", "body"]
LABEL_COLUMNS = ["label", "class", "target", "category"]


def find_column(df: pd.DataFrame, candidates):
    """Return the first matching column name (case-insensitive) or None."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def normalize_label(value) -> str:
    """
    Normalize a variety of possible label representations into
    'REAL' or 'FAKE'. Raises ValueError for unrecognized values so that
    bad rows can be reported rather than silently mislabeled.
    """
    text = str(value).strip().lower()
    if text in ("real", "true", "1", "1.0"):
        return "REAL"
    if text in ("fake", "false", "0", "0.0"):
        return "FAKE"
    raise ValueError(f"Unrecognized label value: {value!r}")


def load_and_validate_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        print("=" * 70)
        print("ERROR: Dataset not found.")
        print(f"Expected a CSV file at: {DATASET_PATH}")
        print()
        print("Please place a labelled fake/real news dataset at:")
        print("    dataset/news.csv")
        print()
        print("Preferred CSV format:")
        print("    title,text,label")
        print('    "Example headline","Example article content...","REAL"')
        print('    "False claim example","Example misleading article...","FAKE"')
        print("=" * 70)
        sys.exit(1)

    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Could not read '{DATASET_PATH}'. Details: {exc}")
        sys.exit(1)

    if df.empty:
        print(f"ERROR: '{DATASET_PATH}' is empty. Please provide a populated dataset.")
        sys.exit(1)

    title_col = find_column(df, TITLE_COLUMNS)
    text_col = find_column(df, TEXT_COLUMNS)
    label_col = find_column(df, LABEL_COLUMNS)

    if text_col is None or label_col is None:
        print("=" * 70)
        print("ERROR: Required columns not found in dataset/news.csv")
        print(f"Detected columns: {list(df.columns)}")
        print()
        print("Your CSV must contain, at minimum, a text/article column and a")
        print("label column. Preferred column names:")
        print("    title,text,label")
        print("=" * 70)
        sys.exit(1)

    # Build a clean working frame with standardized column names.
    working = pd.DataFrame()
    working["title"] = df[title_col] if title_col else ""
    working["text"] = df[text_col]
    working["label_raw"] = df[label_col]

    # Handle missing values.
    working["title"] = working["title"].fillna("")
    working["text"] = working["text"].fillna("")

    before = len(working)
    working = working[working["text"].str.strip() != ""]
    dropped_empty = before - len(working)
    if dropped_empty:
        print(f"Note: dropped {dropped_empty} row(s) with empty article text.")

    # Normalize labels, dropping unrecognized rows rather than guessing.
    normalized_labels = []
    valid_mask = []
    for value in working["label_raw"]:
        try:
            normalized_labels.append(normalize_label(value))
            valid_mask.append(True)
        except ValueError:
            normalized_labels.append(None)
            valid_mask.append(False)

    working["label"] = normalized_labels
    dropped_invalid = (~pd.Series(valid_mask)).sum()
    if dropped_invalid:
        print(f"Note: dropped {dropped_invalid} row(s) with unrecognized label values.")
    working = working[pd.Series(valid_mask).values]

    if working.empty:
        print("ERROR: No valid labelled rows remain after cleaning. Please check your dataset.")
        sys.exit(1)

    if working["label"].nunique() < 2:
        print("ERROR: Dataset must contain both REAL and FAKE examples to train a classifier.")
        sys.exit(1)

    return working.reset_index(drop=True)


def main():
    print("=" * 70)
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("=" * 70)

    df = load_and_validate_dataset()
    print(f"Loaded {len(df)} labelled rows from dataset/news.csv")
    print(f"Class distribution:\n{df['label'].value_counts().to_string()}")

    # Combine title + text, then clean using the SAME preprocessing function
    # used at prediction time (imported from model.py) so training and
    # inference are guaranteed to be consistent.
    print("\nCombining headline + article and cleaning text...")
    combined_text = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()
    df["clean_text"] = combined_text.apply(preprocess_text)

    X = df["clean_text"]
    y = df["label"]

    print("Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression classifier...")
    classifier = LogisticRegression(max_iter=1000, C=1.0)
    classifier.fit(X_train_vec, y_train)

    print("Evaluating on held-out test set...")
    y_pred = classifier.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="FAKE", zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label="FAKE", zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label="FAKE", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"])

    print("\n" + "=" * 70)
    print("Training completed.")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision (FAKE class): {precision * 100:.2f}%")
    print(f"Recall (FAKE class): {recall * 100:.2f}%")
    print(f"F1-score (FAKE class): {f1 * 100:.2f}%")
    print("Confusion Matrix (rows=actual, cols=predicted) [REAL, FAKE]:")
    print(cm)
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "vectorizer": vectorizer,
        "model": classifier,
        "metrics": {
            "accuracy": round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "test_size": len(X_test),
            "train_size": len(X_train),
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")
    print("You can now run the web application with: python app.py")


if __name__ == "__main__":
    main()
