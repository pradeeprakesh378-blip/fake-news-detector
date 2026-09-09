"""
model.py

Reusable Machine Learning helper functions for the Fake News Detection
Flask application.

Responsibilities:
- Load the trained (vectorizer + classifier) bundle saved by train_model.py
- Preprocess raw text in EXACTLY the same way it was preprocessed during training
- Run a prediction and return a structured result

This module intentionally contains NO Flask-specific code so that it can be
reused / unit tested independently of the web layer.
"""

import re
import string
from pathlib import Path

import joblib

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_news_model.pkl"


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class ModelNotFoundError(Exception):
    """Raised when the trained model file does not exist on disk."""
    pass


class ModelLoadError(Exception):
    """Raised when the model file exists but could not be loaded correctly."""
    pass


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
# NOTE: This exact function is imported and reused by train_model.py so that
# the preprocessing applied at training time is guaranteed to be identical to
# the preprocessing applied at prediction time.

_WHITESPACE_RE = re.compile(r"\s+")
_HTML_TAG_RE = re.compile(r"<.*?>")
_URL_RE = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")


def preprocess_text(text: str) -> str:
    """
    Clean raw text so it can be fed into the TF-IDF vectorizer.

    Steps:
      1. Lowercase
      2. Strip HTML tags
      3. Remove URLs
      4. Remove punctuation / non-alphabetic characters
      5. Collapse repeated whitespace

    This is a lightweight, dependency-free cleaning pipeline. It does not
    rely on NLTK so that the project runs even if NLTK corpora have not been
    downloaded.
    """
    if text is None:
        return ""

    text = str(text).lower()
    text = _HTML_TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def combine_headline_and_article(headline: str, article: str) -> str:
    """
    Safely combine a headline and article body into a single text blob.
    Either field may be empty; at least one must contain content
    (validated at the Flask layer).
    """
    headline = headline or ""
    article = article or ""
    return f"{headline.strip()} {article.strip()}".strip()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def model_exists() -> bool:
    """Return True if a trained model file is present on disk."""
    return MODEL_PATH.exists()


def load_model():
    """
    Load the trained vectorizer + classifier bundle from disk.

    Returns:
        dict with keys "vectorizer" and "model"

    Raises:
        ModelNotFoundError: if the .pkl file does not exist
        ModelLoadError: if the file exists but cannot be parsed / is invalid
    """
    if not model_exists():
        raise ModelNotFoundError(
            f"Trained model not found at '{MODEL_PATH}'. "
            "Please place your dataset at dataset/news.csv and run "
            "'python train_model.py' to train and save a model."
        )

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001 - we deliberately want a broad catch here
        raise ModelLoadError(
            "The model file exists but could not be loaded. It may be "
            "corrupted or was saved with an incompatible library version. "
            "Try re-running 'python train_model.py'."
        ) from exc

    if not isinstance(bundle, dict) or "vectorizer" not in bundle or "model" not in bundle:
        raise ModelLoadError(
            "The model file does not contain the expected 'vectorizer' and "
            "'model' keys. Please re-run 'python train_model.py'."
        )

    return bundle


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_news(headline: str, article: str, bundle: dict) -> dict:
    """
    Run a prediction on the given headline/article using an already-loaded
    model bundle (as returned by load_model()).

    Returns a dictionary:
        {
            "label": "REAL" | "FAKE",
            "confidence": float | None,   # percentage 0-100, None if unavailable
            "cleaned_text_length": int,
        }
    """
    vectorizer = bundle["vectorizer"]
    classifier = bundle["model"]

    combined = combine_headline_and_article(headline, article)
    cleaned = preprocess_text(combined)

    features = vectorizer.transform([cleaned])
    prediction = classifier.predict(features)[0]

    confidence = None
    if hasattr(classifier, "predict_proba"):
        try:
            proba = classifier.predict_proba(features)[0]
            # Confidence = probability assigned to the predicted class
            class_index = list(classifier.classes_).index(prediction)
            confidence = round(float(proba[class_index]) * 100, 2)
        except Exception:
            # If probability calculation fails for any reason, do not
            # fabricate a confidence value - simply omit it.
            confidence = None

    label = str(prediction).upper()
    if label not in ("REAL", "FAKE"):
        # Defensive normalization in case labels were stored as 0/1 etc.
        label = "FAKE" if label in ("1", "FAKE NEWS") else "REAL"

    return {
        "label": label,
        "confidence": confidence,
        "cleaned_text_length": len(cleaned),
    }
