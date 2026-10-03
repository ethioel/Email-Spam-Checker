"""Model and evaluation artefact loading, plus prediction helpers.

The trained artefacts live in ``models/`` and the evaluation output in
``results/``. Everything is resolved against the repository root (never the
current working directory) so the app runs correctly from any launch folder.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================================
# Project paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "perceptron_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"

METRICS_PATH = BASE_DIR / "results" / "metrics.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "results" / "confusion_matrix.npy"

# ==========================================================
# Dataset summary
#
# Single source of truth: previously these numbers were copy-pasted across
# app.py, About.py and Performance.py, where they had already drifted apart.
# ==========================================================

DATASET = {
    "Total Emails": 193_812,
    "Ham Emails": 102_159,
    "Spam Emails": 91_653,
    "Training Samples": 155_049,
    "Testing Samples": 38_763,
    "Vocabulary Size": 10_000,
    "N-Grams": "Unigrams + Bigrams",
    "Algorithm": "Perceptron",
    "Vectorizer": "TF-IDF",
}


def get_model_info() -> dict:
    """Return metadata about the trained model."""
    return {
        "Algorithm": DATASET["Algorithm"],
        "Vectorizer": DATASET["Vectorizer"],
        "Features": f"{DATASET['Vocabulary Size']:,}",
        "N-Grams": DATASET["N-Grams"],
        "Classes": {0: "Ham", 1: "Spam"},
    }


# ==========================================================
# Loading
# ==========================================================

@st.cache_resource
def load_model():
    """Load the trained Perceptron model and TF-IDF vectorizer.

    Cached as a resource so the pickles are read from disk once per server
    process rather than on every rerun.

    Returns
    -------
    tuple
        ``(model, vectorizer)``
    """
    for path in (MODEL_PATH, VECTORIZER_PATH):
        if not path.exists():
            st.error(f"❌ Missing model artefact: `{path.name}`.")
            st.info(
                "Run the training notebook first so that `models/` contains "
                "`perceptron_model.pkl` and `tfidf_vectorizer.pkl`."
            )
            st.stop()

    try:
        return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user
        st.error(f"❌ Error loading model: {exc}")
        st.stop()


@st.cache_data
def load_metrics() -> dict:
    """Load evaluation metrics from ``results/metrics.json``."""
    try:
        import json

        with open(METRICS_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        st.error("❌ Missing `results/metrics.json`.")
        st.stop()


@st.cache_data
def load_confusion_matrix() -> np.ndarray:
    """Load the test-set confusion matrix from ``results/confusion_matrix.npy``.

    Rows are the actual class, columns the predicted class, both ordered
    ``[Ham, Spam]``.
    """
    try:
        return np.load(CONFUSION_MATRIX_PATH)
    except FileNotFoundError:
        st.error("❌ Missing `results/confusion_matrix.npy`.")
        st.stop()


def classification_report(matrix: np.ndarray) -> pd.DataFrame:
    """Derive a per-class classification report from the confusion matrix.

    Computing the report from the stored matrix keeps every number on the
    Performance page consistent with ``metrics.json``; the previous hard-coded
    table (all values 0.97/0.98) did not match either artefact.

    Parameters
    ----------
    matrix
        Confusion matrix ordered ``[[ham, spam], [ham, spam]]``.

    Returns
    -------
    pandas.DataFrame
        One row per class plus macro and weighted averages.
    """

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    tn, fp = float(matrix[0][0]), float(matrix[0][1])
    fn, tp = float(matrix[1][0]), float(matrix[1][1])

    ham_support = tn + fp
    spam_support = fn + tp
    total = ham_support + spam_support

    ham_precision = ratio(tn, tn + fn)
    ham_recall = ratio(tn, tn + fp)
    spam_precision = ratio(tp, tp + fp)
    spam_recall = ratio(tp, tp + fn)

    ham_f1 = ratio(2 * ham_precision * ham_recall, ham_precision + ham_recall)
    spam_f1 = ratio(2 * spam_precision * spam_recall, spam_precision + spam_recall)

    def weighted(ham_value: float, spam_value: float) -> float:
        """Weighted average across the two classes, weighted by support."""
        if not total:
            return (ham_value + spam_value) / 2
        return (ham_value * ham_support + spam_value * spam_support) / total

    return pd.DataFrame(
        [
            {
                "Class": "Ham",
                "Precision": ham_precision,
                "Recall": ham_recall,
                "F1-Score": ham_f1,
                "Support": int(ham_support),
            },
            {
                "Class": "Spam",
                "Precision": spam_precision,
                "Recall": spam_recall,
                "F1-Score": spam_f1,
                "Support": int(spam_support),
            },
            {
                "Class": "Macro Avg",
                "Precision": (ham_precision + spam_precision) / 2,
                "Recall": (ham_recall + spam_recall) / 2,
                "F1-Score": (ham_f1 + spam_f1) / 2,
                "Support": int(total),
            },
            {
                "Class": "Weighted Avg",
                "Precision": weighted(ham_precision, spam_precision),
                "Recall": weighted(ham_recall, spam_recall),
                "F1-Score": weighted(ham_f1, spam_f1),
                "Support": int(total),
            },
        ]
    )


# ==========================================================
# Prediction helpers
# ==========================================================

def confidence_from_scores(scores) -> np.ndarray:
    """Map Perceptron decision scores to a 0-100 confidence percentage.

    A Perceptron has no ``predict_proba``, so this is a display-only proxy:
    the magnitude of the decision function saturates at 5.

    Parameters
    ----------
    scores
        Scalar or array of decision scores.

    Returns
    -------
    numpy.ndarray
        Confidence percentages rounded to two decimals.
    """
    values = np.asarray(scores, dtype=float)
    return np.round(np.minimum(np.abs(values) / 5.0, 1.0) * 100, 2)


def calculate_confidence(decision_score: float) -> float:
    """Convert a single decision score into an approximate confidence percentage."""
    return float(confidence_from_scores(decision_score))


def predict_email(email_text: str) -> dict:
    """Classify a single preprocessed email.

    Parameters
    ----------
    email_text
        Preprocessed email text.

    Returns
    -------
    dict
        Keys ``prediction``, ``label``, ``decision_score``, ``confidence``
        and ``vector`` (the sparse TF-IDF row, for explainability).
    """
    model, vectorizer = load_model()

    vector = vectorizer.transform([email_text])
    score = float(model.decision_function(vector)[0])
    prediction = int(model.predict(vector)[0])

    return {
        "prediction": prediction,
        "label": "Spam" if prediction == 1 else "Ham",
        "decision_score": score,
        "confidence": calculate_confidence(score),
        "vector": vector,
    }


def predict_batch(
    texts: list[str],
    model=None,
    vectorizer=None,
    chunk_size: int = 50_000,
    progress=None,
) -> pd.DataFrame:
    """Classify many emails with a single vectorised pass.

    The previous implementation looped row by row, issuing one
    ``vectorizer.transform`` and one ``model.predict`` per email. That is
    quadratic in overhead for large uploads. This version transforms and scores
    whole chunks at once, which is typically two to three orders of magnitude
    faster for a 100k-row file.

    Parameters
    ----------
    texts
        Preprocessed email bodies.
    model, vectorizer
        Optional preloaded artefacts; loaded on demand when omitted.
    chunk_size
        Rows per transform, bounding peak memory for very large inputs.
    progress
        Optional ``callback(done, total)`` invoked after each chunk.

    Returns
    -------
    pandas.DataFrame
        Columns ``Prediction``, ``Decision Score`` and ``Confidence (%)``.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_model()

    total = len(texts)
    scores = np.empty(total, dtype=float)
    predictions = np.empty(total, dtype=np.int64)

    for start in range(0, total, chunk_size):
        chunk = texts[start : start + chunk_size]
        vectors = vectorizer.transform(chunk)
        scores[start : start + len(chunk)] = model.decision_function(vectors)
        predictions[start : start + len(chunk)] = model.predict(vectors)
        if progress is not None:
            progress(min(start + chunk_size, total), total)

    return pd.DataFrame(
        {
            "Prediction": np.where(predictions == 1, "Spam", "Ham"),
            "Decision Score": np.round(scores, 4),
            "Confidence (%)": confidence_from_scores(scores),
        }
    )