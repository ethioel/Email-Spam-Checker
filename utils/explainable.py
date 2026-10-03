"""Explainable AI (XAI) helpers.

Perceptron predictions are explained by attributing the decision score to
individual TF-IDF features::

    contribution = tfidf_value * model_weight

A positive contribution pushes the prediction toward **Spam**, a negative one
toward **Ham**. Because a Perceptron is linear and has no intercept in
``coef_``, summing the contributions reproduces the decision score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPAM, HAM, NEUTRAL = "Spam", "Ham", "Neutral"


def _feature_types(words) -> np.ndarray:
    """Label each feature ``Unigram`` or ``Bigram`` by counting spaces.

    ``get_feature_names_out()`` returns a pandas Index, which on pandas 3.x has
    the new string dtype rather than a NumPy ``U`` dtype. ``np.char.count`` has
    no loop for that dtype, so the values are coerced explicitly.
    """
    values = np.asarray(words, dtype=str)
    return np.where(np.char.count(values, " ") > 0, "Bigram", "Unigram")


def explain_prediction(model, vectorizer, vector) -> pd.DataFrame:
    """Explain one prediction by scoring each active TF-IDF feature.

    Parameters
    ----------
    model
        Trained Perceptron.
    vectorizer
        Fitted TF-IDF vectorizer.
    vector
        Sparse row of TF-IDF values for a single email.

    Returns
    -------
    pandas.DataFrame
        Columns ``Word``, ``Feature Type``, ``TF-IDF``, ``Weight``,
        ``Contribution`` and ``Direction``, sorted by descending absolute
        contribution. Empty when the email matched no known vocabulary.
    """
    tfidf_values = np.asarray(vector.todense()).ravel()
    active = np.flatnonzero(tfidf_values)

    if active.size == 0:
        return pd.DataFrame(
            columns=[
                "Word",
                "Feature Type",
                "TF-IDF",
                "Weight",
                "Contribution",
                "Direction",
            ]
        )

    feature_names = vectorizer.get_feature_names_out()
    weights = model.coef_[0]

    words = feature_names[active]
    tfidf = tfidf_values[active]
    weight = weights[active]
    contribution = tfidf * weight

    # Rank on the exact values, then round for display. Rounding first would
    # merge ties and could reorder features whose contributions differ only
    # beyond the fourth decimal.
    order = np.argsort(-np.abs(contribution))

    return pd.DataFrame(
        {
            "Word": words[order],
            "Feature Type": _feature_types(words[order]),
            "Direction": np.select(
                [contribution[order] > 0, contribution[order] < 0],
                [SPAM, HAM],
                default=NEUTRAL,
            ),
            "Contribution": np.round(contribution[order], 4),
            "TF-IDF": np.round(tfidf[order], 4),
            "Weight": np.round(weight[order], 4),
        }
    )


def top_spam_words(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the strongest features pushing a prediction toward Spam."""
    if df.empty:
        return df
    spam_df = df[df["Contribution"] > 0].sort_values(
        "Contribution", ascending=False
    )
    return spam_df.head(top_n)


def top_ham_words(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the strongest features pushing a prediction toward Ham."""
    if df.empty:
        return df
    ham_df = df[df["Contribution"] < 0].sort_values(
        "Contribution", ascending=True
    )
    return ham_df.head(top_n)


def prediction_summary(df: pd.DataFrame) -> dict:
    """Summarise the feature contributions of a single prediction."""
    if df.empty:
        return {
            "Total Features": 0,
            "Spam Features": 0,
            "Ham Features": 0,
            "Unigrams": 0,
            "Bigrams": 0,
        }

    return {
        "Total Features": int(len(df)),
        "Spam Features": int((df["Direction"] == SPAM).sum()),
        "Ham Features": int((df["Direction"] == HAM).sum()),
        "Unigrams": int((df["Feature Type"] == "Unigram").sum()),
        "Bigrams": int((df["Feature Type"] == "Bigram").sum()),
    }


def vocabulary_frame(model, vectorizer) -> pd.DataFrame:
    """Return every learned feature with its weight and n-gram type.

    Built once and cached by the caller: it is a 10k-row frame recomputed on
    every rerun of the Model Analytics page in the previous implementation.

    Parameters
    ----------
    model
        Trained Perceptron.
    vectorizer
        Fitted TF-IDF vectorizer.

    Returns
    -------
    pandas.DataFrame
        Columns ``Word``, ``Feature Type`` and ``Weight``.
    """
    words = vectorizer.get_feature_names_out()
    return pd.DataFrame(
        {
            "Word": words,
            "Feature Type": _feature_types(words),
            "Weight": model.coef_[0],
        }
    )


def global_feature_importance(model, vectorizer, top_n: int = 20):
    """Return the globally strongest Spam and Ham features.

    Parameters
    ----------
    model
        Trained Perceptron.
    vectorizer
        Fitted TF-IDF vectorizer.
    top_n
        Rows per side.

    Returns
    -------
    tuple
        ``(spam_df, ham_df)``
    """
    importance = vocabulary_frame(model, vectorizer)

    spam = importance.nlargest(top_n, "Weight").reset_index(drop=True)
    ham = importance.nsmallest(top_n, "Weight").reset_index(drop=True)
    return spam, ham


def explain_decision(score: float) -> dict:
    """Turn a Perceptron decision score into a human-readable explanation.

    Parameters
    ----------
    score
        Raw decision function value.

    Returns
    -------
    dict
        Keys ``Prediction``, ``Confidence``, ``Score`` and ``Interpretation``.
    """
    magnitude = abs(score)

    if magnitude >= 5:
        confidence = "Very High"
        interpretation = "The model is extremely confident in this prediction."
    elif magnitude >= 3:
        confidence = "High"
        interpretation = "The prediction is made with high confidence."
    elif magnitude >= 1:
        confidence = "Moderate"
        interpretation = "The prediction is reasonably confident."
    else:
        confidence = "Low"
        interpretation = (
            "The email is close to the decision boundary and may be "
            "difficult to classify."
        )

    return {
        "Prediction": SPAM if score > 0 else HAM,
        "Confidence": confidence,
        "Score": round(score, 4),
        "Interpretation": interpretation,
    }