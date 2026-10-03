"""Text preprocessing used by the training pipeline and the app.

The cleaning order below is the contract between the trained vectorizer and the
live app: changing it will silently invalidate the model. Note in particular
that stopword removal and lemmatisation are both active here, so the NLTK
corpora are a hard requirement rather than an optional nicety.
"""

from __future__ import annotations

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================================
# NLTK resources
# ==========================================================

def _download(resource: str) -> None:
    """Best-effort download of a single NLTK corpus."""
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


def _load_resources() -> tuple[frozenset, WordNetLemmatizer]:
    """Return the stopword set and lemmatizer, downloading corpora if needed.

    Availability is confirmed by actually exercising each resource rather than
    by ``nltk.data.find``. On NLTK 3.10 the wordnet corpus resolves only as
    ``wordnet.zip``, so a plain ``find("corpora/wordnet")`` reports a false
    negative even though the lemmatizer works fine.

    Missing corpora would not raise on their own: the lemmatizer would quietly
    degrade and change the token stream, so a genuinely unusable corpus raises
    a clear error instead of silently skewing predictions.
    """
    for resource in ("stopwords", "wordnet"):
        _download(resource)

    try:
        words = frozenset(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")
    except LookupError as exc:
        raise RuntimeError(
            "The NLTK corpora required by this project are unavailable "
            f"({exc}). Run `python -m nltk.downloader stopwords wordnet` once "
            "with network access, then restart the app."
        ) from exc

    return words, lemmatizer


STOP_WORDS, LEMMATIZER = _load_resources()

# ==========================================================
# Precompiled cleaning rules
#
# Compiled once at import time. The previous implementation recompiled every
# regex and rebuilt the punctuation table on each call, which is measurable
# overhead when preprocessing tens of thousands of emails.
# ==========================================================

_HTML_RE = re.compile(r"<.*?>")
_URL_RE = re.compile(r"http\S+|www\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_NUMBER_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def preprocess_text(text: str) -> str:
    """Clean and normalise a single email.

    Steps: lowercase, strip HTML, URLs, email addresses, numbers and
    punctuation, collapse whitespace, drop stopwords, then lemmatise.

    Parameters
    ----------
    text
        Raw email body.

    Returns
    -------
    str
        Whitespace-separated cleaned tokens, ready for the vectorizer.
    """
    if not isinstance(text, str):
        return ""

    text = _HTML_RE.sub(" ", text.lower())
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _NUMBER_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    return " ".join(
        LEMMATIZER.lemmatize(word) for word in text.split() if word not in STOP_WORDS
    )


def preprocess_series(series: pd.Series) -> pd.Series:
    """Apply :func:`preprocess_text` to every value of a Series.

    Missing values become empty strings rather than ``NaN`` so the result can be
    handed straight to the vectorizer.

    Parameters
    ----------
    series
        Series of raw email bodies.

    Returns
    -------
    pandas.Series
        Series of cleaned emails, aligned with the input index.
    """
    return series.astype("object").where(series.notna(), "").map(preprocess_text)