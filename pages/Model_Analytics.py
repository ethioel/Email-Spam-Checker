"""Explore the vocabulary and weights learned by the Perceptron."""

import numpy as np
import pandas as pd
import streamlit as st

from utils import theme
from utils.explainable import global_feature_importance, vocabulary_frame
from utils.model_loader import load_model
from utils.sidebar import setup_page
from utils.visualization import (
    HAM_COLOR,
    SPAM_COLOR,
    plot_weight_barh,
    plot_weight_histogram,
    plot_pie,
    plot_weight_scatter,
)

# Page configuration, sidebar and stylesheet in one call.
setup_page("Model Analytics", "📊")

model, _vectorizer = load_model()
weights = model.coef_[0]


@st.cache_resource(show_spinner=False)
def _analytics_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the vocabulary and weight tables once and reuse them.

    Takes no arguments on purpose: ``st.cache_resource`` still hashes its
    parameters, and a fitted estimator is not hashable. ``load_model`` is itself
    cached, so the artefacts are only unpickled once.
    """
    fitted_model, fitted_vectorizer = load_model()
    vocabulary = vocabulary_frame(fitted_model, fitted_vectorizer)
    spam, ham = global_feature_importance(fitted_model, fitted_vectorizer, top_n=20)
    return vocabulary, spam, ham


vocab_df, spam_df, ham_df = _analytics_tables()

# ==========================================================
# Header
# ==========================================================

theme.page_header(
    "📊 Model Analytics Dashboard",
    "Understand what the machine learning model has learned from the training data.",
)

st.divider()

# ==========================================================
# Model overview
# ==========================================================

st.header("🧠 Model Overview")

total_features = len(vocab_df)
unigrams = int((vocab_df["Feature Type"] == "Unigram").sum())
bigrams = int((vocab_df["Feature Type"] == "Bigram").sum())

theme.metric_row(
    [
        ("Algorithm", "Perceptron"),
        ("Vocabulary", f"{total_features:,}"),
        ("Unigrams", f"{unigrams:,}"),
        ("Bigrams", f"{bigrams:,}"),
    ]
)

st.divider()

# ==========================================================
# Weight statistics
# ==========================================================

st.header("📈 Model Weight Statistics")

theme.metric_row(
    [
        ("Maximum", f"{weights.max():.4f}"),
        ("Minimum", f"{weights.min():.4f}"),
        ("Mean", f"{weights.mean():.4f}"),
        ("Median", f"{np.median(weights):.4f}"),
        ("Std Dev", f"{weights.std():.4f}"),
    ]
)

st.divider()

st.subheader("📊 Distribution of Model Weights")
st.pyplot(plot_weight_histogram(weights))

st.divider()

# ==========================================================
# Strongest features
# ==========================================================

st.header("🔥 Strongest Learned Features")

spam_col, ham_col = st.columns(2)

with spam_col:
    st.subheader("🚨 Strongest Spam Features")
    st.pyplot(plot_weight_barh(spam_df, "Top Spam Features", SPAM_COLOR))

with ham_col:
    st.subheader("✅ Strongest Ham Features")
    st.pyplot(plot_weight_barh(ham_df, "Top Ham Features", HAM_COLOR))

st.divider()

st.header("📋 Full Weight Tables")

table_col_1, table_col_2 = st.columns(2)

with table_col_1:
    st.subheader("Spam Weights")
    st.dataframe(spam_df, hide_index=True)

with table_col_2:
    st.subheader("Ham Weights")
    st.dataframe(ham_df, hide_index=True)

st.divider()

# ==========================================================
# Vocabulary analysis
# ==========================================================

st.header("📚 Vocabulary Analysis")

type_col, summary_col = st.columns(2)

with type_col:
    st.subheader("Feature Types")
    st.pyplot(plot_pie(vocab_df["Feature Type"].value_counts(), "Feature Types"))

with summary_col:
    st.subheader("Vocabulary Summary")
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": ["Total Features", "Unigrams", "Bigrams"],
                "Value": [f"{total_features:,}", f"{unigrams:,}", f"{bigrams:,}"],
            }
        ),
        hide_index=True,
    )

st.divider()

# ==========================================================
# Vocabulary search
# ==========================================================

st.header("🔍 Search Learned Vocabulary")

keyword = st.text_input("Search Feature", placeholder="Type a word...")

filtered_vocab = vocab_df

if keyword.strip():
    # Literal matching: users typing "c++" or "a.b" should not trigger a regex
    # error or a surprising pattern match.
    filtered_vocab = vocab_df[
        vocab_df["Word"].str.contains(keyword, case=False, na=False, regex=False)
    ]

st.caption(f"{len(filtered_vocab):,} of {total_features:,} features shown.")
st.dataframe(filtered_vocab, hide_index=True, height=400)

st.divider()

# ==========================================================
# Scatter plot
# ==========================================================

st.header("📈 Feature Weight Scatter Plot")
st.pyplot(plot_weight_scatter(weights))

st.divider()

# ==========================================================
# Exports
# ==========================================================

st.header("📥 Export")

st.download_button(
    "📥 Download Model Vocabulary",
    data=vocab_df.to_csv(index=False),
    file_name="model_vocabulary.csv",
    mime="text/csv",
    width="stretch",
)

st.divider()

# ==========================================================
# Summary
# ==========================================================

st.header("🧠 Model Summary")

st.dataframe(
    pd.DataFrame(
        {
            "Property": [
                "Classifier",
                "Vectorizer",
                "Vocabulary Size",
                "Classes",
                "Feature Types",
                "Maximum Weight",
                "Minimum Weight",
                "Average Weight",
            ],
            # All values are strings: mixing text and floats in one column
            # yields an object-dtype column that pyarrow cannot serialise.
            "Value": [
                "Perceptron",
                "TF-IDF",
                f"{total_features:,}",
                "Spam / Ham",
                "Unigram + Bigram",
                f"{weights.max():.4f}",
                f"{weights.min():.4f}",
                f"{weights.mean():.4f}",
            ],
        }
    ),
    hide_index=True,
)

st.divider()

# ==========================================================
# Footer
# ==========================================================

theme.footer()