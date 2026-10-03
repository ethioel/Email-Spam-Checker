"""Landing page for the Email Spam Classification System."""

import streamlit as st

from utils import theme
from utils.model_loader import DATASET, load_confusion_matrix, load_metrics
from utils.sidebar import setup_page

# Page configuration, sidebar and stylesheet in one call.
setup_page("Email Spam Classification", "📧")

metrics = load_metrics()

# ==========================================================
# Header
# ==========================================================

theme.page_header(
    f"📧 {theme.APP_TITLE}",
    "Machine Learning-powered email filtering with Perceptron + TF-IDF",
)

theme.hero(
    "Operational overview",
    "Real-time spam detection, batch scoring, and explainable model insights.",
    ["Perceptron", "TF-IDF", f"{DATASET['Vocabulary Size'] // 1000}k features", "1-2 grams"],
)

st.divider()

# ==========================================================
# Performance
# ==========================================================

st.subheader("📈 Model Performance")

theme.metric_row(
    [
        ("Accuracy", f"{metrics['accuracy'] * 100:.2f}%"),
        ("Precision", f"{metrics['precision'] * 100:.2f}%"),
        ("Recall", f"{metrics['recall'] * 100:.2f}%"),
        ("F1 Score", f"{metrics['f1_score'] * 100:.2f}%"),
    ]
)

st.divider()

# ==========================================================
# Features
# ==========================================================

st.subheader("🧰 What you can do")

theme.feature_row(
    [
        ("⚡ Single Scan", "Predict one email in seconds with explainable scoring."),
        ("📦 Bulk Review", "Upload a CSV and classify thousands of emails in one pass."),
        ("🧠 Model Insight", "See the words and signals driving each prediction."),
    ]
)

st.divider()

# ==========================================================
# Pipeline
# ==========================================================

st.subheader("⚙️ How it works")

theme.step_row(["Email", "Clean", "TF-IDF", "Score", "Result"])

st.divider()

# ==========================================================
# Project facts
# ==========================================================

test_rows = int(load_confusion_matrix().sum())

dataset_col, model_col = st.columns(2)

with dataset_col:
    st.subheader("📊 Dataset")
    st.markdown(
        f"""
        - Total emails: **{DATASET['Total Emails']:,}**
        - Ham: **{DATASET['Ham Emails']:,}**
        - Spam: **{DATASET['Spam Emails']:,}**
        """
    )

with model_col:
    st.subheader("🤖 Model")
    st.markdown(
        f"""
        - Algorithm: **{DATASET['Algorithm']}**
        - Features: **{DATASET['Vocabulary Size']:,} TF-IDF**
        - N-grams: **1–2**
        - Test set: **{test_rows:,}** emails
        """
    )

st.divider()

# ==========================================================
# Quick start
# ==========================================================

st.subheader("🚀 Quick start")

quick_cols = st.columns(3)
with quick_cols[0]:
    st.info("Open the single prediction page to test a message instantly.")
with quick_cols[1]:
    st.info("Upload a CSV to score large batches without leaving the dashboard.")
with quick_cols[2]:
    st.info("Review model analytics for the strongest spam and ham signals.")

# ==========================================================
# Footer
# ==========================================================

st.divider()
theme.footer()