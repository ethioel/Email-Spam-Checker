"""About the project, its pipeline and the dataset."""

from pathlib import Path

import pandas as pd
import streamlit as st

from utils import theme
from utils.model_loader import DATASET, get_model_info, load_metrics
from utils.sidebar import setup_page

setup_page("About", "ℹ️")

metrics = load_metrics()
model_info = get_model_info()

theme.page_header(
    "ℹ️ About This Project",
    "A complete Machine Learning application for Email Spam Detection using "
    "the Perceptron algorithm.",
)

st.divider()

st.header("📌 Project Overview")

st.write(
    """
This application classifies email messages into **Spam** or **Ham** using
Natural Language Processing (NLP) and Machine Learning.

The project demonstrates an end-to-end machine learning workflow, including
data preprocessing, feature engineering using TF-IDF, model training with the
Perceptron algorithm, evaluation, Explainable AI, and deployment through
Streamlit.

The goal is to provide accurate and interpretable spam detection while
showcasing the complete machine learning pipeline.
"""
)

st.divider()

st.header("🚀 Application Features")

st.dataframe(
    pd.DataFrame(
        {
            "Feature": [
                "📧 Single Email Prediction",
                "📂 Batch Prediction",
                "🧠 Explainable AI",
                "📊 Model Analytics & Performance",
                "📥 CSV Export",
                "🔍 Vocabulary Search",
                "🌓 Dark / Light Mode",
            ],
            "Description": [
                "Predict Spam or Ham for one email",
                "Predict hundreds or thousands of emails",
                "Explain why the model made a prediction",
                "Evaluation metrics and learned feature weights",
                "Download prediction results",
                "Search learned vocabulary",
                "Switch between light and dark themes",
            ],
        }
    ),
    hide_index=True,
)

st.divider()

st.header("🛠️ Technologies Used")

st.dataframe(
    pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Streamlit",
                "Scikit-learn",
                "Pandas",
                "NumPy",
                "Matplotlib",
                "NLTK",
                "Joblib",
            ],
            "Purpose": [
                "Programming Language",
                "Web Application",
                "Machine Learning",
                "Data Analysis",
                "Numerical Computing",
                "Visualization",
                "Text Preprocessing",
                "Model Serialization",
            ],
        }
    ),
    hide_index=True,
)

st.divider()

st.header("🧠 Machine Learning Pipeline")

st.dataframe(
    pd.DataFrame(
        {
            "Step": [str(n) for n in range(1, 9)],
            "Process": [
                "Load Dataset",
                "Exploratory Data Analysis (EDA)",
                "Text Preprocessing",
                "TF-IDF Vectorization",
                "Train-Test Split",
                "Train Perceptron Model",
                "Model Evaluation",
                "Deploy with Streamlit",
            ],
            "Description": [
                "Load the Spam/Ham email dataset.",
                "Explore class distribution and text statistics.",
                "Clean text by removing punctuation, stopwords, numbers and "
                "URLs, then lemmatize.",
                "Convert cleaned emails into numerical TF-IDF features using "
                "unigrams and bigrams.",
                "Split the dataset into 80% training and 20% testing.",
                "Train the Perceptron classifier on the TF-IDF vectors.",
                "Evaluate the model using Accuracy, Precision, Recall, F1 Score "
                "and the Confusion Matrix.",
                "Build an interactive web application for prediction and analysis.",
            ],
        }
    ),
    hide_index=True,
)

st.caption(
    "Steps 1–7 are reproduced end to end in the training notebook; step 8 is this app."
)
st.link_button("📓 Training notebook on Kaggle", theme.NOTEBOOK_URL)

st.divider()

st.header("📚 Dataset Information")

theme.metric_row(
    [
        ("Total Emails", f"{DATASET['Total Emails']:,}"),
        ("Ham Emails", f"{DATASET['Ham Emails']:,}"),
        ("Spam Emails", f"{DATASET['Spam Emails']:,}"),
        ("Vocabulary", f"{DATASET['Vocabulary Size']:,}"),
    ]
)

st.info(
    f"""
**Dataset Summary**

• Total Emails: **{DATASET['Total Emails']:,}**
• Classes: **Ham (0)** and **Spam (1)**
• Feature Extraction: **{DATASET['Vectorizer']}**
• N-Grams: **(1, 2)**
• Train/Test Split: **80% / 20%**
"""
)

st.divider()

st.header("🤖 Model Information")

st.dataframe(
    pd.DataFrame(
        {
            "Property": [
                "Algorithm",
                "Learning Method",
                "Vectorizer",
                "Features",
                "N-Grams",
                "Classification",
                "Output",
                "Deployment",
            ],
            "Value": [
                model_info["Algorithm"],
                "Supervised Learning",
                model_info["Vectorizer"],
                model_info["Features"],
                model_info["N-Grams"],
                "Binary",
                "Spam / Ham",
                "Streamlit",
            ],
        }
    ),
    hide_index=True,
)

st.divider()

st.header("📂 Project Structure")

root = Path(__file__).resolve().parent.parent
skip = {".git", ".kilo", "__pycache__", "images"}


def _visible(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.name not in skip and not path.name.startswith(".")
        ),
        key=lambda path: (path.is_file(), path.name.lower()),
    )


def _branch(index: int, total: int) -> str:
    return "└──" if index == total - 1 else "├──"


entries = _visible(root)
lines = [f"{root.name}/"]
for index, entry in enumerate(entries):
    suffix = "/" if entry.is_dir() else ""
    lines.append(f"{_branch(index, len(entries))} {entry.name}{suffix}")
    if not entry.is_dir():
        continue
    children = _visible(entry)
    for position, child in enumerate(children):
        suffix = "/" if child.is_dir() else ""
        lines.append(f"│   {_branch(position, len(children))} {child.name}{suffix}")

st.code("\n".join(lines), language="text")

st.divider()

st.header("🏆 Model Performance")

st.dataframe(
    pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [
                f"{metrics['accuracy'] * 100:.2f}%",
                f"{metrics['precision'] * 100:.2f}%",
                f"{metrics['recall'] * 100:.2f}%",
                f"{metrics['f1_score'] * 100:.2f}%",
            ],
        }
    ),
    hide_index=True,
)

st.success(
    f"The Perceptron classifier achieved an F1 Score of "
    f"**{metrics['f1_score'] * 100:.2f}%** on the held-out test set, "
    f"correctly classifying the large majority of both Ham and Spam emails."
)

st.divider()

st.header("🎯 Future Improvements")

st.dataframe(
    pd.DataFrame(
        {
            "Future Enhancement": [
                "Compare multiple Machine Learning algorithms",
                "Deep Learning using LSTM or Transformers",
                "Multilingual Spam Detection",
                "Real-time Email Classification API",
                "Continuous Model Retraining",
                "Cloud Deployment",
            ],
            "Description": [
                "Evaluate Logistic Regression, SVM, Naive Bayes and Random Forest.",
                "Investigate deep learning models for improved text understanding.",
                "Support spam detection in multiple languages.",
                "Develop a REST API for real-time predictions.",
                "Automatically update the model using newly labeled data.",
                "Deploy using Streamlit Community Cloud or Docker.",
            ],
        }
    ),
    hide_index=True,
)

st.divider()

theme.footer()