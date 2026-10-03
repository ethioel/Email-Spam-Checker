"""Evaluation metrics and learned-feature exploration in one dashboard."""

import numpy as np
import pandas as pd
import streamlit as st

from utils import theme
from utils.explainable import global_feature_importance, vocabulary_frame
from utils.model_loader import (
    DATASET,
    classification_report,
    load_confusion_matrix,
    load_metrics,
    load_model,
)
from utils.sidebar import setup_page
from utils.visualization import (
    HAM_COLOR,
    SPAM_COLOR,
    plot_confusion_matrix,
    plot_metric_bars,
    plot_weight_barh,
    plot_weight_histogram,
    plot_pie,
    plot_weight_scatter,
)

setup_page("Model Analytics", "📊")

metrics = load_metrics()
matrix = load_confusion_matrix()
report_df = classification_report(matrix)

model, _vectorizer = load_model()
weights = model.coef_[0]


@st.cache_resource(show_spinner=False)
def _analytics_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Vocabulary and weight tables, built once per server process.

    Takes no arguments on purpose: ``st.cache_resource`` still hashes its
    parameters, and a fitted estimator is not hashable. ``load_model`` is itself
    cached, so the artefacts are only unpickled once.
    """
    fitted_model, fitted_vectorizer = load_model()
    vocabulary = vocabulary_frame(fitted_model, fitted_vectorizer)
    spam, ham = global_feature_importance(fitted_model, fitted_vectorizer, top_n=20)
    return vocabulary, spam, ham


vocab_df, spam_df, ham_df = _analytics_tables()

total_features = len(vocab_df)
unigrams = int((vocab_df["Feature Type"] == "Unigram").sum())
bigrams = int((vocab_df["Feature Type"] == "Bigram").sum())

tn, fp = int(matrix[0][0]), int(matrix[0][1])
fn, tp = int(matrix[1][0]), int(matrix[1][1])
test_rows = int(matrix.sum())

theme.page_header(
    "📊 Model Analytics & Performance",
    "How well the Perceptron scores, and what it actually learned.",
)

st.divider()

theme.metric_row(
    [
        ("Accuracy", f"{metrics['accuracy'] * 100:.2f}%"),
        ("Precision", f"{metrics['precision'] * 100:.2f}%"),
        ("Recall", f"{metrics['recall'] * 100:.2f}%"),
        ("F1 Score", f"{metrics['f1_score'] * 100:.2f}%"),
        ("Features", f"{total_features:,}"),
    ]
)

performance_tab, analytics_tab = st.tabs(["📈 Performance", "🧠 Model Analytics"])

with performance_tab:
    st.divider()

    st.header("🏆 Performance Metrics")

    theme.metric_row(
        [
            ("Accuracy", f"{metrics['accuracy'] * 100:.2f}%"),
            ("Precision", f"{metrics['precision'] * 100:.2f}%"),
            ("Recall", f"{metrics['recall'] * 100:.2f}%"),
            ("F1 Score", f"{metrics['f1_score'] * 100:.2f}%"),
        ]
    )

    st.divider()

    st.header("📊 Metric Comparison")

    comparison_df = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1"],
            "Score": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
            ],
        }
    )

    st.pyplot(plot_metric_bars(comparison_df))

    st.divider()

    st.header("📋 Classification Report")

    st.caption(
        "Derived from the stored test-set confusion matrix. Precision and recall "
        "are reported for the **Spam** class as the positive label."
    )

    display_report = report_df.copy()
    for column in ("Precision", "Recall", "F1-Score"):
        display_report[column] = display_report[column].map("{:.4f}".format)
    display_report["Support"] = display_report["Support"].map("{:,.0f}".format)

    st.dataframe(display_report, hide_index=True)

    st.divider()

    st.header("📊 Confusion Matrix")

    st.caption(
        "Rows are the actual class, columns the predicted class. "
        f"{test_rows:,} test emails."
    )

    st.pyplot(plot_confusion_matrix(matrix))

    st.markdown(
        f"""
- ✅ **True Negatives** — {tn:,} Ham emails correctly kept
- ❌ **False Positives** — {fp:,} Ham emails wrongly flagged as Spam
- ❌ **False Negatives** — {fn:,} Spam emails that slipped through
- ✅ **True Positives** — {tp:,} Spam emails correctly caught
"""
    )

    st.divider()

    st.header("📚 Evaluation Dataset")

    theme.metric_row(
        [
            ("Training Samples", f"{DATASET['Training Samples']:,}"),
            ("Testing Samples", f"{DATASET['Testing Samples']:,}"),
            ("Vocabulary", f"{DATASET['Vocabulary Size']:,}"),
            ("Total Emails", f"{DATASET['Total Emails']:,}"),
        ]
    )

    st.divider()

    st.header("🧠 Performance Interpretation")

    st.success(
        f"✅ Accuracy of **{metrics['accuracy'] * 100:.2f}%** means the model "
        f"classified {(tn + tp):,} of {test_rows:,} test emails correctly."
    )
    st.info(
        f"🎯 Precision of **{metrics['precision'] * 100:.2f}%** means that when the "
        f"model predicts **Spam** it is correct {metrics['precision'] * 100:.1f}% of "
        f"the time — only {fp:,} Ham emails are wrongly flagged."
    )
    st.info(
        f"📬 Recall of **{metrics['recall'] * 100:.2f}%** means {tp:,} of the "
        f"{tp + fn:,} Spam emails are caught, leaving {fn:,} false negatives."
    )
    st.success(
        f"🏆 F1 Score of **{metrics['f1_score'] * 100:.2f}%** shows a strong balance "
        f"between precision and recall."
    )
    st.warning(
        "⚠️ Despite the strong scores, emails sitting close to the decision "
        "boundary can still be misclassified. Precision and recall are measured on "
        "a held-out test split, so real-world figures will vary with distribution "
        "shift."
    )

    st.divider()

    st.header("📈 Overall Summary")

    st.dataframe(
        pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "Training Samples",
                    "Testing Samples",
                    "Vocabulary Size",
                    "True Negatives",
                    "False Positives",
                    "False Negatives",
                    "True Positives",
                ],
                "Value": [
                    f"{metrics['accuracy'] * 100:.2f}%",
                    f"{metrics['precision'] * 100:.2f}%",
                    f"{metrics['recall'] * 100:.2f}%",
                    f"{metrics['f1_score'] * 100:.2f}%",
                    f"{DATASET['Training Samples']:,}",
                    f"{DATASET['Testing Samples']:,}",
                    f"{DATASET['Vocabulary Size']:,}",
                    f"{tn:,}",
                    f"{fp:,}",
                    f"{fn:,}",
                    f"{tp:,}",
                ],
            }
        ),
        hide_index=True,
    )

with analytics_tab:
    st.divider()

    st.header("🧠 Model Overview")

    theme.metric_row(
        [
            ("Algorithm", "Perceptron"),
            ("Vocabulary", f"{total_features:,}"),
            ("Unigrams", f"{unigrams:,}"),
            ("Bigrams", f"{bigrams:,}"),
        ]
    )

    st.divider()

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

    st.header("🔍 Search Learned Vocabulary")

    keyword = st.text_input("Search Feature", placeholder="Type a word...")

    filtered_vocab = vocab_df

    if keyword.strip():
        filtered_vocab = vocab_df[
            vocab_df["Word"].str.contains(keyword, case=False, na=False, regex=False)
        ]

    st.caption(f"{len(filtered_vocab):,} of {total_features:,} features shown.")
    st.dataframe(filtered_vocab, hide_index=True, height=400)

    st.divider()

    st.header("📈 Feature Weight Scatter Plot")
    st.pyplot(plot_weight_scatter(weights))

    st.divider()

    st.header("📥 Export")

    st.download_button(
        "📥 Download Model Vocabulary",
        data=vocab_df.to_csv(index=False),
        file_name="model_vocabulary.csv",
        mime="text/csv",
        width="stretch",
    )

    st.divider()

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
                # Every value is a string: mixing text and floats in one column
                # yields an object dtype that pyarrow cannot serialise.
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
theme.footer()