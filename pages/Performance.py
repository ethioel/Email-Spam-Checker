"""Evaluation metrics for the trained Perceptron classifier.

Every number on this page is read from ``results/metrics.json`` and
``results/confusion_matrix.npy``. The previous version hard-coded them, and the
values had drifted: the displayed confusion matrix (tn=19987, fp=442) implied
97.38% accuracy, contradicting the 97.68% quoted two sections above it.
"""

import pandas as pd
import streamlit as st

from utils import theme
from utils.model_loader import (
    DATASET,
    classification_report,
    load_confusion_matrix,
    load_metrics,
)
from utils.sidebar import setup_page
from utils.visualization import plot_confusion_matrix, plot_metric_bars

# Page configuration, sidebar and stylesheet in one call.
setup_page("Performance", "📈")

metrics = load_metrics()
matrix = load_confusion_matrix()
report_df = classification_report(matrix)

# ==========================================================
# Header
# ==========================================================

theme.page_header(
    "📈 Model Performance Dashboard",
    "Evaluation metrics for the trained Perceptron classifier.",
)

st.divider()

# ==========================================================
# Headline metrics
# ==========================================================

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

# ==========================================================
# Metric comparison
# ==========================================================

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

# ==========================================================
# Classification report
# ==========================================================

st.header("📋 Classification Report")

st.caption(
    "Derived from the stored test-set confusion matrix. Precision and recall "
    "are reported for the **Spam** class as the positive label."
)

# Formatted as strings rather than via DataFrame.style, whose helper kwargs
# differ between pandas 2.x and 3.x.
display_report = report_df.copy()
for column in ("Precision", "Recall", "F1-Score"):
    display_report[column] = display_report[column].map("{:.4f}".format)
display_report["Support"] = display_report["Support"].map("{:,.0f}".format)

st.dataframe(display_report, hide_index=True)

st.divider()

# ==========================================================
# Confusion matrix
# ==========================================================

st.header("📊 Confusion Matrix")

st.caption(
    "Rows are the actual class, columns the predicted class. "
    f"{int(matrix.sum()):,} test emails."
)

st.pyplot(plot_confusion_matrix(matrix))

tn, fp = int(matrix[0][0]), int(matrix[0][1])
fn, tp = int(matrix[1][0]), int(matrix[1][1])

st.markdown(
    f"""
- ✅ **True Negatives** — {tn:,} Ham emails correctly kept
- ❌ **False Positives** — {fp:,} Ham emails wrongly flagged as Spam
- ❌ **False Negatives** — {fn:,} Spam emails that slipped through
- ✅ **True Positives** — {tp:,} Spam emails correctly caught
"""
)

st.divider()

# ==========================================================
# Evaluation dataset
# ==========================================================

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

# ==========================================================
# Interpretation
# ==========================================================

st.header("🧠 Performance Interpretation")

st.success(
    f"✅ Accuracy of **{metrics['accuracy'] * 100:.2f}%** means the model "
    f"classified {(tn + tp):,} of {int(matrix.sum()):,} test emails correctly."
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

# ==========================================================
# Overall summary
# ==========================================================

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

st.divider()

# ==========================================================
# Footer
# ==========================================================

theme.footer()