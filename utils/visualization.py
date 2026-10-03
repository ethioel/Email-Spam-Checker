"""Matplotlib chart helpers.

Only the charts actually rendered by the app live here. Colours default to the
dark theme palette and can be overridden per call so charts stay legible when
the user switches to light mode.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from utils.theme import palette

# Perceptron sign convention: positive is Spam, negative is Ham.
SPAM_COLOR = "#f87171"
HAM_COLOR = "#4ade80"


def _figure(width: float, height: float, p: dict):
    """Create a figure with axes styled for the active theme."""
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_facecolor(p["chart_bg"])
    fig.patch.set_facecolor(p["chart_bg"])
    ax.tick_params(colors=p["muted"])
    ax.xaxis.label.set_color(p["muted"])
    ax.yaxis.label.set_color(p["muted"])
    ax.title.set_color(p["text"])
    for spine in ax.spines.values():
        spine.set_color(p["chart_border"])
    return fig, ax


def _finish(fig, layout: str = "tight"):
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_confidence_gauge(confidence: float, width: float = 8.0):
    """Horizontal bar showing a prediction's confidence percentage."""
    p = palette()
    fig, ax = _figure(width, 1.3, p)
    value = max(0.0, min(float(confidence), 100.0))
    ax.barh(0, value, height=0.45, color=p["accent"])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([])
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Prediction Confidence")
    return _finish(fig)


def plot_feature_contributions(explanation_df, top_n: int = 10):
    """Bar chart of the strongest feature contributions for one email."""
    if explanation_df is None or explanation_df.empty:
        return None

    p = palette()
    top = explanation_df.reindex(
        explanation_df["Contribution"].abs().nlargest(top_n).index
    )

    fig, ax = _figure(max(top_n * 0.45, 4.0), max(len(top) * 0.32, 2.5), p)
    colors = [SPAM_COLOR if c > 0 else HAM_COLOR for c in top["Contribution"]]
    ax.barh(top["Word"], top["Contribution"], color=colors)
    ax.axvline(0, color=p["chart_border"], linewidth=1)
    ax.set_title("Top Influential Words")
    ax.set_xlabel("Contribution (positive → Spam, negative → Ham)")
    ax.invert_yaxis()
    return _finish(fig)


def plot_weight_barh(frame, title: str, color: str):
    """Horizontal bar chart of feature weights for a named class."""
    p = palette()
    fig, ax = _figure(max(len(frame) * 0.32, 4.0), max(len(frame) * 0.28, 2.5), p)
    ax.barh(frame["Word"], frame["Weight"], color=color)
    ax.set_title(title)
    ax.set_xlabel("Weight")
    ax.axvline(0, color=p["chart_border"], linewidth=1)
    ax.invert_yaxis()
    return _finish(fig)


def plot_weight_histogram(weights):
    """Histogram of every learned Perceptron weight."""
    p = palette()
    fig, ax = _figure(10, 4, p)
    ax.hist(weights, bins=50, color=p["accent"], edgecolor="none")
    ax.set_title("Perceptron Weight Distribution")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Number of Features")
    return _finish(fig)


def plot_weight_scatter(weights):
    """Scatter of learned weights against feature index."""
    p = palette()
    fig, ax = _figure(10, 4, p)
    ax.scatter(range(len(weights)), weights, s=8, color=p["accent"], alpha=0.6)
    ax.set_title("Perceptron Learned Weights")
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Weight")
    return _finish(fig)


def plot_length_histogram(lengths):
    """Histogram of email lengths measured in characters."""
    p = palette()
    fig, ax = _figure(8, 4, p)
    ax.hist(lengths, bins=30, color=p["accent"], edgecolor="none")
    ax.set_title("Email Length Distribution")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Number of Emails")
    return _finish(fig)


def plot_pie(counts, title: str):
    """Pie chart of a label count series.

    Spam and Ham slices always get their semantic colours; any other label set
    (feature types, for example) falls back to a theme accent cycle.
    """
    p = palette()
    accents = [p["accent"], "#a78bfa", "#fbbf24", "#f472b6", "#2dd4bf"]
    colors = [
        SPAM_COLOR
        if str(label).lower() == "spam"
        else HAM_COLOR
        if str(label).lower() == "ham"
        else accents[index % len(accents)]
        for index, label in enumerate(counts.index)
    ]
    fig, ax = _figure(5, 5, p)
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"color": p["text"]},
        wedgeprops={"edgecolor": p["chart_bg"]},
    )
    ax.set_title(title)
    ax.axis("equal")
    return _finish(fig)


def plot_metric_bars(metrics_df):
    """Grouped bar chart of the headline evaluation metrics."""
    p = palette()
    labels = metrics_df["Metric"].tolist()
    height = 0.8 / len(labels)
    positions = np.arange(len(labels))

    fig, ax = _figure(8, 4, p)
    for offset, (_, row) in zip(
        np.linspace(-0.4 + height / 2, 0.4 - height / 2, len(labels)),
        metrics_df.iterrows(),
        strict=True,
    ):
        ax.bar(positions + offset, row["Score"], width=height, color=p["accent"])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.9, 1.0)
    ax.set_title("Metric Comparison")
    ax.set_ylabel("Score")
    return _finish(fig)


def plot_confusion_matrix(matrix, labels=("Ham", "Spam")):
    """Annotated confusion matrix heatmap.

    Parameters
    ----------
    matrix
        2x2 array; rows are the actual class, columns the prediction.
    labels
        Class names ordered ``[actual, predicted]``.
    """
    p = palette()
    fig, ax = _figure(5, 5, p)
    ax.imshow(matrix, cmap="Blues")

    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    peak = float(np.max(matrix)) if matrix.size else 1.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row][column]
            ax.text(
                column,
                row,
                f"{value:,.0f}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="#0f172a" if value > peak / 2 else p["text"],
            )

    fig.tight_layout()
    return _finish(fig, layout="tight")