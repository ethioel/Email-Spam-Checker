"""Single email classification with explainable AI output."""

from datetime import datetime

import pandas as pd
import streamlit as st

from utils import theme
from utils.explainable import (
    explain_decision,
    explain_prediction,
    prediction_summary,
    top_ham_words,
    top_spam_words,
)
from utils.model_loader import load_model, predict_email
from utils.preprocessing import preprocess_text
from utils.sidebar import setup_page
from utils.visualization import plot_confidence_gauge, plot_feature_contributions

# Page configuration, sidebar and stylesheet in one call.
setup_page("Single Prediction", "📧")

model, vectorizer = load_model()
st.session_state.setdefault("history", [])

# ==========================================================
# Header
# ==========================================================

theme.page_header(
    "📧 Single Email Prediction",
    "Predict whether an email is Spam or Ham with the trained Perceptron model.",
)

st.divider()

# ==========================================================
# Input
# ==========================================================

left, right = st.columns([2.3, 1])

with left:
    typed_email = st.text_area(
        "Paste Email",
        height=320,
        placeholder=(
            "Example\n\nCongratulations!\n\nYou have won a $1000 Gift Card.\n\n"
            "Click here to claim your prize.\n\nOR\n\nPaste any email here..."
        ),
    )

with right:
    st.subheader("Upload Email")
    uploaded_file = st.file_uploader("Upload TXT File", type=["txt"])

    st.markdown("---")

    theme.metric_row(
        [
            ("Vocabulary", "10,000"),
            ("Algorithm", "Perceptron"),
            ("Features", "TF-IDF"),
            ("N-grams", "1-2"),
        ]
    )

# An uploaded file wins over whatever is currently typed.
email = (
    uploaded_file.read().decode("utf-8", errors="replace")
    if uploaded_file is not None
    else typed_email
)

st.divider()

# ==========================================================
# Predict
# ==========================================================

if st.button("🚀 Predict Email", width="stretch"):
    if not email.strip():
        st.warning("⚠️ Please enter or upload an email.")
        st.stop()

    with st.spinner("Analyzing email..."):
        clean_email = preprocess_text(email)
        result = predict_email(clean_email)

        if not clean_email:
            st.warning(
                "⚠️ Nothing usable was left after cleaning (no words matched "
                "the model's vocabulary). Try an email with more text."
            )
            st.stop()

        st.session_state.history.append(
            {
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Prediction": result["label"],
                "Confidence (%)": result["confidence"],
                "Decision Score": round(result["decision_score"], 4),
            }
        )

    st.success("✅ Analysis Completed")
    st.divider()

    # ----------------------------------------------------------
    # Result
    # ----------------------------------------------------------

    label = result["label"]
    decision_score = result["decision_score"]
    confidence = result["confidence"]

    result_col, confidence_col = st.columns(2)

    with result_col:
        st.subheader("📌 Prediction")
        if label == "Spam":
            st.error("🚨 SPAM EMAIL")
        else:
            st.success("✅ HAM EMAIL")
        st.metric("Decision Score", f"{decision_score:.4f}")

    with confidence_col:
        st.subheader("🎯 Confidence")
        st.metric("Estimated Confidence", f"{confidence:.1f}%")
        st.pyplot(plot_confidence_gauge(confidence))

    st.divider()

    # ----------------------------------------------------------
    # Decision interpretation
    # ----------------------------------------------------------

    st.subheader("🧠 Decision Interpretation")

    explanation = explain_decision(decision_score)

    theme.metric_row(
        [
            ("Predicted Class", explanation["Prediction"]),
            ("Confidence Level", explanation["Confidence"]),
            ("Decision Score", explanation["Score"]),
        ]
    )

    st.info(explanation["Interpretation"])
    st.caption(
        "Positive score → Spam · Negative score → Ham · "
        "Larger magnitude means higher confidence · "
        "Scores near zero indicate uncertainty."
    )

    with st.expander("🧹 View Preprocessed Email"):
        st.write(clean_email)

    st.divider()

    # ----------------------------------------------------------
    # Explainable AI
    # ----------------------------------------------------------

    st.header("🧠 Explainable AI (XAI)")

    explanation_df = explain_prediction(model, vectorizer, result["vector"])

    if explanation_df.empty:
        st.warning("No influential words were found.")
    else:
        summary = prediction_summary(explanation_df)

        st.success(
            f"The prediction was influenced by **{summary['Total Features']}** "
            f"features ({summary['Spam Features']} toward Spam, "
            f"{summary['Ham Features']} toward Ham)."
        )

        theme.metric_row(
            [
                ("Total Features", summary["Total Features"]),
                ("Spam Features", summary["Spam Features"]),
                ("Ham Features", summary["Ham Features"]),
                ("Unigrams", summary["Unigrams"]),
                ("Bigrams", summary["Bigrams"]),
            ]
        )

        st.divider()

        st.subheader("📊 Top Feature Contributions")
        st.pyplot(plot_feature_contributions(explanation_df, top_n=10))
        st.caption(
            "Positive contributions push the prediction toward Spam; "
            "negative contributions push it toward Ham."
        )

        st.divider()

        spam_col, ham_col = st.columns(2)

        with spam_col:
            st.subheader("🚨 Top Spam Features")
            st.dataframe(top_spam_words(explanation_df), hide_index=True)

        with ham_col:
            st.subheader("✅ Top Ham Features")
            st.dataframe(top_ham_words(explanation_df), hide_index=True)

        st.divider()

        with st.expander("📋 Complete Feature Contributions"):
            st.dataframe(explanation_df, hide_index=True)

        st.download_button(
            "📥 Download Explanation (CSV)",
            data=explanation_df.to_csv(index=False),
            file_name="feature_contributions.csv",
            mime="text/csv",
        )

    st.divider()

# ==========================================================
# Prediction history
# ==========================================================

st.header("📜 Prediction History")

if not st.session_state.history:
    st.info("No predictions have been made in this session.")
else:
    history_df = pd.DataFrame(st.session_state.history)

    theme.metric_row(
        [
            ("Total Predictions", len(history_df)),
            ("Spam", int((history_df["Prediction"] == "Spam").sum())),
            ("Ham", int((history_df["Prediction"] == "Ham").sum())),
            (
                "Average Confidence",
                f"{round(history_df['Confidence (%)'].mean(), 2)}%",
            ),
        ]
    )

    st.divider()

    counts_col, trend_col = st.columns(2)

    with counts_col:
        st.subheader("📊 Prediction Distribution")
        st.bar_chart(history_df["Prediction"].value_counts())

    with trend_col:
        st.subheader("📈 Confidence Trend")
        st.line_chart(
            history_df.reset_index(drop=True)["Confidence (%)"]
        )

    st.divider()

    st.subheader("📋 Prediction Records")
    st.dataframe(history_df, hide_index=True)

    download_col, clear_col = st.columns(2)

    with download_col:
        st.download_button(
            "📥 Download Prediction History",
            data=history_df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
            width="stretch",
        )

    with clear_col:
        if st.button(
            "🗑️ Clear Prediction History",
            width="stretch",
            type="secondary",
        ):
            st.session_state.history = []
            st.rerun()

# ==========================================================
# About the model
# ==========================================================

st.divider()
st.header("ℹ️ About This Model")

info_col, xai_col = st.columns(2)

with info_col:
    st.info(
        """
### Machine Learning Model

- Algorithm: **Perceptron**
- Text Representation: **TF-IDF**
- Vocabulary Size: **10,000**
- Features: **Unigrams + Bigrams**
- Task: **Binary Email Classification**
"""
    )

with xai_col:
    st.info(
        """
### Explainable AI

This application explains predictions by displaying:

- Influential words
- Feature contributions
- Decision score
- Prediction confidence

This improves transparency and helps users understand why the model
classified an email as Spam or Ham.
"""
    )

# ==========================================================
# Footer
# ==========================================================

st.divider()
theme.footer()