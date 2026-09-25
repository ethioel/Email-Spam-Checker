# ==========================================================
# Single Email Prediction
# Email Spam Classification System
# ==========================================================

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.preprocessing import preprocess_text

from utils.model_loader import (
    load_model,
    predict_email,
    calculate_confidence
)

from utils.explainable import (
    explain_prediction,
    top_spam_words,
    top_ham_words,
    explain_decision
)

from utils.visualization import (
    plot_confidence_gauge,
    plot_feature_contributions,
    plot_decision_score
)

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Single Prediction",
    page_icon="📧",
    layout="wide"
)

# ==========================================================
# Load Model
# ==========================================================

model, vectorizer = load_model()

# ==========================================================
# Session State
# ==========================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==========================================================
# Theme Toggle
# ==========================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

st.session_state.theme_mode = "dark" if st.sidebar.toggle("Dark mode", value=st.session_state.theme_mode == "dark") else "light"

# ==========================================================
# Custom CSS
# ==========================================================
if st.session_state.theme_mode == "dark":
    palette = {
        "bg": "linear-gradient(135deg, #020817 0%, #0f172a 35%, #111827 100%)",
        "sidebar": "linear-gradient(180deg, rgba(15,23,42,0.98), rgba(15,23,42,0.9))",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "panel": "rgba(15, 23, 42, 0.7)",
        "border": "rgba(148, 163, 184, 0.18)",
        "shadow": "rgba(2, 6, 23, 0.3)"
    }
else:
    palette = {
        "bg": "linear-gradient(135deg, #f8fbff 0%, #eef6ff 35%, #f8fafc 100%)",
        "sidebar": "linear-gradient(180deg, rgba(15,23,42,0.958), rgba(15,23,42,0.9))",
        "text": "#0f172a",
        "muted": "#475569",
        "panel": "rgba(255,255,255,0.82)",
        "border": "rgba(148, 163, 184, 0.22)",
        "shadow": "rgba(15, 23, 42, 0.08)"
    }

st.markdown(
f"""
<style>

html, body, [data-testid="stAppViewContainer"] {{
    background: {palette['bg']};
    color: {palette['text']};
}}

[data-testid="stSidebar"] {{
    background: {palette['sidebar']};
    border-right: 1px solid {palette['border']};
    color: #f8fafc;
}}

.block-container {{
    padding-top: 2rem;
}}

.stAlert, .stDataFrame, [data-testid="stExpander"], .stFileUploader, .stTextArea {{
    border-radius: 16px;
    border: 1px solid {palette['border']};
    box-shadow: 0 8px 20px {palette['shadow']};
}}

.footer {{
    text-align: center;
    color: {palette['muted']};
    margin-top: 40px;
    font-size: 0.9rem;
}}

div[data-testid="stButton"] > button {{
    border-radius: 12px;
    font-weight: 600;
}}

p, li, h1, h2, h3, h4, h5, h6, label, .stMarkdown {{
    color: {palette['text']};
}}

</style>
""",
unsafe_allow_html=True
)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("📧 Single Prediction")

st.sidebar.markdown("---")

st.sidebar.success(
"""
Predict whether a single
email is Spam or Ham.
"""
)

st.sidebar.markdown("## Current Model")

st.sidebar.write("✅ Perceptron")

st.sidebar.write("✅ TF-IDF")

st.sidebar.write("✅ 10,000 Features")

st.sidebar.write("✅ Unigrams + Bigrams")

st.sidebar.markdown("---")

st.sidebar.info(
"""
Tips

• Paste the complete email.

• Longer emails generally
produce better predictions.

• Decision Score indicates
how confident the model is.
"""
)

# ==========================================================
# Header
# ==========================================================

st.title("📧 Single Email Prediction")

st.caption(
"""
Predict whether an email is Spam or Ham using the trained
Perceptron Machine Learning model.
"""
)

st.divider()

# ==========================================================
# Layout
# ==========================================================

left, right = st.columns([2.3,1])

# ==========================================================
# Left Column
# ==========================================================

with left:

    email = st.text_area(

        "Paste Email",

        height=320,

        placeholder="""
Example

Congratulations!

You have won a $1000 Gift Card.

Click here to claim your prize.

OR

Paste any email here...
"""
    )

# ==========================================================
# Right Column
# ==========================================================

with right:

    st.subheader("Upload Email")

    uploaded_file = st.file_uploader(

        "Upload TXT File",

        type=["txt"]

    )

    if uploaded_file is not None:

        email = uploaded_file.read().decode("utf-8")

    st.markdown("---")

    st.metric(
        "Vocabulary",
        "10,000"
    )

    st.metric(
        "Algorithm",
        "Perceptron"
    )

    st.metric(
        "Features",
        "TF-IDF"
    )

    st.metric(
        "N-grams",
        "1-2"
    )

st.divider()

# ==========================================================
# Predict Button
# ==========================================================

predict = st.button(

    "🚀 Predict Email",

    use_container_width=True

)

# ==========================================================
# Prediction Engine
# ==========================================================

if predict:

    if not email.strip():

        st.warning(
            "⚠️ Please enter or upload an email."
        )

        st.stop()

    with st.spinner("Analyzing email..."):

        # --------------------------------------
        # Preprocess
        # --------------------------------------

        clean_email = preprocess_text(email)

        # --------------------------------------
        # Prediction
        # --------------------------------------

        result = predict_email(clean_email)

        prediction = result["prediction"]

        label = result["label"]

        decision_score = result["decision_score"]

        vector = result["vector"]

        model = result["model"]

        vectorizer = result["vectorizer"]

        confidence = calculate_confidence(
            decision_score
        )

        # --------------------------------------
        # Save History
        # --------------------------------------

        st.session_state.history.append(

            {

                "Time":
                datetime.now().strftime("%H:%M:%S"),

                "Prediction":
                label,

                "Confidence (%)":
                confidence,

                "Decision Score":
                round(decision_score,4)

            }

        )

    st.success("✅ Analysis Completed")

    st.divider()

    # ======================================================
    # Prediction Result
    # ======================================================

    col1, col2 = st.columns(2)

    # ------------------------------------------------------
    # Prediction Card
    # ------------------------------------------------------

    with col1:

        st.subheader("📌 Prediction")

        if prediction == 1:

            st.error("🚨 SPAM EMAIL")

        else:

            st.success("✅ HAM EMAIL")

        st.metric(
            "Decision Score",
            f"{decision_score:.4f}"
        )

    # ------------------------------------------------------
    # Confidence
    # ------------------------------------------------------

    with col2:

        st.subheader("🎯 Confidence")

        st.metric(
            "Estimated Confidence",
            f"{confidence:.1f}%"
        )

        fig = plot_confidence_gauge(confidence)

        st.pyplot(
            fig,
            use_container_width=True
        )

    st.divider()

    # ======================================================
    # Decision Interpretation
    # ======================================================

    st.subheader("🧠 Decision Interpretation")

    explanation = explain_decision(decision_score)

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Predicted Class",
            explanation["Prediction"]
        )

    with c2:

        st.metric(
            "Confidence Level",
            explanation["Confidence"]
        )

    with c3:

        st.metric(
            "Decision Score",
            explanation["Score"]
        )

    st.info(
        """
### Understanding the Decision Score

• Positive score → Spam

• Negative score → Ham

• Larger absolute values indicate higher confidence.

• Scores close to zero indicate uncertainty.
"""
    )

    st.divider()

    # ======================================================
    # Preprocessed Email
    # ======================================================

    with st.expander("🧹 View Preprocessed Email"):

        st.write(clean_email)

    st.divider()

    # ======================================================
    # Explainable AI
    # ======================================================

    st.header("🧠 Explainable AI (XAI)")

    explanation_df = explain_prediction(
        model,
        vectorizer,
        vector
    )

    if explanation_df.empty:

        st.warning(
            "No influential words were found."
        )

    else:

        st.success(
            f"The prediction was influenced by **{len(explanation_df)}** features."
        )

        # --------------------------------------------------
        # Feature Summary
        # --------------------------------------------------

        total_features = len(explanation_df)

        spam_features = (
            explanation_df["Direction"] == "Spam"
        ).sum()

        ham_features = (
            explanation_df["Direction"] == "Ham"
        ).sum()

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Total Features",
                total_features
            )

        with m2:

            st.metric(
                "Spam Features",
                spam_features
            )

        with m3:

            st.metric(
                "Ham Features",
                ham_features
            )

        st.divider()

        # --------------------------------------------------
        # Feature Contribution Chart
        # --------------------------------------------------

        st.subheader("📊 Top Feature Contributions")

        fig = plot_feature_contributions(
            explanation_df,
            top_n=10
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        st.caption(
            """
Positive contributions push the prediction toward Spam.

Negative contributions push the prediction toward Ham.
"""
        )

        st.divider()

        # --------------------------------------------------
        # Top Words
        # --------------------------------------------------

        left, right = st.columns(2)

        with left:

            st.subheader("🚨 Top Spam Features")

            spam_df = top_spam_words(
                explanation_df,
                top_n=10
            )

            st.dataframe(
                spam_df,
                use_container_width=True,
                hide_index=True
            )

        with right:

            st.subheader("✅ Top Ham Features")

            ham_df = top_ham_words(
                explanation_df,
                top_n=10
            )

            st.dataframe(
                ham_df,
                use_container_width=True,
                hide_index=True
            )

        st.divider()

        # --------------------------------------------------
        # Complete Feature Table
        # --------------------------------------------------

        st.subheader("📋 Complete Feature Contributions")

        st.dataframe(
            explanation_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        csv = explanation_df.to_csv(index=False)

        st.download_button(
            "📥 Download Explanation (CSV)",
            data=csv,
            file_name="feature_contributions.csv",
            mime="text/csv"
        )

    st.divider()

# ==========================================================
# Prediction History
# ==========================================================

st.header("📜 Prediction History")

if len(st.session_state.history) == 0:

    st.info("No predictions have been made in this session.")

else:

    history_df = pd.DataFrame(
        st.session_state.history
    )

    # ======================================================
    # Summary Metrics
    # ======================================================

    total_predictions = len(history_df)

    spam_predictions = (
        history_df["Prediction"] == "Spam"
    ).sum()

    ham_predictions = (
        history_df["Prediction"] == "Ham"
    ).sum()

    avg_confidence = round(
        history_df["Confidence (%)"].mean(),
        2
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Total Predictions",
            total_predictions
        )

    with c2:

        st.metric(
            "Spam",
            spam_predictions
        )

    with c3:

        st.metric(
            "Ham",
            ham_predictions
        )

    with c4:

        st.metric(
            "Average Confidence",
            f"{avg_confidence}%"
        )

    st.divider()

    # ======================================================
    # Prediction Distribution
    # ======================================================

    st.subheader("📊 Prediction Distribution")

    prediction_counts = (
        history_df["Prediction"]
        .value_counts()
    )

    st.bar_chart(prediction_counts)

    st.divider()

    # ======================================================
    # Confidence Trend
    # ======================================================

    st.subheader("📈 Confidence Trend")

    confidence_df = history_df.copy()

    confidence_df.index = range(
        1,
        len(confidence_df) + 1
    )

    st.line_chart(
        confidence_df["Confidence (%)"]
    )

    st.divider()

    # ======================================================
    # History Table
    # ======================================================

    st.subheader("📋 Prediction Records")

    st.dataframe(

        history_df,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ======================================================
    # Download History
    # ======================================================

    csv = history_df.to_csv(index=False)

    st.download_button(

        label="📥 Download Prediction History",

        data=csv,

        file_name="prediction_history.csv",

        mime="text/csv",

        use_container_width=True

    )

    # ======================================================
    # Clear History
    # ======================================================

    if st.button(

        "🗑️ Clear Prediction History",

        use_container_width=True,

        type="secondary"

    ):

        st.session_state.history = []

        st.success(
            "Prediction history cleared successfully."
        )

        st.rerun()

# ==========================================================
# About the Model
# ==========================================================

st.divider()

st.header("ℹ️ About This Model")

left, right = st.columns(2)

with left:

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

with right:

    st.info(
        """
### Explainable AI

This application explains predictions by displaying:

- Influential words
- Feature contributions
- Decision score
- Prediction confidence

This improves transparency and helps users understand why the model classified an email as Spam or Ham.
"""
    )

# ==========================================================
# Footer
# ==========================================================

st.divider()

st.markdown(
"""
<div style="text-align:center;color:gray;">

## 📧 Email Spam Classification System

Built with

<b>Python • Scikit-learn • TF-IDF • Perceptron • Streamlit</b>

Machine Learning Assignment

© 2026 Oli Bakala

</div>
""",
unsafe_allow_html=True
)        