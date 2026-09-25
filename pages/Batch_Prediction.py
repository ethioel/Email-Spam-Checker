# ==========================================================
# Batch Email Prediction
# ==========================================================

import time
import pandas as pd
import streamlit as st

from utils.preprocessing import preprocess_text
from utils.model_loader import (
    load_model,
    calculate_confidence
)

st.set_page_config(
    page_title="Batch Prediction",
    page_icon="📂",
    layout="wide"
)

model, vectorizer = load_model()

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

st.markdown(f"""
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

.metric-card {{
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 8px 20px {palette['shadow']};
}}

.stDataFrame, .stFileUploader, .stSelectbox, .stButton, .stTextInput {{
    border-radius: 14px;
}}

p, li, h1, h2, h3, h4, h5, h6, label, .stMarkdown {{
    color: {palette['text']};
}}

</style>
""",
unsafe_allow_html=True)

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.title("📂 Batch Prediction")

st.sidebar.markdown("---")

st.sidebar.success("""
Upload a CSV file and classify
thousands of emails automatically.
""")

st.sidebar.markdown("### Supported Formats")

st.sidebar.write("✅ CSV")

st.sidebar.write("✅ UTF-8")

st.sidebar.markdown("---")

st.sidebar.info("""
The dataset must contain
at least one column
containing email text.
""")

# ==========================================================
# Header
# ==========================================================

st.title("📂 Batch Email Classification")

st.caption(
"""
Upload a CSV file and classify
multiple emails using the trained
Perceptron model.
"""
)

st.divider()

# ==========================================================
# Upload CSV
# ==========================================================

uploaded_file = st.file_uploader(

    "Upload CSV",

    type=["csv"]

)

if uploaded_file is None:

    st.info("Upload a CSV file to begin.")

    st.stop()

# ==========================================================
# Read Dataset
# ==========================================================

df = pd.read_csv(uploaded_file)

st.success(f"Loaded {len(df):,} rows.")

# ==========================================================
# Dataset Preview
# ==========================================================

with st.expander("Dataset Preview", expanded=True):

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# ==========================================================
# Column Selection
# ==========================================================

text_column = st.selectbox(

    "Select the email text column",

    df.columns

)

if text_column is None:

    st.stop()

st.divider()

predict_batch = st.button(

    "🚀 Predict Entire Dataset",

    use_container_width=True

)

# ==========================================================
# Batch Prediction Engine
# ==========================================================

if predict_batch:

    start_time = time.time()

    progress_bar = st.progress(0)

    status = st.empty()

    results = []

    total_rows = len(df)

    # Make a copy of the uploaded dataset
    result_df = df.copy()

    # ------------------------------------------------------
    # Predict Each Email
    # ------------------------------------------------------

    for i, email in enumerate(result_df[text_column]):

        if pd.isna(email):

            email = ""

        # ----------------------------
        # Preprocess
        # ----------------------------

        clean_email = preprocess_text(str(email))

        # ----------------------------
        # Vectorize
        # ----------------------------

        vector = vectorizer.transform([clean_email])

        # ----------------------------
        # Prediction
        # ----------------------------

        prediction = model.predict(vector)[0]

        decision_score = model.decision_function(vector)[0]

        confidence = calculate_confidence(
            decision_score
        )

        label = "Spam" if prediction == 1 else "Ham"

        results.append({

            "Original Email": str(email),

            "Clean Email": clean_email,

            "Prediction": label,

            "Decision Score": round(
                decision_score,
                4
            ),

            "Confidence (%)": confidence

        })

        # ----------------------------
        # Update Progress
        # ----------------------------

        percent = int(

            ((i + 1) / total_rows) * 100

        )

        progress_bar.progress(percent)

        status.text(

            f"Processing {i+1:,} / {total_rows:,} emails..."

        )

    # ------------------------------------------------------
    # Finish
    # ------------------------------------------------------

    end_time = time.time()

    elapsed = end_time - start_time

    speed = total_rows / elapsed if elapsed > 0 else 0

    progress_bar.empty()

    status.empty()

    # ------------------------------------------------------
    # Create Results DataFrame
    # ------------------------------------------------------

    predictions = pd.DataFrame(results)

    result_df = pd.concat(

        [

            result_df.reset_index(drop=True),

            predictions.drop(columns=["Original Email"])

        ],

        axis=1

    )
    

    st.success("✅ Batch prediction completed successfully!")

    # ======================================================
    # Performance Metrics
    # ======================================================

    st.subheader("⚡ Batch Performance")

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(

            "Emails Processed",

            f"{total_rows:,}"

        )

    with m2:

        st.metric(

            "Execution Time",

            f"{elapsed:.2f} sec"

        )

    with m3:

        st.metric(

            "Processing Speed",

            f"{speed:.1f} Emails/sec"

        )

    st.divider()

    # ======================================================
    # Prediction Summary
    # ======================================================

    spam_count = (

        result_df["Prediction"] == "Spam"

    ).sum()

    ham_count = (

        result_df["Prediction"] == "Ham"

    ).sum()

    avg_conf = round(

        result_df["Confidence (%)"].mean(),

        2

    )
    # Save results for later use
    st.session_state["result_df"] = result_df
    

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(

            "Total Emails",

            total_rows

        )

    with c2:

        st.metric(

            "Spam",

            spam_count

        )

    with c3:

        st.metric(

            "Ham",

            ham_count

        )

    with c4:

        st.metric(

            "Average Confidence",

            f"{avg_conf}%"

        )

    st.divider()

# ==========================================================
# Prediction Analytics Dashboard
# ==========================================================

st.header("📊 Prediction Analytics")
# ==========================================================
# Load prediction results
# ==========================================================

if "result_df" not in st.session_state:

    st.info("👆 Upload a CSV file and click **Predict Entire Dataset** to view analytics.")

    st.stop()

result_df = st.session_state["result_df"]

total_rows = len(result_df)

spam_count = (result_df["Prediction"] == "Spam").sum()

ham_count = (result_df["Prediction"] == "Ham").sum()

avg_conf = round(
    result_df["Confidence (%)"].mean(),
    2
)

# ----------------------------------------------------------
# Filter & Search
# ----------------------------------------------------------

left, right = st.columns([1, 2])

with left:

    prediction_filter = st.selectbox(

        "Filter Prediction",

        [

            "All",

            "Spam",

            "Ham"

        ]

    )

with right:

    search = st.text_input(

        "Search Email",

        placeholder="Search by keyword..."

    )

# ----------------------------------------------------------
# Apply Filter
# ----------------------------------------------------------

filtered_df = result_df.copy()

if prediction_filter != "All":

    filtered_df = filtered_df[

        filtered_df["Prediction"] == prediction_filter

    ]

if search.strip():

    filtered_df = filtered_df[

        filtered_df[text_column]

        .astype(str)

        .str.contains(

            search,

            case=False,

            na=False

        )

    ]

st.success(

    f"Displaying {len(filtered_df):,} of {len(result_df):,} emails."

)

st.divider()

# ==========================================================
# Charts
# ==========================================================

left, right = st.columns(2)

# ----------------------------------------------------------
# Spam vs Ham
# ----------------------------------------------------------

with left:

    st.subheader("📈 Spam vs Ham")

    counts = result_df["Prediction"].value_counts()

    st.bar_chart(counts)

# ----------------------------------------------------------
# Pie Chart
# ----------------------------------------------------------

with right:

    st.subheader("🥧 Prediction Distribution")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(

        counts,

        labels=counts.index,

        autopct="%1.1f%%",

        startangle=90

    )

    ax.axis("equal")

    st.pyplot(fig)

st.divider()

# ==========================================================
# Confidence Distribution
# ==========================================================

st.subheader("📊 Confidence Distribution")

st.bar_chart(

    result_df["Confidence (%)"]

)

st.divider()

# ==========================================================
# Decision Score Distribution
# ==========================================================

st.subheader("📈 Decision Scores")

st.line_chart(

    result_df["Decision Score"]

)

st.divider()

# ==========================================================
# Dataset Preview
# ==========================================================

st.subheader("📋 Prediction Results")

st.dataframe(

    filtered_df,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================================
# Download Results
# ==========================================================

csv = filtered_df.to_csv(index=False)

st.download_button(

    label="📥 Download Predictions",

    data=csv,

    file_name="batch_predictions.csv",

    mime="text/csv"

)

# ==========================================================
# Dataset Statistics
# ==========================================================

st.divider()

st.header("📈 Dataset Statistics")

stats_df = filtered_df.copy()

stats_df["Email Length"] = (
    stats_df[text_column]
    .astype(str)
    .str.len()
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Average Length",
        f"{stats_df['Email Length'].mean():.0f} chars"
    )

with c2:
    st.metric(
        "Shortest Email",
        f"{stats_df['Email Length'].min()} chars"
    )

with c3:
    st.metric(
        "Longest Email",
        f"{stats_df['Email Length'].max()} chars"
    )

with c4:
    st.metric(
        "Median Length",
        f"{stats_df['Email Length'].median():.0f} chars"
    )

st.divider()

# ==========================================================
# Email Length Distribution
# ==========================================================

st.subheader("📊 Email Length Distribution")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(
    stats_df["Email Length"],
    bins=30
)

ax.set_xlabel("Characters")

ax.set_ylabel("Number of Emails")

st.pyplot(fig)

st.divider()

# ==========================================================
# Most Frequent Spam Words
# ==========================================================

st.header("🧠 Most Frequent Spam Words")

spam_words = (
    filtered_df[
        filtered_df["Prediction"]=="Spam"
    ]["Clean Email"]
    .str.split()
    .explode()
)

if len(spam_words):

    spam_counts = (
        spam_words
        .value_counts()
        .head(20)
    )

    st.bar_chart(spam_counts)

    st.dataframe(
        spam_counts.rename("Count"),
        use_container_width=True
    )

else:

    st.info("No Spam emails found.")

st.divider()

# ==========================================================
# Most Frequent Ham Words
# ==========================================================

st.header("🟢 Most Frequent Ham Words")

ham_words = (
    filtered_df[
        filtered_df["Prediction"]=="Ham"
    ]["Clean Email"]
    .str.split()
    .explode()
)

if len(ham_words):

    ham_counts = (
        ham_words
        .value_counts()
        .head(20)
    )

    st.bar_chart(ham_counts)

    st.dataframe(
        ham_counts.rename("Count"),
        use_container_width=True
    )

else:

    st.info("No Ham emails found.")

st.divider()

# ==========================================================
# Final Summary
# ==========================================================

st.header("📋 Batch Summary")

summary = pd.DataFrame({

    "Metric":[

        "Total Emails",

        "Spam",

        "Ham",

        "Spam Rate",

        "Ham Rate",

        "Average Confidence"

    ],

    "Value":[

        total_rows,

        spam_count,

        ham_count,

        f"{spam_count/total_rows*100:.2f}%",

        f"{ham_count/total_rows*100:.2f}%",

        f"{avg_conf:.2f}%"

    ]

})

st.dataframe(

    summary,

    use_container_width=True,

    hide_index=True

)

st.divider()

# ==========================================================
# Footer
# ==========================================================

st.markdown(
"""
<div style="text-align:center">

### 📂 Batch Email Classification

Built with

**Python • Scikit-learn • TF-IDF • Perceptron • Streamlit**

Machine Learning Assignment

© 2026 Oli Bakala

</div>
""",
unsafe_allow_html=True
)