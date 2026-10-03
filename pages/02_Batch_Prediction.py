"""Batch email classification from an uploaded CSV file."""

import hashlib
import io
import time

import pandas as pd
import streamlit as st

from utils import theme
from utils.model_loader import load_model, predict_batch
from utils.preprocessing import preprocess_series
from utils.sidebar import setup_page
from utils.visualization import plot_length_histogram, plot_pie

setup_page("Batch Prediction", "📂")

SIGNATURE_KEY = "batch_signature"
RESULT_KEY = "batch_result"
COLUMN_KEY = "batch_text_column"


@st.cache_data(show_spinner="Parsing CSV…")
def _read_csv(raw: bytes, signature: str) -> pd.DataFrame:
    """Parse the uploaded CSV. Cached so filtering does not re-read the file."""
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def _top_words(
    clean: pd.Series, predictions: pd.Series, label: str, top_n: int = 20
) -> pd.Series:
    """Most frequent cleaned tokens among predictions of a single label."""
    mask = predictions == label
    if not mask.any():
        return pd.Series(dtype="int64")
    return clean[mask].str.split().explode().value_counts().head(top_n)


theme.page_header(
    "📂 Batch Email Classification",
    "Upload a CSV file and classify every email with the trained Perceptron model.",
)

st.divider()

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

raw_bytes = uploaded_file.getvalue()
signature = hashlib.sha1(raw_bytes).hexdigest()

# A different file invalidates everything derived from the previous one.
if st.session_state.get(SIGNATURE_KEY) != signature:
    st.session_state[SIGNATURE_KEY] = signature
    st.session_state[RESULT_KEY] = None
    st.session_state[COLUMN_KEY] = None

source_df = _read_csv(raw_bytes, signature)

if source_df.empty:
    st.warning("The uploaded CSV has no rows.")
    st.stop()

st.success(f"Loaded {len(source_df):,} rows × {source_df.shape[1]} columns.")

with st.expander("Dataset Preview", expanded=True):
    st.dataframe(source_df.head(10))

text_column = st.selectbox("Select the email text column", list(source_df.columns))

st.divider()

if st.button("🚀 Predict Entire Dataset", width="stretch"):
    model, vectorizer = load_model()

    progress_bar = st.progress(0.0, "Preprocessing emails…")
    status = st.empty()
    started = time.perf_counter()

    cleaned = preprocess_series(source_df[text_column])

    def report(done: int, total: int) -> None:
        fraction = done / total if total else 1.0
        progress_bar.progress(
            fraction, f"Scoring emails… {done:,} / {total:,}"
        )

    predictions = predict_batch(
        cleaned.tolist(),
        model=model,
        vectorizer=vectorizer,
        chunk_size=2_000,
        progress=report,
    )

    elapsed = time.perf_counter() - started
    speed = len(source_df) / elapsed if elapsed > 0 else 0.0

    st.session_state[RESULT_KEY] = pd.concat(
        [
            source_df.reset_index(drop=True),
            cleaned.rename("Clean Email"),
            predictions,
        ],
        axis=1,
    )
    st.session_state[COLUMN_KEY] = text_column

    progress_bar.empty()
    status.empty()

    st.success("✅ Batch prediction completed successfully!")

    theme.metric_row(
        [
            ("Emails Processed", f"{len(source_df):,}"),
            ("Execution Time", f"{elapsed:.2f} sec"),
            ("Processing Speed", f"{speed:,.0f} emails/sec"),
        ]
    )

    st.divider()

result_df = st.session_state.get(RESULT_KEY)

if result_df is None:
    st.header("📊 Prediction Analytics")
    st.info(
        "👆 Upload a CSV file and click **Predict Entire Dataset** to view analytics."
    )
    st.stop()

# The widget resets when the file changes, so the column is always read from the
# stored result instead of the widget state.
analysed_column = st.session_state[COLUMN_KEY]

total_rows = len(result_df)
spam_count = int((result_df["Prediction"] == "Spam").sum())
ham_count = int((result_df["Prediction"] == "Ham").sum())
avg_confidence = round(float(result_df["Confidence (%)"].mean()), 2)

theme.metric_row(
    [
        ("Total Emails", f"{total_rows:,}"),
        ("Spam", f"{spam_count:,}"),
        ("Ham", f"{ham_count:,}"),
        ("Average Confidence", f"{avg_confidence}%"),
    ]
)

st.divider()

filter_col, search_col = st.columns([1, 2])

with filter_col:
    prediction_filter = st.selectbox("Filter Prediction", ["All", "Spam", "Ham"])

with search_col:
    search = st.text_input("Search Email", placeholder="Search by keyword...")

filtered_df = result_df

if prediction_filter != "All":
    filtered_df = filtered_df[filtered_df["Prediction"] == prediction_filter]

if search.strip():
    filtered_df = filtered_df[
        filtered_df[analysed_column]
        .astype(str)
        .str.contains(search, case=False, na=False, regex=False)
    ]

st.info(f"Displaying {len(filtered_df):,} of {total_rows:,} emails.")

st.divider()

if filtered_df.empty:
    st.warning("No emails match the current filter.")
    st.divider()
else:
    counts_col, pie_col = st.columns(2)

    with counts_col:
        st.subheader("📈 Spam vs Ham")
        st.bar_chart(result_df["Prediction"].value_counts())

    with pie_col:
        st.subheader("🥧 Prediction Distribution")
        st.pyplot(plot_pie(result_df["Prediction"].value_counts(), "Predictions"))

    st.divider()

    st.subheader("📊 Confidence Distribution")
    st.bar_chart(filtered_df["Confidence (%)"])

    st.divider()

    st.subheader("📈 Decision Scores")
    st.line_chart(filtered_df["Decision Score"])

    st.divider()

    st.subheader("📈 Dataset Statistics")

    lengths = filtered_df[analysed_column].astype(str).str.len()

    theme.metric_row(
        [
            ("Average Length", f"{lengths.mean():.0f} chars"),
            ("Shortest Email", f"{lengths.min()} chars"),
            ("Longest Email", f"{lengths.max()} chars"),
            ("Median Length", f"{lengths.median():.0f} chars"),
        ]
    )

    st.divider()

    st.subheader("📊 Email Length Distribution")
    st.pyplot(plot_length_histogram(lengths))

    st.divider()

    st.header("🧠 Most Frequent Words")

    clean_column = filtered_df["Clean Email"]

    for label, heading in (("Spam", "🚨 Spam Words"), ("Ham", "🟢 Ham Words")):
        st.subheader(heading)
        word_counts = _top_words(
            clean_column, filtered_df["Prediction"], label, top_n=20
        )
        if word_counts.empty:
            st.info(f"No {label} emails in the current filter.")
        else:
            st.bar_chart(word_counts)
            st.dataframe(word_counts.rename("Count").to_frame())
        st.divider()

    st.subheader("📋 Prediction Results")
    st.dataframe(filtered_df, hide_index=True)

    st.download_button(
        "📥 Download Predictions",
        data=filtered_df.to_csv(index=False),
        file_name="batch_predictions.csv",
        mime="text/csv",
        width="stretch",
    )

    st.divider()

    st.header("📋 Batch Summary")

    st.dataframe(
        pd.DataFrame(
            {
                "Metric": [
                    "Total Emails",
                    "Spam",
                    "Ham",
                    "Spam Rate",
                    "Ham Rate",
                    "Average Confidence",
                ],
                "Value": [
                    f"{total_rows:,}",
                    f"{spam_count:,}",
                    f"{ham_count:,}",
                    f"{spam_count / total_rows * 100:.2f}%" if total_rows else "—",
                    f"{ham_count / total_rows * 100:.2f}%" if total_rows else "—",
                    f"{avg_confidence:.2f}%",
                ],
            }
        ),
        hide_index=True,
    )

    st.divider()

theme.footer()