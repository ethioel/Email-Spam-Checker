import streamlit as st

# ==========================================================
# Page Configuration
# ==========================================================
st.set_page_config(
    page_title="Email Spam Classification",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# Theme Toggle
# ==========================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

# Force page-level theme choice from the sidebar toggle
st.session_state.theme_mode = "dark" if st.sidebar.toggle("Dark mode", value=st.session_state.theme_mode == "dark") else "light"

# ==========================================================
# Custom CSS
# ==========================================================
if st.session_state.theme_mode == "dark":
    theme_palette = {
        "bg": "linear-gradient(135deg, #020817 0%, #0f172a 35%, #111827 100%)",
        "panel": "rgba(15, 23, 42, 0.82)",
        "panel_soft": "rgba(30, 41, 59, 0.8)",
        "border": "rgba(148, 163, 184, 0.18)",
        "text": "#e2e8f0",
        "muted": "#94a3b8",
        "accent": "#7dd3fc",
        "accent_text": "#082f49",
        "hero": "linear-gradient(135deg, rgba(14,165,233,0.18), rgba(59,130,246,0.08))",
        "sidebar": "linear-gradient(180deg, rgba(15,23,42,0.98), rgba(15,23,42,0.9))",
        "card": "rgba(15, 23, 42, 0.7)",
        "shadow": "rgba(2, 6, 23, 0.38)"
    }
else:
    theme_palette = {
        "bg": "linear-gradient(135deg, #f8fbff 0%, #eef6ff 35%, #f8fafc 100%)",
        "panel": "rgba(255,255,255,0.8)",
        "panel_soft": "rgba(255,255,255,0.7)",
        "border": "rgba(148, 163, 184, 0.22)",
        "text": "#0f172a",
        "muted": "#475569",
        "accent": "#0284c7",
        "accent_text": "#ffffff",
        "hero": "linear-gradient(135deg, rgba(125, 211, 252, 0.18), rgba(59, 130, 246, 0.08))",
        "sidebar": "linear-gradient(180deg, rgba(15,23,42,0.958), rgba(15,23,42,0.9))",
        "card": "rgba(255,255,255,0.82)",
        "shadow": "rgba(15, 23, 42, 0.1)"
    }

st.markdown(f"""
<style>

html, body, [data-testid="stAppViewContainer"] {{
    background: {theme_palette['bg']};
    color: {theme_palette['text']};
}}

[data-testid="stSidebar"] {{
    background: {theme_palette['sidebar']};
    border-right: 1px solid {theme_palette['border']};
    color: #f8fafc;
}}

section[data-testid="stSidebar"] > div {{
    padding-top: 1rem;
}}

.main-title {{
    font-size: 2.7rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: -0.06em;
    background: linear-gradient(135deg, {theme_palette['accent']}, #1d4ed8, #38bdf8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.2rem;
}}

.subtitle {{
    text-align: center;
    font-size: 1.08rem;
    color: {theme_palette['muted']};
    margin-bottom: 1.4rem;
    font-weight: 500;
}}

.hero-banner {{
    background: {theme_palette['hero']};
    border: 1px solid {theme_palette['border']};
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 20px 45px {theme_palette['shadow']};
    margin-bottom: 1.25rem;
}}

.pill-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.6rem;
}}

.pill {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(125, 211, 252, 0.25);
    border-radius: 999px;
    color: {theme_palette['text']};
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.45rem 0.8rem;
}}

.metric-card {{
    background: {theme_palette['card']};
    padding: 1.15rem 1rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid {theme_palette['border']};
    box-shadow: 0 12px 28px {theme_palette['shadow']};
}}

.feature-box {{
    padding: 1.1rem 1rem;
    border-radius: 16px;
    border: 1px solid {theme_palette['border']};
    background: {theme_palette['panel_soft']};
    margin-bottom: 15px;
    box-shadow: 0 8px 20px {theme_palette['shadow']};
}}

.footer {{
    text-align: center;
    color: {theme_palette['muted']};
    font-size: 0.92rem;
    padding-top: 0.5rem;
}}

[data-testid="stMetricValue"] {{
    font-size: 1.6rem;
    font-weight: 700;
}}

div[data-testid="stButton"] > button {{
    border-radius: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
}}

.st-emotion-cache-1v0mbdj {{
    background: {theme_palette['panel']};
    border: 1px solid {theme_palette['border']};
    border-radius: 16px;
}}

p, li, h1, h2, h3, h4, h5, h6, label, .stMarkdown {{
    color: {theme_palette['text']};
}}

.stDataFrame, .stFileUploader, .stSelectbox, .stTextInput, .stTextArea, .stNumberInput, .stButton > button {{
    border-radius: 12px;
}}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("� MailGuard")
st.sidebar.caption("Spam detection for real inboxes")

st.sidebar.markdown("### Navigate")
for label, page in [
    ("Single Prediction", "Single_Prediction"),
    ("Batch Prediction", "Batch_Prediction"),
    ("Model Analytics", "Model_Analytics"),
    ("Performance", "Performance"),
    ("About", "About"),
]:
    st.sidebar.page_link(f"pages/{page}.py", label=f"{label}")

st.sidebar.markdown("---")

st.sidebar.markdown("### Model health")
st.sidebar.metric("Accuracy", "97.68%")
st.sidebar.metric("F1 Score", "97.53%")

st.sidebar.markdown("---")

# ==========================================================
# Header
# ==========================================================
st.markdown(
    "<h1 class='main-title'>📧 Email Spam Classification System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Machine Learning-powered email filtering with Perceptron + TF-IDF</p>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='hero-banner'>
        <div style='display:flex; justify-content:space-between; align-items:center; gap:1rem; flex-wrap:wrap;'>
            <div>
                <strong style='font-size:1.1rem; color:#0f172a;'>Operational overview</strong>
                <div style='margin-top:0.35rem; color:#334155;'>Real-time spam detection, batch scoring, and explainable model insights.</div>
            </div>
            <div class='pill-row'>
                <span class='pill'>Perceptron</span>
                <span class='pill'>TF-IDF</span>
                <span class='pill'>10k features</span>
                <span class='pill'>1–2 grams</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# Dashboard Metrics
# ==========================================================
st.subheader("📈 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        "97.68%"
    )

with col2:
    st.metric(
        "Precision",
        "98.21%"
    )

with col3:
    st.metric(
        "Recall",
        "96.86%"
    )

with col4:
    st.metric(
        "F1 Score",
        "97.53%"
    )

st.divider()

# ==========================================================
# Minimal Feature Overview
# ==========================================================
feature_cols = st.columns(3)

with feature_cols[0]:
    with st.container(border=True):
        st.markdown("### ⚡ Single Scan")
        st.write("Predict one email in seconds with explainable scoring.")

with feature_cols[1]:
    with st.container(border=True):
        st.markdown("### 📦 Bulk Review")
        st.write("Upload a CSV and classify thousands of emails in one pass.")

with feature_cols[2]:
    with st.container(border=True):
        st.markdown("### 🧠 Model Insight")
        st.write("See the words and signals driving each prediction.")

st.divider()

# ==========================================================
# Compact Workflow
# ==========================================================
st.subheader("⚙️ Workflow")

workflow_cols = st.columns(5)
steps = [
    "Email",
    "Clean",
    "TF-IDF",
    "Score",
    "Result"
]
for idx, col in enumerate(workflow_cols):
    with col:
        st.markdown(f"<div style='text-align:center; padding:0.7rem 0.5rem; border:1px solid rgba(148,163,184,0.2); border-radius:12px; background:rgba(255,255,255,0.05);'> <strong>{steps[idx]}</strong> </div>", unsafe_allow_html=True)
        if idx < len(steps)-1:
            st.markdown("<div style='text-align:center; margin-top:0.4rem; color:#38bdf8;'>↓</div>", unsafe_allow_html=True)

st.divider()

# ==========================================================
# Minimal summary section
# ==========================================================
summary_left, summary_right = st.columns(2)

with summary_left:
    st.subheader("📊 Dataset")
    st.markdown("- Total emails: **193,812**")
    st.markdown("- Ham: **102,159**")
    st.markdown("- Spam: **91,653**")

with summary_right:
    st.subheader("🤖 Model")
    st.markdown("- Algorithm: **Perceptron**")
    st.markdown("- Features: **10,000 TF-IDF**")
    st.markdown("- N-grams: **1–2**")

st.divider()

# ==========================================================
# Quick actions
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

st.markdown(
"""
<div class="footer">

Developed by <b>Oli Bakala</b>

Machine Learning | Natural Language Processing | Streamlit

</div>
""",
unsafe_allow_html=True
)