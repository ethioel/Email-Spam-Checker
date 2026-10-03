"""Shared visual theme for the Email Spam Classification application.

This module is the single source of truth for the colour palettes, the global
stylesheet and the small presentation helpers reused across every page. Pages
never inline their own CSS; they call :func:`utils.sidebar.setup_page`, which
delegates here, so the app can never drift out of sync.
"""

from __future__ import annotations

from string import Template

import streamlit as st

# ==========================================================
# Branding
# ==========================================================

APP_NAME = "MailGuard"
APP_TITLE = "Email Spam Classification System"

THEME_KEY = "theme_mode"
TOGGLE_KEY = "sidebar_dark_mode"

# ==========================================================
# Palettes
# ==========================================================

_PALETTES = {
    "dark": {
        "bg": "linear-gradient(135deg, #020817 0%, #0f172a 35%, #111827 100%)",
        "sidebar": "linear-gradient(180deg, rgba(15,23,42,0.98), rgba(15,23,42,0.9))",
        "hero": "linear-gradient(135deg, rgba(14,165,233,0.18), rgba(59,130,246,0.08))",
        "panel": "rgba(15, 23, 42, 0.82)",
        "panel_soft": "rgba(30, 41, 59, 0.55)",
        "card": "rgba(15, 23, 42, 0.7)",
        "chip": "rgba(148, 163, 184, 0.08)",
        "pill_bg": "rgba(14, 165, 233, 0.12)",
        "pill_border": "rgba(125, 211, 252, 0.25)",
        # Hex variants for matplotlib, which cannot parse CSS rgba() values.
        "chart_bg": "#0f172a",
        "chart_border": "#334155",
        "border": "rgba(148, 163, 184, 0.18)",
        "shadow": "rgba(2, 6, 23, 0.38)",
        "text": "#e2e8f0",
        "heading": "#f1f5f9",
        "muted": "#94a3b8",
        "accent": "#7dd3fc",
        "accent_ink": "#082f49",
        "hero_text": "#e2e8f0",
        "hero_muted": "#cbd5e1",
    },
    "light": {
        "bg": "linear-gradient(135deg, #f8fbff 0%, #eef6ff 35%, #f8fafc 100%)",
        "sidebar": "linear-gradient(180deg, rgba(15,23,42,0.958), rgba(15,23,42,0.9))",
        "hero": "linear-gradient(135deg, rgba(125, 211, 252, 0.18), rgba(59, 130, 246, 0.08))",
        "panel": "rgba(255, 255, 255, 0.82)",
        "panel_soft": "rgba(241, 245, 249, 0.85)",
        "card": "rgba(255, 255, 255, 0.82)",
        "chip": "rgba(15, 23, 42, 0.04)",
        "pill_bg": "rgba(2, 132, 199, 0.10)",
        "pill_border": "rgba(2, 132, 199, 0.28)",
        # Hex variants for matplotlib, which cannot parse CSS rgba() values.
        "chart_bg": "#ffffff",
        "chart_border": "#cbd5e1",
        "border": "rgba(148, 163, 184, 0.22)",
        "shadow": "rgba(15, 23, 42, 0.10)",
        "text": "#0f172a",
        "heading": "#020617",
        "muted": "#475569",
        "accent": "#0284c7",
        "accent_ink": "#ffffff",
        "hero_text": "#0f172a",
        "hero_muted": "#334155",
    },
}

# ==========================================================
# Global stylesheet
# ==========================================================

# The <style> wrapper is mandatory. Streamlit renders markdown with raw HTML
# enabled, but bare CSS rules are not HTML: without the wrapper the browser
# treats them as a paragraph of text and prints them on the page.
_CSS = Template(
    """
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: $bg;
    color: $text;
}

[data-testid="stSidebar"] {
    background: $sidebar;
    border-right: 1px solid $border;
    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
}

.main-title {
    font-size: 2.7rem;
    font-weight: 800;
    text-align: center;
    letter-spacing: -0.06em;
    background: linear-gradient(135deg, $accent, #1d4ed8, #38bdf8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin-bottom: 0.2rem;
}

.subtitle {
    text-align: center;
    font-size: 1.08rem;
    color: $muted;
    margin-bottom: 1.4rem;
    font-weight: 500;
}

.hero-banner {
    background: $hero;
    border: 1px solid $border;
    border-radius: 20px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 20px 45px $shadow;
    margin-bottom: 1.25rem;
}

.hero-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: $hero_text;
}

.hero-text {
    margin-top: 0.35rem;
    color: $hero_muted;
}

.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-top: 0.6rem;
}

.pill {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: $pill_bg;
    border: 1px solid $pill_border;
    border-radius: 999px;
    color: $text;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.45rem 0.8rem;
}

.metric-card, .feature-box {
    padding: 1.1rem 1rem;
    border-radius: 16px;
    text-align: center;
    border: 1px solid $border;
    background: $panel_soft;
    box-shadow: 0 8px 20px $shadow;
    margin-bottom: 15px;
}

.step-chip {
    text-align: center;
    padding: 0.7rem 0.5rem;
    border: 1px solid $border;
    border-radius: 12px;
    background: $chip;
}

.footer {
    text-align: center;
    color: $muted;
    font-size: 0.92rem;
    padding-top: 0.5rem;
}

.brand-mark {
    display: flex;
    align-items: center;
    gap: 10px;
}

.brand-name {
    font-size: 1.15rem;
    color: #f8fafc;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem;
    font-weight: 700;
}

div[data-testid="stButton"] > button, .stDownloadButton > button {
    border-radius: 12px;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.stAlert, .stDataFrame, [data-testid="stExpander"],
.stFileUploader, .stSelectbox, .stTextInput, .stTextArea,
.stNumberInput {
    border-radius: 14px;
}

p, li, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
    color: $text;
}

h1, h2, h3, h4 {
    color: $heading;
}
</style>
"""
)


# ==========================================================
# Theme state
# ==========================================================

def mode() -> str:
    """Return the active theme mode without assuming it was initialised."""
    return st.session_state.get(THEME_KEY, "dark")


def palette() -> dict:
    """Return the colour palette for the active theme."""
    return _PALETTES[mode()]


def inject_css() -> None:
    """Inject the global stylesheet for the active theme."""
    st.markdown(_CSS.substitute(palette()), unsafe_allow_html=True)


def render_toggle() -> str:
    """Render the single dark/light toggle and return the active mode."""
    st.session_state[THEME_KEY] = (
        "dark"
        if st.sidebar.toggle(
            "🌙 Dark mode",
            value=mode() == "dark",
            key=TOGGLE_KEY,
        )
        else "light"
    )
    return st.session_state[THEME_KEY]


# ==========================================================
# Presentation helpers
# ==========================================================

def hero(title: str, subtitle: str, pills: list[str]) -> None:
    """Render the gradient hero banner with pill badges."""
    badges = "".join(f'<span class="pill">{pill}</span>' for pill in pills)
    st.markdown(
        f"""
        <div class="hero-banner">
            <div style="display:flex;justify-content:space-between;
                        align-items:center;gap:1rem;flex-wrap:wrap;">
                <div>
                    <strong class="hero-heading">{title}</strong>
                    <div class="hero-text">{subtitle}</div>
                </div>
                <div class="pill-row">{badges}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    """Render the centered gradient page title and subtitle."""
    st.markdown(f"<h1 class='main-title'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p class='subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def metric_row(items: list[tuple[str, object]]) -> None:
    """Render a row of equal-width metric cards.

    Parameters
    ----------
    items
        Sequence of ``(label, value)`` pairs. At most six are shown side by
        side before wrapping onto a second row.
    """
    per_row = min(max(len(items), 1), 6)
    for start in range(0, len(items), per_row):
        chunk = items[start : start + per_row]
        for column, (label, value) in zip(st.columns(len(chunk)), chunk, strict=True):
            with column:
                st.metric(label, value)


def feature_row(items: list[tuple[str, str]]) -> None:
    """Render a row of bordered feature cards."""
    for start in range(0, len(items), 3):
        chunk = items[start : start + 3]
        for column, (heading, body) in zip(
            st.columns(len(chunk)), chunk, strict=True
        ):
            with column, st.container(border=True):
                st.markdown(f"### {heading}")
                st.write(body)


def step_row(steps: list[str]) -> None:
    """Render a horizontal row of numbered pipeline steps."""
    for column, step in zip(st.columns(len(steps)), steps, strict=True):
        with column:
            st.markdown(
                f"<div class='step-chip'><strong>{step}</strong></div>",
                unsafe_allow_html=True,
            )


def footer() -> None:
    """Render the shared page footer."""
    st.markdown(
        f"""
        <div class="footer">
            <b>{APP_TITLE}</b><br>
            Machine Learning • Natural Language Processing • Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )