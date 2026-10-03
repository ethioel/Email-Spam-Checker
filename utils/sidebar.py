"""Shared page chrome: configuration, sidebar and theme injection.

Every page starts with a single call to :func:`setup_page`. That guarantees the
page config is set first (Streamlit requires it), the sidebar renders before the
stylesheet so a theme toggle flip is always honoured on the same rerun, and every
page shares one stylesheet instead of embedding its own copy.

Navigation is intentionally *not* rendered here. Streamlit already discovers the
files in ``pages/`` and draws the navigation itself; adding manual
``st.sidebar.page_link`` entries produced duplicate links and, on Streamlit
>= 1.36, raised ``StreamlitPageNotFoundError`` because ``page_link`` only
accepts pages registered through ``st.Page``/``st.navigation``. Sidebar order is
therefore controlled by the numeric filename prefixes in ``pages/``.
"""

from __future__ import annotations

import streamlit as st

from utils import theme


def _render_brand() -> None:
    """Render the branded header, linking to the deployed app."""
    st.sidebar.markdown(
        f"""
        <div class="brand-mark">
            <span style="font-size:1.6rem;">📧</span>
            <a class="brand-name" href="{theme.APP_URL}"
               target="_blank" rel="noopener">{theme.APP_NAME}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Intelligent spam detection for real inboxes")


def _render_model_health(metrics: dict) -> None:
    """Render the model health footer using the evaluated metrics."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model health")
    st.sidebar.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    st.sidebar.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")


def render_sidebar(metrics: dict) -> None:
    """Render brand, theme toggle and model health using the loaded metrics."""
    _render_brand()
    st.sidebar.markdown("---")
    theme.render_toggle()
    _render_model_health(metrics)


def setup_page(title: str, icon: str) -> None:
    """Bootstrap a page: config, sidebar and stylesheet, in that order."""
    from utils.model_loader import load_metrics

    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_sidebar(load_metrics())
    theme.inject_css()