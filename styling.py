"""
styling.py
Central place for CSS overrides and the shared Plotly chart theme.
Keeps the original RWE Command Center look (navy/teal headings, light grey
page background, red accent sliders) but fixes two things the user flagged:

  1. Text areas / text inputs / "black boxes" -> white background,
     thin light-blue border, dark readable text.
  2. Chart axis lines / gridlines -> dark, visible on a white background
     (the demo theme previously rendered them white-on-white).
"""

import plotly.graph_objects as go
import plotly.io as pio

PRIMARY_NAVY = "#1b3a5c"
ACCENT_BLUE = "#2f6fb0"
BORDER_BLUE = "#a9cdf0"
LIGHT_BG = "#f4f6f8"
AXIS_DARK = "#1f2933"
GRID_GREY = "#d5dbe0"

CUSTOM_CSS = f"""
<style>
/* ---------- page background ---------- */
.stApp {{
    background-color: {LIGHT_BG};
}}

/* ---------- headings ---------- */
h1, h2, h3 {{
    color: {PRIMARY_NAVY};
}}

/* ---------- generic "black box" containers (text_area / text_input / code-style boxes) ---------- */
.stTextArea textarea,
.stTextInput input,
.stNumberInput input,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {{
    background-color: #ffffff !important;
    color: {AXIS_DARK} !important;
    border: 1px solid {BORDER_BLUE} !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}}

/* dropdown menu popover */
div[data-baseweb="popover"] div[role="listbox"] {{
    background-color: #ffffff !important;
    border: 1px solid {BORDER_BLUE} !important;
}}
div[data-baseweb="popover"] li {{
    color: {AXIS_DARK} !important;
}}
div[data-baseweb="popover"] li:hover {{
    background-color: #eaf3fc !important;
}}

/* placeholder text */
.stTextArea textarea::placeholder,
.stTextInput input::placeholder {{
    color: #7a8794 !important;
}}

/* ---------- custom "info card" boxes used throughout the app ---------- */
.rwe-card {{
    background-color: #ffffff;
    border: 1px solid {BORDER_BLUE};
    border-radius: 8px;
    padding: 16px 18px;
    margin-bottom: 14px;
    color: {AXIS_DARK};
}}
.rwe-card-title {{
    font-weight: 600;
    color: {PRIMARY_NAVY};
    margin-bottom: 6px;
}}
.rwe-insight {{
    background-color: #eef6ff;
    border-left: 4px solid {ACCENT_BLUE};
    border-radius: 6px;
    padding: 14px 16px;
    margin-top: 10px;
    color: {AXIS_DARK};
}}
.rwe-kpi {{
    background-color: #ffffff;
    border: 1px solid {BORDER_BLUE};
    border-radius: 8px;
    padding: 14px;
    text-align: left;
}}
.rwe-kpi-label {{
    font-size: 12px;
    letter-spacing: 0.04em;
    color: #5b6b7a;
    text-transform: uppercase;
}}
.rwe-kpi-value {{
    font-size: 26px;
    font-weight: 700;
    color: {PRIMARY_NAVY};
}}
.rwe-kpi-sub {{
    font-size: 12px;
    color: #7a8794;
}}

/* ---------- home tiles ---------- */
.rwe-tile button {{
    height: 90px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    background-color: #ffffff !important;
    border: 1px solid {BORDER_BLUE} !important;
    color: {PRIMARY_NAVY} !important;
    border-radius: 10px !important;
}}
.rwe-tile button:hover {{
    background-color: #eaf3fc !important;
    border-color: {ACCENT_BLUE} !important;
}}

/* ---------- top bar ---------- */
.rwe-topbar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid {BORDER_BLUE};
    padding-bottom: 10px;
    margin-bottom: 18px;
}}

/* ---------- sliders keep red accent (as in original) ---------- */
div[data-baseweb="slider"] > div > div {{
    background: {GRID_GREY} !important;
}}

/* tabs */
button[data-baseweb="tab"] {{
    color: {PRIMARY_NAVY};
}}
</style>
"""


def inject_css(st):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def themed_figure(fig: go.Figure, title: str = None, height: int = 420) -> go.Figure:
    """Apply a consistent light theme with dark, visible axis lines/gridlines."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color=AXIS_DARK, size=13),
        title=title,
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        showline=True, linewidth=1.4, linecolor=AXIS_DARK,
        gridcolor=GRID_GREY, zerolinecolor=GRID_GREY,
        tickfont=dict(color=AXIS_DARK),
        title_font=dict(color=AXIS_DARK),
    )
    fig.update_yaxes(
        showline=True, linewidth=1.4, linecolor=AXIS_DARK,
        gridcolor=GRID_GREY, zerolinecolor=GRID_GREY,
        tickfont=dict(color=AXIS_DARK),
        title_font=dict(color=AXIS_DARK),
    )
    return fig


def card(st, title, body_html):
    st.markdown(
        f'<div class="rwe-card"><div class="rwe-card-title">{title}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def insight_box(st, text_html):
    st.markdown(f'<div class="rwe-insight">{text_html}</div>', unsafe_allow_html=True)


def kpi(st, label, value, sub=""):
    st.markdown(
        f"""<div class="rwe-kpi">
                <div class="rwe-kpi-label">{label}</div>
                <div class="rwe-kpi-value">{value}</div>
                <div class="rwe-kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )
