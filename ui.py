"""
ui.py — Styling & theme module for TerraLeaf Disease Analyser
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Theme: Deep Forest Night · Harvest Gold · Bioluminescent Green
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage in main.py (add at top, after st.set_page_config):

    import ui
    ui.apply_theme()

Helper functions:
    ui.sidebar_header()              → logo + branding block
    ui.sidebar_footer()              → sidebar bottom caption
    ui.page_header("PREDICT")        → top status strip
    ui.section_title("...")          → styled section label
    ui.card("Label", "Value")        → metric card
    ui.severity_badge("Mild")        → coloured severity badge
    ui.recommendation_box(text, sev) → tinted corner-accent box
    ui.upload_empty_state()          → empty upload placeholder
    ui.gauge(value, title, ...)      → themed Plotly gauge
    ui.PLOTLY_BASE                   → dict for fig.update_layout(**ui.PLOTLY_BASE)
    ui.SEV_COLOR                     → {"Healthy": "#4ade80", ...}
    ui.GREEN_SCALE                   → Plotly green-yellow color scale
    ui.GY_HEATMAP_SCALE              → green-to-yellow heatmap scale
"""

import os
import streamlit as st
import plotly.graph_objects as go


# ── Shared constants ──────────────────────────────────────────────────────────

SEV_COLOR = {
    "Healthy":  "#4ade80",
    "Mild":     "#facc15",
    "Moderate": "#fb923c",
    "Severe":   "#f87171",
}

GREEN_SCALE = [
    [0.0,  "#052e16"],
    [0.35, "#166534"],
    [0.65, "#16a34a"],
    [0.85, "#86efac"],
    [1.0,  "#fde047"],
]

GY_HEATMAP_SCALE = [
    [0.0,  "#052e16"],
    [0.3,  "#15803d"],
    [0.55, "#22c55e"],
    [0.75, "#a3e635"],
    [1.0,  "#fde047"],
]

# Alias for backward compat
RG_HEATMAP_SCALE = GY_HEATMAP_SCALE

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f0fdf4", family="Syne"),
    xaxis=dict(
        gridcolor="rgba(74,222,128,0.07)",
        tickfont=dict(color="#5a8a5a"),
        linecolor="rgba(74,222,128,0.12)",
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor="rgba(74,222,128,0.07)",
        tickfont=dict(color="#5a8a5a"),
        linecolor="rgba(74,222,128,0.12)",
        zeroline=False,
    ),
    margin=dict(t=40, b=20, l=10, r=10),
)


# ── Main theme injector ───────────────────────────────────────────────────────

def apply_theme():
    """
    Inject all CSS into the Streamlit app.
    Call ONCE right after st.set_page_config() in main.py.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── KEYFRAMES ─────────────────────────────────────────────────────── */
    @keyframes pulse-dot {
        0%,100% { opacity:1; transform:scale(1); box-shadow: 0 0 0 0 rgba(74,222,128,0.6); }
        50%      { opacity:.8; transform:scale(1.2); box-shadow: 0 0 0 4px rgba(74,222,128,0); }
    }
    @keyframes pulse-pill {
        0%,100% { box-shadow: 0 0 0 0 rgba(74,222,128,0.0); }
        50%      { box-shadow: 0 0 18px rgba(74,222,128,0.25); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position:  200% center; }
    }
    @keyframes float-in {
        from { opacity:0; transform:translateY(8px); }
        to   { opacity:1; transform:translateY(0); }
    }
    @keyframes scan-line {
        0%   { top: -2px; }
        100% { top: 100%; }
    }

    /* ── CSS VARIABLES ─────────────────────────────────────────────────── */
    :root {
        --bg-base:        #050c05;
        --bg-surface:     #09130a;
        --bg-elevated:    #0e1b0e;
        --bg-card:        #0b170c;
        --bg-glass:       rgba(9,19,10,0.85);

        --green-vivid:    #4ade80;
        --green-bright:   #22c55e;
        --green-mid:      #16a34a;
        --green-dark:     #166534;
        --green-deep:     #052e16;
        --green-neon:     #39ff6a;

        --gold-vivid:     #fde047;
        --gold-bright:    #facc15;
        --gold-mid:       #eab308;
        --gold-muted:     #ca8a04;
        --gold-deep:      #713f12;

        --amber:          #fb923c;

        --text-primary:   #edfaed;
        --text-secondary: #a7d9a7;
        --text-muted:     #547a54;
        --text-dim:       #2d4a2d;

        --border-green:   rgba(74,222,128,0.14);
        --border-gold:    rgba(253,224,71,0.2);
        --border-bright:  rgba(74,222,128,0.4);

        --shadow-card:    0 8px 40px rgba(0,0,0,0.7), 0 1px 0 rgba(74,222,128,0.06) inset;
        --glow-green:     0 0 30px rgba(74,222,128,0.12), 0 0 80px rgba(74,222,128,0.04);
        --glow-gold:      0 0 30px rgba(253,224,71,0.15), 0 0 80px rgba(253,224,71,0.05);
    }

    /* ── GLOBAL ─────────────────────────────────────────────────────────── */
    html, body, [data-testid="stApp"] {
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Subtle dot-grid texture */
    [data-testid="stApp"]::before {
        content: '';
        position: fixed; top:0; left:0; right:0; bottom:0;
        background-image:
            radial-gradient(circle, rgba(74,222,128,0.028) 1px, transparent 1px);
        background-size: 32px 32px;
        pointer-events: none; z-index: 0;
    }

    /* Ambient glow blobs */
    [data-testid="stApp"]::after {
        content: '';
        position: fixed; top:0; left:0; right:0; bottom:0;
        background:
            radial-gradient(ellipse 65% 50% at 15% 8%,  rgba(22,163,74,0.065) 0%, transparent 55%),
            radial-gradient(ellipse 45% 38% at 85% 88%,  rgba(253,224,71,0.045) 0%, transparent 50%),
            radial-gradient(ellipse 38% 30% at 55% 45%,  rgba(74,222,128,0.03)  0%, transparent 45%);
        pointer-events: none; z-index: 0;
    }

    /* ── SIDEBAR ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(178deg, #060d06 0%, #08110a 55%, #0b190b 100%) !important;
        border-right: 1px solid var(--border-green) !important;
        box-shadow: 8px 0 48px rgba(0,0,0,0.65), 1px 0 0 rgba(74,222,128,0.05) !important;
        position: relative !important;
        z-index: 10 !important;
    }
    /* Top accent line */
    [data-testid="stSidebar"]::before {
        content: '';
        position: absolute; top:0; left:0; right:0; height:2px;
        background: linear-gradient(90deg,
            transparent 0%, var(--green-dark) 20%,
            var(--gold-bright) 60%, var(--green-dark) 80%, transparent 100%);
        z-index: 1;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stSidebar"] hr {
        border: none !important;
        border-top: 1px solid var(--border-green) !important;
        margin: .7rem 0 !important;
        opacity: 0.7 !important;
    }

    /* ── SIDEBAR LOGO IMAGE ──────────────────────────────────────────────── */
    [data-testid="stSidebar"] [data-testid="stImage"] img {
        display: block !important;
        margin: .6rem auto .2rem auto !important;
        filter: drop-shadow(0 0 12px rgba(74,222,128,0.35)) drop-shadow(0 0 28px rgba(74,222,128,0.12));
        transition: filter 0.3s ease;
    }
    [data-testid="stSidebar"] [data-testid="stImage"] img:hover {
        filter: drop-shadow(0 0 20px rgba(74,222,128,0.5)) drop-shadow(0 0 44px rgba(74,222,128,0.18));
    }

    /* ── SIDEBAR BRAND TEXT ─────────────────────────────────────────────── */
    /* Gradient MUST be a CSS class — Streamlit sanitiser strips inline
       -webkit-background-clip and -webkit-text-fill-color from st.markdown */
    .sidebar-brand-wrap {
        text-align: center;
        padding: .1rem 0 .4rem 0;
    }
    .sidebar-brand {
        font-family: 'Syne', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -.02em;
        line-height: 1.1;
        background: linear-gradient(115deg, #4ade80 0%, #22c55e 35%, #fde047 80%, #facc15 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        display: inline-block;
    }
    .sidebar-sub {
        font-family: 'JetBrains Mono', monospace;
        font-size: .57rem;
        color: #3d6b3d;
        letter-spacing: .22em;
        margin-top: .3rem;
        text-transform: uppercase;
    }

    /* ── MAIN AREA ───────────────────────────────────────────────────────── */
    [data-testid="stMain"] { position:relative !important; z-index:1 !important; }
    section[data-testid="stMainBlockContainer"] { padding-top:1.6rem !important; }

    /* ── HEADINGS ────────────────────────────────────────────────────────── */
    h1 {
        font-family: 'Syne', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(110deg,
            var(--green-vivid) 0%, var(--green-bright) 35%,
            var(--gold-bright) 75%, var(--gold-vivid) 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        line-height: 1.1 !important;
        margin-bottom: .3rem !important;
    }
    h2 {
        font-family: 'Syne', sans-serif !important;
        color: var(--green-bright) !important;
        font-weight: 700 !important;
        -webkit-text-fill-color: var(--green-bright) !important;
    }
    h3 {
        font-family: 'Syne', sans-serif !important;
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        -webkit-text-fill-color: var(--text-secondary) !important;
    }

    /* ── STATUS PILL ─────────────────────────────────────────────────────── */
    .status-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(74,222,128,0.07);
        border: 1px solid rgba(74,222,128,0.25);
        border-radius: 100px;
        padding: 4px 14px 4px 10px;
        font-size: .63rem;
        color: #4ade80;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: .12em;
        animation: pulse-pill 2.6s ease-in-out infinite;
    }
    .status-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        background: #4ade80;
        display: inline-block;
        box-shadow: 0 0 8px #4ade80;
        flex-shrink: 0;
        animation: pulse-dot 1.6s ease-in-out infinite;
    }

    /* ── CARDS ───────────────────────────────────────────────────────────── */
    .card {
        background: linear-gradient(150deg, #0c1d0c 0%, #101c10 50%, #141d0e 100%);
        border: 1px solid var(--border-green);
        border-top: 2px solid var(--green-vivid);
        border-radius: 14px;
        padding: 1.3rem 1.7rem;
        margin-bottom: 1rem;
        box-shadow: var(--shadow-card);
        position: relative; overflow: hidden;
        transition: box-shadow 0.3s ease, transform 0.2s ease;
        animation: float-in 0.4s ease both;
    }
    .card:hover {
        box-shadow: var(--shadow-card), 0 0 0 1px rgba(74,222,128,0.15);
        transform: translateY(-1px);
    }
    .card::before {
        content: '';
        position: absolute; top:0; left:0; right:0; height:1px;
        background: linear-gradient(90deg,
            transparent 0%, rgba(74,222,128,0.45) 25%,
            rgba(253,224,71,0.3) 65%, transparent 100%);
    }
    .card::after {
        content: '';
        position: absolute; top:0; right:0; width:90px; height:90px;
        background: radial-gradient(circle at top right, rgba(253,224,71,0.06) 0%, transparent 65%);
    }
    .card h2 {
        margin: 0 0 .4rem 0 !important;
        font-size: .65rem !important;
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--text-muted) !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        font-weight: 600 !important;
        -webkit-text-fill-color: var(--text-muted) !important;
    }
    .card p {
        margin: 0;
        font-size: 1.95rem;
        font-weight: 700;
        color: var(--gold-bright);
        font-family: 'Syne', sans-serif;
        letter-spacing: -0.02em;
        -webkit-text-fill-color: var(--gold-bright);
        line-height: 1.1;
    }

    /* ── SEVERITY BADGES ─────────────────────────────────────────────────── */
    .badge-healthy {
        background: linear-gradient(135deg, rgba(5,46,22,0.9), rgba(20,83,45,0.9));
        border: 1px solid rgba(74,222,128,0.45);
        border-radius: 6px; padding: 5px 20px;
        color: #4ade80; font-weight: 600; font-size: .9rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: .1em; text-transform: uppercase;
        box-shadow: 0 0 16px rgba(74,222,128,0.2), 0 2px 10px rgba(0,0,0,0.4);
        display: inline-block;
    }
    .badge-mild {
        background: linear-gradient(135deg, rgba(26,18,0,0.9), rgba(61,44,0,0.9));
        border: 1px solid rgba(253,224,71,0.45);
        border-radius: 6px; padding: 5px 20px;
        color: #fde047; font-weight: 600; font-size: .9rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: .1em; text-transform: uppercase;
        box-shadow: 0 0 16px rgba(253,224,71,0.18), 0 2px 10px rgba(0,0,0,0.4);
        display: inline-block;
    }
    .badge-moderate {
        background: linear-gradient(135deg, rgba(28,15,0,0.9), rgba(67,20,7,0.9));
        border: 1px solid rgba(251,146,60,0.45);
        border-radius: 6px; padding: 5px 20px;
        color: #fb923c; font-weight: 600; font-size: .9rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: .1em; text-transform: uppercase;
        box-shadow: 0 0 16px rgba(251,146,60,0.18), 0 2px 10px rgba(0,0,0,0.4);
        display: inline-block;
    }
    .badge-severe {
        background: linear-gradient(135deg, rgba(31,0,0,0.9), rgba(69,10,10,0.9));
        border: 1px solid rgba(248,113,113,0.45);
        border-radius: 6px; padding: 5px 20px;
        color: #f87171; font-weight: 600; font-size: .9rem;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: .1em; text-transform: uppercase;
        box-shadow: 0 0 16px rgba(248,113,113,0.18), 0 2px 10px rgba(0,0,0,0.4);
        display: inline-block;
    }

    /* ── SECTION TITLE ───────────────────────────────────────────────────── */
    .section-title {
        display: flex; align-items: center; gap: .7rem;
        font-size: .68rem; font-weight: 600;
        color: var(--gold-bright);
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.2em; text-transform: uppercase;
        margin: 2.2rem 0 1rem 0;
    }
    .section-title::before {
        content: '';
        display: inline-block; width: 3px; height: 16px;
        background: linear-gradient(180deg, var(--green-vivid), var(--gold-bright));
        border-radius: 2px; flex-shrink: 0;
    }
    .section-title::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(90deg, var(--border-gold), transparent);
    }

    /* ── METRICS ─────────────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: linear-gradient(150deg, #0c1b0c, #101e10) !important;
        border: 1px solid var(--border-green) !important;
        border-bottom: 2px solid var(--gold-mid) !important;
        border-radius: 12px !important;
        padding: 1rem 1.2rem !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.45), 0 1px 0 rgba(74,222,128,0.05) inset !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(74,222,128,0.3) !important;
        border-bottom-color: var(--gold-bright) !important;
        box-shadow: 0 6px 28px rgba(0,0,0,0.5), var(--glow-green) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .65rem !important;
        color: var(--text-muted) !important;
        letter-spacing: .14em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.95rem !important;
        font-weight: 800 !important;
        color: var(--gold-bright) !important;
        letter-spacing: -0.03em !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .7rem !important;
    }

    /* ── BUTTONS ─────────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #0d2010 0%, #14512c 100%) !important;
        color: var(--green-vivid) !important;
        border: 1px solid rgba(74,222,128,0.3) !important;
        border-radius: 9px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        font-size: .93rem !important;
        letter-spacing: .03em !important;
        padding: .58rem 1.9rem !important;
        box-shadow: 0 0 22px rgba(74,222,128,0.07), 0 2px 10px rgba(0,0,0,0.5) !important;
        transition: all 0.22s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #14512c 0%, #166534 100%) !important;
        border-color: rgba(74,222,128,0.55) !important;
        box-shadow: 0 0 32px rgba(74,222,128,0.16), 0 4px 18px rgba(0,0,0,0.5) !important;
        transform: translateY(-1px) !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a1300 0%, #3c2b00 55%, #6d3c10 100%) !important;
        color: var(--gold-vivid) !important;
        border-color: rgba(253,224,71,0.38) !important;
        box-shadow: 0 0 26px rgba(253,224,71,0.1), 0 2px 10px rgba(0,0,0,0.5) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3c2b00 0%, #6d3c10 55%, #92400e 100%) !important;
        border-color: rgba(253,224,71,0.6) !important;
        box-shadow: 0 0 36px rgba(253,224,71,0.18), 0 4px 18px rgba(0,0,0,0.5) !important;
        transform: translateY(-1px) !important;
        color: #fff !important;
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0a1a0a, #0e2410) !important;
        color: var(--text-secondary) !important;
        border: 1px solid var(--border-green) !important;
        border-radius: 9px !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 600 !important;
        transition: all 0.22s ease !important;
    }
    .stDownloadButton > button:hover {
        color: var(--green-vivid) !important;
        border-color: rgba(74,222,128,0.45) !important;
        box-shadow: 0 0 20px rgba(74,222,128,0.1) !important;
    }

    /* ── FILE UPLOADER ───────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, #0a160a, #0d1c0d) !important;
        border: 1px dashed rgba(74,222,128,0.2) !important;
        border-radius: 12px !important;
        transition: border-color 0.25s, box-shadow 0.25s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(74,222,128,0.38) !important;
        box-shadow: var(--glow-green) !important;
    }
    [data-testid="stFileUploader"] label {
        color: var(--text-secondary) !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: none !important;
    }

    /* ── ALERTS / INFO BOXES ─────────────────────────────────────────────── */
    [data-testid="stAlert"] {
        background: linear-gradient(145deg, #0c180c, #101c10) !important;
        border: 1px solid var(--border-green) !important;
        border-left: 3px solid var(--green-bright) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stAlert"][data-baseweb="notification"][kind="info"] {
        border-left-color: var(--green-bright) !important;
    }
    [data-testid="stAlert"][data-baseweb="notification"][kind="warning"] {
        border-left-color: var(--gold-bright) !important;
        background: linear-gradient(145deg, #160f00, #1c1500) !important;
        border-color: rgba(253,224,71,0.15) !important;
    }
    [data-testid="stAlert"][data-baseweb="notification"][kind="error"] {
        border-left-color: #f87171 !important;
        background: linear-gradient(145deg, #150000, #1c0505) !important;
        border-color: rgba(248,113,113,0.15) !important;
    }
    [data-testid="stAlert"][data-baseweb="notification"][kind="success"] {
        border-left-color: var(--green-vivid) !important;
        background: linear-gradient(145deg, #071a07, #0b1e0b) !important;
    }

    /* ── DATAFRAME / TABLE ───────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-green) !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4) !important;
    }
    [data-testid="stDataFrame"] thead th {
        background: linear-gradient(135deg, #0e1e0e, #131e10) !important;
        color: var(--gold-bright) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .7rem !important;
        letter-spacing: .1em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid var(--border-gold) !important;
    }
    [data-testid="stDataFrame"] tbody tr:nth-child(even) td {
        background: rgba(74,222,128,0.02) !important;
    }
    [data-testid="stDataFrame"] tbody tr:hover td {
        background: rgba(253,224,71,0.04) !important;
    }

    /* ── MULTISELECT ─────────────────────────────────────────────────────── */
    [data-baseweb="tag"] {
        background: linear-gradient(135deg, #0e2210, #14532d) !important;
        border: 1px solid rgba(74,222,128,0.3) !important;
        border-radius: 5px !important;
        color: var(--green-vivid) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── RADIO / SELECT ──────────────────────────────────────────────────── */
    /* Hide default colourful Streamlit radio dot, replace with themed one */
    [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
        border-color: rgba(74,222,128,0.35) !important;
        background: transparent !important;
        width: 14px !important; height: 14px !important;
    }
    [data-testid="stRadio"] [data-baseweb="radio"][aria-checked="true"] > div:first-child {
        border-color: var(--gold-bright) !important;
        background: var(--gold-bright) !important;
        box-shadow: 0 0 8px rgba(253,224,71,0.45) !important;
    }
    [data-testid="stRadio"] [data-baseweb="radio"][aria-checked="true"] > div:first-child > div {
        background: #050c05 !important;
        width: 5px !important; height: 5px !important;
    }
    [data-testid="stRadio"] label {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: .9rem !important;
        transition: color 0.2s !important;
    }
    [data-testid="stRadio"] label:hover { color: var(--gold-bright) !important; }

    /* Sidebar nav section heading — "HOW IT WORKS" bold mono label */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p strong {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .65rem !important;
        letter-spacing: .18em !important;
        text-transform: uppercase !important;
        color: var(--text-muted) !important;
        -webkit-text-fill-color: var(--text-muted) !important;
    }

    /* Sidebar footer — no blue highlight, pure themed colour */
    .sidebar-footer {
        font-family: 'JetBrains Mono', monospace;
        font-size: .57rem;
        color: #2a472a;
        text-align: center;
        letter-spacing: .1em;
        line-height: 2;
        padding: .6rem 0;
        background: transparent !important;
    }
    .sidebar-footer * {
        background: transparent !important;
        color: #2a472a !important;
        text-decoration: none !important;
        -webkit-text-fill-color: #2a472a !important;
    }
    .sidebar-footer-diamond {
        color: #3d6b3d !important;
        -webkit-text-fill-color: #3d6b3d !important;
    }

    /* ── EXPANDER ────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: linear-gradient(145deg, #090f09, #0c150c) !important;
        border: 1px solid var(--border-green) !important;
        border-radius: 10px !important;
        margin-bottom: .5rem !important;
        overflow: hidden !important;
        transition: border-color 0.2s !important;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(74,222,128,0.25) !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: .93rem !important;
        font-weight: 500 !important;
        padding: .65rem 1rem !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--gold-bright) !important;
        background: rgba(253,224,71,0.025) !important;
    }
    [data-testid="stExpander"][open] {
        border-color: var(--border-gold) !important;
    }

    /* ── PROGRESS BAR ────────────────────────────────────────────────────── */
    [data-testid="stProgressBar"] > div {
        background: linear-gradient(90deg,
            var(--green-dark), var(--green-bright) 50%, var(--gold-bright)) !important;
        box-shadow: 0 0 14px rgba(74,222,128,0.3) !important;
        border-radius: 4px !important;
        transition: width 0.3s ease !important;
    }
    [data-testid="stProgressBar"] {
        background: rgba(74,222,128,0.05) !important;
        border: 1px solid rgba(74,222,128,0.08) !important;
        border-radius: 4px !important;
        height: 6px !important;
    }

    /* ── SLIDER ──────────────────────────────────────────────────────────── */
    [data-testid="stSlider"] [role="slider"] {
        background: var(--gold-bright) !important;
        box-shadow: 0 0 12px rgba(253,224,71,0.5) !important;
        border: 2px solid var(--gold-vivid) !important;
    }
    [data-testid="stSlider"] [data-testid="stSlider-track-inner"] {
        background: linear-gradient(90deg, var(--green-dark), var(--gold-mid)) !important;
    }
    [data-testid="stSlider"] p {
        color: var(--text-muted) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .7rem !important;
    }

    /* ── NUMBER INPUT ────────────────────────────────────────────────────── */
    [data-testid="stNumberInput"] input {
        background: #0c1a0c !important;
        border-color: var(--border-green) !important;
        border-radius: 7px !important;
        color: var(--text-primary) !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    [data-testid="stNumberInput"] input:focus {
        border-color: var(--gold-mid) !important;
        box-shadow: 0 0 0 3px rgba(253,224,71,0.1) !important;
    }

    /* ── SPINNER ─────────────────────────────────────────────────────────── */
    [data-testid="stSpinner"] > div {
        border-color: var(--gold-bright) transparent transparent transparent !important;
    }

    /* ── SCROLLBAR ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar       { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--green-dark), var(--gold-deep));
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--green-mid); }

    /* ── MISC ────────────────────────────────────────────────────────────── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border-green) !important;
        margin: 1rem 0 !important;
        opacity: 0.6 !important;
    }
    .stCaption, small, caption {
        color: var(--text-muted) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: .67rem !important;
    }
    #MainMenu, footer, header { visibility: hidden !important; }

    /* Hide sidebar collapse/expand button arrow label */
    [data-testid="collapsedControl"] { display: none !important; }
    button[kind="header"]            { display: none !important; }

    /* Hide the "keyboard_double_arrow_left/right" material icon text
       that appears when the Google Fonts icon font fails to load */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarExpandButton"] {
        display: none !important;
    }
    /* Catch-all for the floating chevron button in newer Streamlit versions */
    section[data-testid="stSidebar"] > div:first-child > div:first-child button {
        display: none !important;
    }
    kbd {
        background: #111f11;
        border: 1px solid var(--border-green);
        border-bottom-width: 2px;
        border-radius: 5px;
        padding: 1px 7px;
        font-family: 'JetBrains Mono', monospace;
        font-size: .78em;
        color: var(--green-vivid);
    }

    /* ── PAGE HEADER STRIP ───────────────────────────────────────────────── */
    .page-header {
        display: flex; align-items: center; gap: .9rem;
        background: linear-gradient(90deg,
            rgba(22,163,74,0.065) 0%, rgba(253,224,71,0.035) 60%, transparent 100%);
        border: 1px solid var(--border-green);
        border-left: 3px solid var(--gold-bright);
        border-radius: 9px;
        padding: .8rem 1.5rem;
        margin-bottom: 1.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: .65rem;
        color: var(--text-muted);
        letter-spacing: .16em;
        text-transform: uppercase;
        position: relative;
        overflow: hidden;
    }
    .page-header::before {
        content: '◆';
        color: var(--green-vivid);
        font-size: .58rem;
        flex-shrink: 0;
    }
    .page-header::after {
        content: '';
        position: absolute; top:0; right:0; bottom:0; width:60px;
        background: linear-gradient(90deg, transparent, rgba(253,224,71,0.02));
    }
    .page-header span { color: var(--gold-vivid); font-weight: 600; }

    /* ── CORNER ACCENT BOX ───────────────────────────────────────────────── */
    .corner-accent {
        position: relative;
        background: linear-gradient(145deg, #09150a, #0d190d);
        border: 1px solid var(--border-green);
        border-radius: 10px;
        padding: 1.15rem 1.5rem;
        margin: .6rem 0;
        transition: border-color 0.25s;
    }
    .corner-accent:hover { border-color: rgba(74,222,128,0.25); }
    .corner-accent::before {
        content: '';
        position: absolute; top:-1px; left:-1px;
        width: 22px; height: 22px;
        border-top: 2px solid var(--green-vivid);
        border-left: 2px solid var(--green-vivid);
        border-radius: 10px 0 0 0;
    }
    .corner-accent::after {
        content: '';
        position: absolute; bottom:-1px; right:-1px;
        width: 22px; height: 22px;
        border-bottom: 2px solid var(--gold-bright);
        border-right: 2px solid var(--gold-bright);
        border-radius: 0 0 10px 0;
    }

    /* ── UPLOAD EMPTY STATE ──────────────────────────────────────────────── */
    .upload-zone {
        border: 1px dashed rgba(74,222,128,0.18);
        border-radius: 14px;
        padding: 3.5rem 2.5rem;
        text-align: center;
        background:
            radial-gradient(ellipse 55% 55% at 50% 50%, rgba(22,163,74,0.04) 0%, transparent 70%),
            linear-gradient(145deg, #070d07, #0a140a);
        color: var(--text-muted);
        font-family: 'Inter', sans-serif;
        transition: border-color 0.25s, box-shadow 0.25s;
    }
    .upload-zone:hover {
        border-color: rgba(253,224,71,0.22);
        box-shadow: var(--glow-gold);
    }

    /* ── GALLERY BADGE ───────────────────────────────────────────────────── */
    .thumb-sev {
        display: inline-block; border-radius: 5px;
        padding: 2px 9px; font-size: .7rem; font-weight: 600;
        font-family: 'JetBrains Mono', monospace; text-transform: uppercase;
    }

    /* ── DIVIDER WITH LABEL ──────────────────────────────────────────────── */
    .divider-label {
        display: flex; align-items: center; gap: .8rem;
        margin: 1.6rem 0;
        color: var(--text-dim);
        font-family: 'JetBrains Mono', monospace;
        font-size: .62rem; letter-spacing: .16em; text-transform: uppercase;
    }
    .divider-label::before, .divider-label::after {
        content: ''; flex: 1; height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-green), transparent);
    }

    </style>
    """, unsafe_allow_html=True)


# ── UI helper functions ───────────────────────────────────────────────────────

def sidebar_header():
    """
    Displays the TerraLeaf logo image (terraleaf_icon.png from root)
    with branding text and animated online pill.
    Falls back to emoji if icon not found.

    KEY RULES:
    - NO st.columns() inside sidebar — breaks markdown rendering
    - NO inline -webkit-background-clip on text — Streamlit sanitiser strips it
    - Use CSS classes (defined in apply_theme) for gradient text
    - Keep st.image() separate from st.markdown() blocks
    """
    icon_path = "terraleaf_icon.png"

    # Logo — plain st.image(), no columns wrapper
    if os.path.exists(icon_path):
        st.image(icon_path, width=80, use_container_width=False)
    else:
        st.markdown(
            "<div style='text-align:center;font-size:3.5rem;padding:.5rem 0'>🌿</div>",
            unsafe_allow_html=True,
        )

    # Brand name uses .sidebar-brand CSS class (gradient defined in apply_theme)
    st.markdown("""
<div class='sidebar-brand-wrap'>
    <div class='sidebar-brand'>TerraLeaf</div>
    <div class='sidebar-sub'>Disease Analyser &middot; v2.0</div>
    <div style='margin-top:.85rem;'>
        <span class='status-pill'>
            <span class='status-dot'></span>
            ONLINE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)


def sidebar_footer():
    """Styled sidebar bottom caption."""
    st.markdown("""
<div class='sidebar-footer'>
    PyTorch &nbsp;&middot;&nbsp; CNN + Regressor<br>
    <span class='sidebar-footer-diamond'>&#9670;</span>&nbsp; Powered by Anthropic
</div>""", unsafe_allow_html=True)


def page_header(mode_label: str, extra: str = ""):
    """Top status-bar strip."""
    extra_html = f"<span style='color:#547a54'>·</span> {extra}" if extra else ""
    st.markdown(f"""
    <div class='page-header'>
        SYSTEM ACTIVE &nbsp;
        <span style='color:#547a54'>·</span>
        &nbsp; MODE: <span>{mode_label}</span>
        &nbsp; {extra_html}
    </div>""", unsafe_allow_html=True)


def section_title(text: str):
    """Styled section label."""
    st.markdown(f"<div class='section-title'>{text}</div>", unsafe_allow_html=True)


def card(title: str, value: str):
    """Dark elevated metric card with gold value text."""
    st.markdown(f"""
    <div class='card'>
        <h2>{title}</h2>
        <p>{value}</p>
    </div>""", unsafe_allow_html=True)


def severity_badge(severity: str):
    """Inline severity badge."""
    st.markdown(
        f"<span class='badge-{severity.lower()}'>{severity}</span>",
        unsafe_allow_html=True
    )


def recommendation_box(text: str, severity: str = ""):
    """Corner-accent recommendation box, border tinted by severity."""
    color = SEV_COLOR.get(severity, "#4ade80")
    st.markdown(f"""
    <div class='corner-accent'
         style='color:#edfaed; font-family:Inter,sans-serif;
                font-size:.97rem; line-height:1.65;
                border-color:{color}20;'>
        {text}
    </div>""", unsafe_allow_html=True)


def upload_empty_state():
    """Empty-state placeholder shown before any file is uploaded."""
    st.markdown("""
    <div class='upload-zone'>
        <div style='
            font-family: Syne, sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: #3d6b3d;
            margin-bottom: .9rem;
            letter-spacing: -.01em;
        '>
            📤 Upload Images or a ZIP Folder to Begin
        </div>
        <p style='line-height:2.1; color:#3d6b3d; font-size:.86rem; margin:0;'>
            <span style='color:#a7d9a7; font-weight:500;'>Select images</span>
            &nbsp;· PNG &nbsp;JPG &nbsp;JPEG &nbsp;WEBP
            &nbsp;|&nbsp; Hold <kbd>Ctrl</kbd> / <kbd>⌘</kbd> for multiple<br>
            <span style='color:#a7d9a7; font-weight:500;'>ZIP folder</span>
            &nbsp;· Zip up your image folder — up to 5,000 images
        </p>
    </div>""", unsafe_allow_html=True)


def divider_label(text: str):
    """Horizontal rule with a centred label."""
    st.markdown(f"<div class='divider-label'>{text}</div>", unsafe_allow_html=True)


def gauge(value, title, mn=0, mx=100, color="#4ade80"):
    """
    Themed Plotly gauge with green → gold gradient steps.
    Usage: st.plotly_chart(ui.gauge(value, "TITLE"), use_container_width=True)
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={
            "text": title,
            "font": {"size": 10, "color": "#547a54", "family": "JetBrains Mono"},
        },
        gauge={
            "axis": {
                "range": [mn, mx],
                "tickcolor": "#2d4a2d",
                "tickfont": {"color": "#2d4a2d", "size": 8},
                "tickwidth": 1,
            },
            "bar": {"color": color, "thickness": 0.2},
            "bgcolor": "#070d07",
            "bordercolor": "rgba(74,222,128,0.1)",
            "borderwidth": 1,
            "steps": [
                {"range": [mn,                   mn+(mx-mn)*0.33], "color": "rgba(5,46,22,0.5)"},
                {"range": [mn+(mx-mn)*0.33, mn+(mx-mn)*0.66],      "color": "rgba(22,101,52,0.35)"},
                {"range": [mn+(mx-mn)*0.66, mx],                   "color": "rgba(113,63,18,0.3)"},
            ],
            "threshold": {
                "line": {"color": "#fde047", "width": 2},
                "thickness": 0.7,
                "value": value,
            },
        },
        number={
            "font": {"size": 20, "color": "#facc15", "family": "Syne"},
            "suffix": "",
        },
    ))
    fig.update_layout(
        height=205,
        margin=dict(t=40, b=4, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#edfaed"},
    )
    return fig