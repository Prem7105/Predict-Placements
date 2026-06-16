import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# ── Page Config (must be first) ──
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load Model & Data ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(BASE_DIR, 'placement.csv'))
    if 'Unnamed' in df.columns[0]:
        df = df.iloc[:, 1:]
    return df

try:
    clf, scaler = load_model()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"File missing: {e}")
    st.stop()

total = len(df)
placed = int(df['placement'].sum())
rate = (placed / total) * 100
avg_cgpa_placed = df[df['placement'] == 1]['cgpa'].mean()

# ── CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
    --bg: #FFFFFF;
    --fg: #0a0a0a;
    --muted: #F5F5F5;
    --accent: #FF3000;
    --gray: #999;
    --font: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background: var(--bg) !important;
    font-family: var(--font) !important;
}
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    pointer-events: none; z-index: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.012'/%3E%3C/svg%3E");
    background-size: 256px;
}
.stApp h1,.stApp h2,.stApp h3,.stApp p,.stApp span,.stApp label,.stApp div {
    color: var(--fg) !important;
    font-family: var(--font) !important;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: var(--bg) !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container {
    max-width: 1000px !important;
    padding: 0.5rem 2rem 2rem 2rem !important;
}

/* ── Hero ── */
.hero {
    padding: 1.5rem 0 1rem 0;
    border-bottom: 3px solid var(--fg);
    margin-bottom: 0;
    display: flex;
    align-items: baseline;
    gap: 1.2rem;
}
.hero-eyebrow {
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--accent) !important;
    white-space: nowrap;
}
.hero h1 {
    font-size: clamp(1.8rem, 4vw, 2.8rem) !important;
    font-weight: 900 !important;
    line-height: 1 !important;
    letter-spacing: -1.5px !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}
.hero h1 .red { color: var(--accent) !important; }

/* ── Stats Strip ── */
.stats-strip {
    display: flex;
    border-bottom: 3px solid var(--fg);
}
.stats-strip .s-item {
    flex: 1;
    padding: 0.7rem 1.2rem;
    border-right: 1px solid #ddd;
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
}
.stats-strip .s-item:last-child { border-right: none; }
.stats-strip .s-val {
    font-size: 1.1rem;
    font-weight: 900;
    color: var(--fg) !important;
    letter-spacing: -0.5px;
}
.stats-strip .s-lbl {
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--gray) !important;
}

/* ── Section Header ── */
.sec-head {
    padding: 1rem 0 0.8rem 0;
}
.sec-num {
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-bottom: 0.2rem;
}
.sec-title {
    font-size: 1.4rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: -0.5px;
    line-height: 1;
    color: var(--fg) !important;
}

/* ── Input Container (st.container border override) ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 3px solid var(--fg) !important;
    border-radius: 0 !important;
    transition: border-color 0.15s ease-out;
    overflow: hidden;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--accent) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 1.2rem 1.5rem 0.8rem 1.5rem !important;
    gap: 0.2rem !important;
}
.ib-label {
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin-bottom: 0.1rem;
}
.ib-title {
    font-size: 1.1rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: -0.3px;
    color: var(--fg) !important;
    margin-bottom: 0;
}
.ib-range {
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--gray) !important;
    margin-bottom: 0.5rem;
}

/* Slider styling */
div[data-testid="stSlider"] {
    padding: 0 !important;
    margin-top: 0.3rem !important;
}
div[data-testid="stSlider"] label {
    display: none !important;
}
div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    font-size: 1.6rem !important;
    font-weight: 900 !important;
    color: var(--fg) !important;
    font-family: var(--font) !important;
    letter-spacing: -0.5px !important;
}
.stSlider > div > div > div > div {
    background-color: var(--accent) !important;
}
.stSlider > div > div > div {
    background-color: #e0e0e0 !important;
}

/* ── Predict Button ── */
.stButton > button {
    background-color: var(--accent) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 0px !important;
    padding: 1.4rem 3rem !important;
    font-size: 0.9rem !important;
    font-weight: 800 !important;
    font-family: var(--font) !important;
    letter-spacing: 5px !important;
    text-transform: uppercase !important;
    transition: all 0.15s ease-out !important;
    width: 100% !important;
    box-shadow: none !important;
    outline: none !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button:hover {
    background-color: var(--fg) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    transform: translateY(-2px) !important;
}
.stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
/* Force white text in all states */
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* ── Result ── */
.result-box {
    padding: 3rem;
    margin-top: 0;
    animation: fadeUp 0.35s ease-out;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-box.positive {
    background: var(--fg);
}
.result-box.negative {
    background: var(--accent);
}
.result-box .r-eyebrow {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 5px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.result-box.positive .r-eyebrow { color: var(--accent) !important; }
.result-box.negative .r-eyebrow { color: rgba(255,255,255,0.6) !important; }

.result-box .r-verdict {
    font-size: clamp(2.5rem, 6vw, 4.5rem);
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: -2px;
    line-height: 0.95;
    margin-bottom: 1.2rem;
    color: #FFFFFF !important;
}
.result-box .r-detail {
    font-size: 1rem;
    font-weight: 400;
    line-height: 1.7;
    color: rgba(255,255,255,0.65) !important;
    max-width: 550px;
}
.result-box .r-detail strong {
    color: #FFFFFF !important;
    font-weight: 700;
}

/* ── Confidence ── */
.conf-bar {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.5rem 2rem;
    border: 3px solid var(--fg);
    margin-top: 0;
    background: var(--bg);
}
.conf-bar .c-label {
    font-size: 0.55rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--gray) !important;
    white-space: nowrap;
}
.conf-bar .c-track {
    flex: 1;
    height: 8px;
    background: var(--muted);
}
.conf-bar .c-fill {
    height: 100%;
    transition: width 0.6s ease-out;
}
.conf-bar .c-fill.pos { background: var(--fg); }
.conf-bar .c-fill.neg { background: var(--accent); }
.conf-bar .c-num {
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: -1px;
    min-width: 85px;
    text-align: right;
}
.conf-bar .c-num.pos { color: var(--fg) !important; }
.conf-bar .c-num.neg { color: var(--accent) !important; }

/* ── Divider ── */
.divider { border: none; border-top: 3px solid var(--fg); margin: 2.5rem 0 0 0; }

/* ── Responsive ── */
@media (max-width: 768px) {
    .hero h1 { letter-spacing: -2px !important; }
    .stats-strip { flex-wrap: wrap; }
    .stats-strip .s-item { flex: 1 1 50%; }
    .conf-bar { flex-wrap: wrap; }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>PLACEMENT <span class="red">PREDICTOR.</span></h1>
    <div class="hero-eyebrow">ML-Powered</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  STATS STRIP
# ═══════════════════════════════════════════
st.markdown(f"""
<div class="stats-strip">
    <div class="s-item">
        <span class="s-val">{total}</span>
        <span class="s-lbl">Students</span>
    </div>
    <div class="s-item">
        <span class="s-val">{rate:.0f}%</span>
        <span class="s-lbl">Placed</span>
    </div>
    <div class="s-item">
        <span class="s-val">{avg_cgpa_placed:.1f}</span>
        <span class="s-lbl">Avg CGPA (Placed)</span>
    </div>
    <div class="s-item">
        <span class="s-val">LR</span>
        <span class="s-lbl">Model</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  INPUTS
# ═══════════════════════════════════════════
st.markdown("""
<div class="sec-head">
    <div class="sec-num">01 — Input</div>
    <div class="sec-title">ENTER YOUR DATA.</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")

with col1:
    with st.container(border=True):
        st.markdown('<div class="ib-label">Parameter 01</div><div class="ib-title">CGPA Score</div><div class="ib-range">Range: 0.0 — 10.0</div>', unsafe_allow_html=True)
        cgpa = st.slider("cgpa", min_value=0.0, max_value=10.0, value=7.0, step=0.1, label_visibility="collapsed")

with col2:
    with st.container(border=True):
        st.markdown('<div class="ib-label">Parameter 02</div><div class="ib-title">IQ Score</div><div class="ib-range">Range: 50 — 200</div>', unsafe_allow_html=True)
        iq = st.slider("iq", min_value=50, max_value=200, value=110, step=1, label_visibility="collapsed")


# ═══════════════════════════════════════════
#  BUTTON
# ═══════════════════════════════════════════
st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1.5, 2, 1.5])
with btn_col:
    predict = st.button("PREDICT NOW", use_container_width=True)


# ═══════════════════════════════════════════
#  RESULT
# ═══════════════════════════════════════════
if predict:
    q = scaler.transform(np.array([[cgpa, iq]]))
    pred = clf.predict(q)[0]
    prob = clf.predict_proba(q)[0]

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="sec-head" style="padding-top:0">
        <div class="sec-num">02 — Result</div>
        <div class="sec-title">PREDICTION.</div>
    </div>
    """, unsafe_allow_html=True)

    if pred == 1:
        st.markdown(f"""
        <div class="result-box positive">
            <div class="r-eyebrow">Positive Outcome</div>
            <div class="r-verdict">YOU WILL<br>GET PLACED.</div>
            <div class="r-detail">
                CGPA <strong>{cgpa}</strong> &middot; IQ <strong>{iq}</strong> — your profile
                matches historically placed candidates. Keep building.
            </div>
        </div>
        <div class="conf-bar">
            <span class="c-label">Confidence</span>
            <div class="c-track"><div class="c-fill pos" style="width:{prob[1]*100:.0f}%"></div></div>
            <span class="c-num pos">{prob[1]*100:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box negative">
            <div class="r-eyebrow">Needs Improvement</div>
            <div class="r-verdict">NOT YET.<br>KEEP PUSHING.</div>
            <div class="r-detail">
                CGPA <strong>{cgpa}</strong> &middot; IQ <strong>{iq}</strong> — placed students
                average <strong>{avg_cgpa_placed:.1f}</strong> CGPA. Close the gap.
            </div>
        </div>
        <div class="conf-bar">
            <span class="c-label">Risk Level</span>
            <div class="c-track"><div class="c-fill neg" style="width:{prob[0]*100:.0f}%"></div></div>
            <span class="c-num neg">{prob[0]*100:.0f}%</span>
        </div>
        """, unsafe_allow_html=True)
