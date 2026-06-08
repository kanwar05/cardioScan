import streamlit as st
import pandas as pd
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioScan · Heart Risk Predictor",
    page_icon="🫀",
    layout="centered",
)

# ── Theme toggle (stored in session state) ───────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── CSS injection ─────────────────────────────────────────────────────────────
def inject_css(dark: bool):
    if dark:
        bg          = "#0d0f14"
        surface     = "#161a23"
        card        = "#1e2330"
        border      = "#2a3040"
        text_primary = "#eef0f6"
        text_secondary = "#8b93a8"
        accent      = "#e05260"
        accent_glow = "rgba(224,82,96,0.25)"
        success_bg  = "#0f2318"
        success_text = "#4cde8a"
        error_bg    = "#2a1218"
        error_text  = "#f07080"
        input_bg    = "#1e2330"
        slider_track = "#2a3040"
        toggle_bg   = "#2a3040"
        toggle_knob = "#8b93a8"
    else:
        bg          = "#f4f6fb"
        surface     = "#ffffff"
        card        = "#ffffff"
        border      = "#dde2ee"
        text_primary = "#1a1e2e"
        text_secondary = "#6b7592"
        accent      = "#d63a4a"
        accent_glow = "rgba(214,58,74,0.18)"
        success_bg  = "#edfaf4"
        success_text = "#1a7a4a"
        error_bg    = "#fdeef0"
        error_text  = "#c0303d"
        input_bg    = "#f9fafc"
        slider_track = "#dde2ee"
        toggle_bg   = "#dde2ee"
        toggle_knob = "#ffffff"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: {text_primary};
    }}
    .stApp {{
        background: {bg};
    }}
    .block-container {{
        max-width: 720px;
        padding: 2rem 1.5rem 4rem;
    }}

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ── Top bar ── */
    .topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2.5rem;
    }}
    .brand {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}
    .brand-icon {{
        font-size: 1.6rem;
        line-height: 1;
    }}
    .brand-name {{
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 1.4rem;
        color: {text_primary};
        letter-spacing: -0.5px;
    }}
    .brand-dot {{
        color: {accent};
    }}

    /* ── Hero section ── */
    .hero {{
        text-align: center;
        margin-bottom: 2.5rem;
    }}
    .hero-badge {{
        display: inline-block;
        background: {accent_glow};
        color: {accent};
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 0.35rem 0.9rem;
        border-radius: 100px;
        margin-bottom: 1rem;
        border: 1px solid {accent};
    }}
    .hero-title {{
        font-family: 'Syne', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.15;
        color: {text_primary};
        margin-bottom: 0.75rem;
        letter-spacing: -1px;
    }}
    .hero-title span {{
        color: {accent};
    }}
    .hero-sub {{
        color: {text_secondary};
        font-size: 0.95rem;
        font-weight: 300;
        max-width: 440px;
        margin: 0 auto;
        line-height: 1.6;
    }}

    /* ── Section headers ── */
    .section-label {{
        font-family: 'Syne', sans-serif;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: {accent};
        margin: 2rem 0 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    .section-label::after {{
        content: '';
        flex: 1;
        height: 1px;
        background: {border};
    }}

    /* ── Cards ── */
    .param-card {{
        background: {card};
        border: 1px solid {border};
        border-radius: 16px;
        padding: 1.5rem 1.5rem 0.75rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}

    /* ── Streamlit widget overrides ── */
    .stSlider > div > div > div > div {{
        background: {accent} !important;
    }}
    .stSlider > div > div > div {{
        background: {slider_track} !important;
    }}

    div[data-baseweb="select"] > div {{
        background: {input_bg} !important;
        border-color: {border} !important;
        border-radius: 10px !important;
        color: {text_primary} !important;
    }}
    div[data-baseweb="select"] > div:focus-within {{
        border-color: {accent} !important;
        box-shadow: 0 0 0 3px {accent_glow} !important;
    }}

    div[data-baseweb="input"] > div {{
        background: {input_bg} !important;
        border-color: {border} !important;
        border-radius: 10px !important;
        color: {text_primary} !important;
    }}
    div[data-baseweb="input"] > div:focus-within {{
        border-color: {accent} !important;
        box-shadow: 0 0 0 3px {accent_glow} !important;
    }}

    input, textarea, select {{
        color: {text_primary} !important;
        background: {input_bg} !important;
    }}

    label, .stSlider label, .stSelectbox label, .stNumberInput label {{
        color: {text_secondary} !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.2px;
    }}

    /* ── Predict button ── */
    .stButton > button {{
        width: 100%;
        background: {accent} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 2rem !important;
        font-family: 'Syne', sans-serif !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 20px {accent_glow} !important;
        margin-top: 0.5rem;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px {accent_glow} !important;
        filter: brightness(1.08);
    }}
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* ── Result cards ── */
    .result-card {{
        border-radius: 16px;
        padding: 1.75rem 2rem;
        margin-top: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1.25rem;
        animation: slideUp 0.4s ease;
    }}
    .result-card.danger {{
        background: {error_bg};
        border: 1.5px solid {error_text};
    }}
    .result-card.safe {{
        background: {success_bg};
        border: 1.5px solid {success_text};
    }}
    .result-icon {{
        font-size: 2.5rem;
        line-height: 1;
        flex-shrink: 0;
    }}
    .result-title {{
        font-family: 'Syne', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }}
    .result-card.danger .result-title {{ color: {error_text}; }}
    .result-card.safe .result-title {{ color: {success_text}; }}
    .result-body {{
        font-size: 0.85rem;
        color: {text_secondary};
        line-height: 1.5;
    }}

    /* ── Divider ── */
    .divider {{
        height: 1px;
        background: {border};
        margin: 2rem 0;
    }}

    /* ── Footer ── */
    .footer {{
        text-align: center;
        color: {text_secondary};
        font-size: 0.78rem;
        margin-top: 3rem;
        line-height: 1.8;
    }}
    .footer span {{ color: {accent}; }}

    /* ── Animation ── */
    @keyframes slideUp {{
        from {{ opacity: 0; transform: translateY(16px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ── Metric chips ── */
    .chip-row {{
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 1rem;
    }}
    .chip {{
        background: {accent_glow};
        color: {accent};
        border: 1px solid {accent};
        border-radius: 100px;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css(st.session_state.dark_mode)

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("KNN_heart_model.pkl")
    scaler   = joblib.load("heart_scaler.pkl")
    columns  = joblib.load("heart_columns.pkl")
    return model, scaler, columns

model, scaler, expected_columns = load_artifacts()

# ── Top bar with theme toggle ─────────────────────────────────────────────────
col_brand, col_toggle = st.columns([5, 1])
with col_brand:
    st.markdown("""
    <div class="brand">
        <span class="brand-icon">🫀</span>
        <span class="brand-name">Cardio<span class="brand-dot">Scan</span></span>
    </div>
    """, unsafe_allow_html=True)
with col_toggle:
    icon = "☀️" if st.session_state.dark_mode else "🌙"
    if st.button(icon, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered · KNN Model</div>
    <div class="hero-title">Know Your <span>Heart Risk</span></div>
    <p class="hero-sub">
        Fill in your clinical parameters below for an instant cardiovascular disease risk assessment.
    </p>
</div>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Personal Info</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 18, 100, 40)
with c2:
    sex = st.selectbox("Biological Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")

st.markdown('<div class="section-label">Vital Signs</div>', unsafe_allow_html=True)

c3, c4 = st.columns(2)
with c3:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
with c4:
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

c5, c6 = st.columns(2)
with c5:
    max_hr = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
with c6:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.markdown('<div class="section-label">Cardiac Assessment</div>', unsafe_allow_html=True)

c7, c8 = st.columns(2)
with c7:
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"],
        help="ATA: Atypical Angina · NAP: Non-Anginal · TA: Typical Angina · ASY: Asymptomatic")
with c8:
    resting_ecg = st.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])

c9, c10 = st.columns(2)
with c9:
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"], format_func=lambda x: "Yes" if x == "Y" else "No")
with c10:
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

oldpeak = st.slider("Oldpeak — ST Depression (mm)", 0.0, 6.0, 1.0, step=0.1)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🫀  Run Risk Assessment"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.markdown("""
        <div class="result-card danger">
            <div class="result-icon">⚠️</div>
            <div>
                <div class="result-title">Elevated Cardiovascular Risk Detected</div>
                <div class="result-body">
                    Your parameters suggest a higher likelihood of heart disease. 
                    Please consult a cardiologist for a thorough clinical evaluation.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card safe">
            <div class="result-icon">✅</div>
            <div>
                <div class="result-title">Low Cardiovascular Risk</div>
                <div class="result-body">
                    Your current parameters indicate a lower risk profile. 
                    Maintain a heart-healthy lifestyle and schedule regular check-ups.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <span>Kanwar</span> · CardioScan v2.0<br>
    ⚕️ This tool is for informational purposes only and does not replace medical advice.
</div>
""", unsafe_allow_html=True)

