import streamlit as st
import numpy as np
import joblib
import pandas as pd

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="centered"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}

.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f7971e, #ffd200, #f7971e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.hero p {
    color: #b0aecb;
    font-size: 1.05rem;
    font-weight: 300;
}

.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(10px);
}

.section-title {
    color: #ffd200;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.result-box {
    background: linear-gradient(135deg, #f7971e22, #ffd20022);
    border: 1px solid #ffd20055;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-value {
    font-size: 3rem;
    font-weight: 800;
    color: #ffd200;
    line-height: 1;
}

.result-label {
    color: #b0aecb;
    font-size: 0.95rem;
    margin-top: 0.5rem;
}

.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

.metric-card {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}

.metric-val {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f7971e;
}

.metric-lbl {
    font-size: 0.75rem;
    color: #8884a8;
    margin-top: 0.2rem;
}

div[data-testid="stSlider"] > label,
div[data-testid="stNumberInput"] > label,
.stSelectbox label {
    color: #c8c6e0 !important;
    font-weight: 400;
    font-size: 0.9rem;
}

.stButton > button {
    background: linear-gradient(90deg, #f7971e, #ffd200);
    color: #1a1833;
    border: none;
    border-radius: 50px;
    padding: 0.75rem 2.5rem;
    font-size: 1rem;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    width: 100%;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    letter-spacing: 0.04em;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px #ffd20044;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Scaler ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model files: {e}\n\nMake sure `model.pkl` and `scaler.pkl` are in the same folder as `app.py`.")

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>☀️ Solar Power Predictor</h1>
    <p>XGBoost-powered prediction of solar energy output (Joules per 3-hour period)</p>
</div>
""", unsafe_allow_html=True)

# ─── Input Form ───────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌍 Environmental Conditions</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    distance_to_solar_noon = st.slider(
        "Distance to Solar Noon (radians)",
        min_value=0.0, max_value=1.6, value=0.3, step=0.01,
        help="0 = exactly noon, ~1.57 = sunrise/sunset"
    )
    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0, max_value=60.0, value=25.0, step=0.5
    )
    sky_cover = st.selectbox(
        "Sky Cover",
        options=[0, 1, 2, 3, 4],
        index=0,
        format_func=lambda x: {0:"0 — Clear ☀️", 1:"1 — Mostly Clear 🌤️", 2:"2 — Partly Cloudy ⛅", 3:"3 — Mostly Cloudy 🌥️", 4:"4 — Overcast ☁️"}[x]
    )

with col2:
    visibility = st.slider(
        "Visibility (km)",
        min_value=0.0, max_value=20.0, value=10.0, step=0.5
    )
    humidity = st.slider(
        "Humidity (%)",
        min_value=0, max_value=100, value=45, step=1
    )
    wind_speed = st.slider(
        "Wind Speed (m/s)",
        min_value=0.0, max_value=20.0, value=3.5, step=0.1
    )

st.markdown('</div>', unsafe_allow_html=True)

# ─── Predict ──────────────────────────────────────────────────────────────────
predict_clicked = st.button("⚡ Predict Power Output")

if predict_clicked and model_loaded:
    input_df = pd.DataFrame([{
        'distance_to_solar_noon': distance_to_solar_noon,
        'temperature': temperature,
        'sky_cover': sky_cover,
        'visibility': visibility,
        'humidity': humidity,
        'wind_speed': wind_speed
    }])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Convert to kWh equivalent for context
    kwh = prediction / 3_600_000

    # Determine solar condition label
    if sky_cover == 0 and distance_to_solar_noon < 0.3:
        condition = "🌟 Ideal Solar Conditions"
        color_hint = "#ffd200"
    elif sky_cover >= 3 or distance_to_solar_noon > 1.2:
        condition = "🌑 Low Solar Output Expected"
        color_hint = "#8884a8"
    else:
        condition = "⛅ Moderate Solar Conditions"
        color_hint = "#f7971e"

    st.markdown(f"""
    <div class="result-box">
        <div style="color:{color_hint}; font-size:0.9rem; font-weight:600; letter-spacing:0.1em; margin-bottom:0.8rem; text-transform:uppercase;">
            {condition}
        </div>
        <div class="result-value">{prediction:,.0f}</div>
        <div class="result-label">Joules generated per 3-hour period</div>
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-val">{kwh:.4f}</div>
                <div class="metric-lbl">kWh Equivalent</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{prediction/1000:,.1f}</div>
                <div class="metric-lbl">Kilojoules</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{kwh * 8:.4f}</div>
                <div class="metric-lbl">kWh / Day (est.)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Show input summary
    with st.expander("📋 View Input Summary"):
        st.dataframe(input_df.style.format({
            'distance_to_solar_noon': '{:.3f}',
            'temperature': '{:.1f}°C',
            'visibility': '{:.1f} km',
            'humidity': '{:.0f}%',
            'wind_speed': '{:.1f} m/s'
        }), use_container_width=True)

elif predict_clicked and not model_loaded:
    st.error("Model files not found. Please check that `model.pkl` and `scaler.pkl` are in the same directory.")

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:2.5rem; padding:1.2rem;
     border-top:1px solid rgba(255,255,255,0.08); color:#6b6890; font-size:0.85rem;">
    Built with ❤️ by <span style="color:#ffd200; font-weight:600;">Tejes Raju Ingole</span>
</div>
""", unsafe_allow_html=True)

# ─── Feature Guide ────────────────────────────────────────────────────────────
with st.expander("📖 Feature Reference Guide"):
    st.markdown("""
    | Feature | Range | Effect on Power |
    |---|---|---|
    | **Distance to Solar Noon** | 0 – 1.57 rad | ↓ closer to noon = ↑ power |
    | **Temperature** | °C | ↑ temp (clear sky) = ↑ power |
    | **Sky Cover** | 0–4 scale | ↑ cloud cover = ↓ power |
    | **Visibility** | km | ↑ visibility = ↑ power |
    | **Humidity** | % | ↑ humidity = ↓ power |
    | **Wind Speed** | m/s | weak positive effect |
    """)
    st.markdown("**Model:** XGBoost Regressor (tuned) — R² ≈ 0.90 | MAE ≈ 1600 J | RMSE ≈ 3300 J")