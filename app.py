
import streamlit as st

import pandas as pd
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Solar Power Prediction", page_icon="🔆", layout="wide")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
/* Dark grey background */
[data-testid="stAppViewContainer"] {
    background: #1e1e1e;
    color: #f2f2f2;
    font-family: 'Trebuchet MS', sans-serif;
}

/* Patterned sidebar */
[data-testid="stSidebar"] {
  background:
    repeating-conic-gradient(
        from 30deg,
        #0000 90deg 120deg,
        #3c3c3c 0deg 180deg
      )
      calc(0.5 * 200px) calc(0.5 * 200px * 0.577),
    repeating-conic-gradient(
      from 30deg,
      #1d1d1d 0deg 60deg,
      #4e4f51 0deg 120deg,
      #3c3c3c 0deg 180deg
    );
  background-size: 200px calc(200px * 0.577);
  color: #f2f2f2;
}

/* Headings */
h1, h2, h3 {
  text-align: center;
  color: #ffcc00;
}

/* Loader */
.loader {
  position: relative;
  width: 150px;
  height: 150px;
  background: transparent;
  border-radius: 50%;
  box-shadow: 25px 25px 75px rgba(0, 0, 0, 0.55);
  border: 1px solid #333;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  margin: 25px auto;
}
.loader::before {
  content: '';
  position: absolute;
  inset: 20px;
  background: transparent;
  border: 1px dashed #444;
  border-radius: 50%;
  box-shadow: inset -5px -5px 25px rgba(0, 0, 0, 0.25),
              inset 5px 5px 35px rgba(0, 0, 0, 0.25);
}
.loader::after {
  content: '';
  position: absolute;
  width: 50px;
  height: 50px;
  border-radius: 50%;
  border: 1px dashed #444;
  box-shadow: inset -5px -5px 25px rgba(0, 0, 0, 0.25),
              inset 5px 5px 35px rgba(0, 0, 0, 0.25);
}
.loader span {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 50%;
  height: 100%;
  background: transparent;
  transform-origin: top left;
  animation: radar81 2s linear infinite;
  border-top: 1px dashed #fff;
}
.loader span::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: #fc5185;
  transform-origin: top left;
  transform: rotate(-55deg);
  filter: blur(30px) drop-shadow(20px 20px 20px #fc5185);
}
@keyframes radar81 {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Results text */
.result {
  text-align: center;
  font-size: 1.2rem;
  color: #00ff88;
  margin-top: 15px;
}

/* Chart container */
.chart-container {
  display: flex;
  justify-content: center;
  align-items: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL & DATA ------------------
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
df = pd.read_csv("solarpowergeneration.csv")
power_col = [col for col in df.columns if "power" in col.lower()][0]
max_power = df[power_col].max()
avg_power = df[power_col].mean()

# ------------------ HEADER ------------------
st.markdown("<h1>🔆 Solar Power Generation Predictor</h1>", unsafe_allow_html=True)
st.write("""
    Solar energy generation is the process of converting **sunlight into usable electrical power** using photovoltaic (PV) panels.
    These panels absorb sunlight and create energy through the **photovoltaic effect**.
    This project uses **Machine Learning (XGBoost)** to predict solar power generation
    based on environmental factors like temperature, humidity, sky cover, and visibility.

    ---
    ### ☀️ From Sunlight to Electricity: The Journey of Solar Energy
    - **Sunlight:** Radiant energy from the sun  
    - **Solar Panels:** Convert sunlight into direct current (DC) electricity  
    - **Inverter:** Converts DC into alternating current (AC) usable at homes  
    - **Electricity:** Powers household and industrial appliances

    ---
    """)
st.markdown(
    "<h2 style='text-align: center;'> Let’s Predict How Much Solar Energy Will Be Generated!</h2>",
    unsafe_allow_html=True
)
st.write("<p style='text-align:center;'>Predict solar energy output (in Joules per 3-hour period) using environmental conditions.</p>", unsafe_allow_html=True)

# ------------------ METRICS ------------------
col1, col2, col3 = st.columns(3)
col1.metric("⚡ Highest Power", f"{max_power:.2f} J")
col2.metric("🌤 Avg Power", f"{avg_power:.2f} J")
col3.metric("🤖 Model", "XGBoost")

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("🌍 Enter Environmental Data")

distance = st.sidebar.number_input("☀️ Distance to Solar Noon", 0.0, 1.0, 0.1)
temperature = st.sidebar.number_input("🌡 Temperature (°C)", -20.0, 60.0, 25.0)

# 🌥 Sky Cover dropdown
sky_options = {
    "☀️ 0 → Clear Sky": 0,
    "🌤️ 1 → Few Clouds": 1,
    "⛅ 2 → Scattered Clouds": 2,
    "🌥️ 3 → Broken Clouds": 3,
    "☁️ 4 → Overcast": 4
}
sky_label = st.sidebar.selectbox("🌥 Sky Cover Condition", list(sky_options.keys()))
sky_cover = sky_options[sky_label]

# 👁 Visibility slider with fine steps
visibility = round(st.sidebar.slider("👁 Visibility (km)", 0.0, 10.0, 5.0, step=0.25), 2)

humidity = st.sidebar.number_input("💧 Humidity (%)", 0.0, 100.0, 40.0)
wind_speed = st.sidebar.number_input("💨 Wind Speed (m/s)", 0.0, 50.0, 5.0)

# ------------------ PREDICTION ------------------
if st.button("🚀 Predict Power Generation"):
    st.markdown("<div class='loader'><span></span></div>", unsafe_allow_html=True)
    time.sleep(2)

    # Prepare and predict
    data = pd.DataFrame({
        'distance_to_solar_noon': [distance],
        'temperature': [temperature],
        'sky_cover': [sky_cover],
        'visibility': [visibility],
        'humidity': [humidity],
        'wind_speed': [wind_speed]
    })
    scaled = scaler.transform(data)
    prediction = model.predict(scaled)[0]

    # Display result
    st.markdown(f"<div class='result'>🔋 Predicted Solar Power Generated: {prediction:.2f} J (per 3-hour period)</div>", unsafe_allow_html=True)

    # ------------------ BEAUTIFUL LINE CHART ------------------
    chart_df = pd.DataFrame({
        "Label": ["Predicted Power", "Highest Recorded Power"],
        "Value": [prediction, max_power]
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(chart_df["Label"], chart_df["Value"], marker='o', linewidth=4, color="#fc5185", alpha=0.9)
    ax.fill_between(chart_df["Label"], chart_df["Value"], color="#fc518540")
    ax.set_facecolor("#2b2b2b")
    fig.patch.set_facecolor("#2b2b2b")
    ax.set_title("⚡ Solar Power Comparison", color="#ffcc00", fontsize=14)
    ax.set_ylabel("Power (Joules)", color="#f2f2f2")
    ax.tick_params(colors='#f2f2f2')
    ax.grid(alpha=0.3, color="#888")
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.pyplot(fig, use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.write("🔆 **Add your input values for the variables in the sidebar to start the prediction!**")
st.write("🔆 **Click \"Predict Power Generation\" to explore the real-time prediction model!**")

st.markdown("<hr><p style='text-align:center;'>Developed by Tejes Raju Ingole | Futuristic Radar UI 🌌 | Powered by Streamlit & XGBoost ⚡</p>", unsafe_allow_html=True)

