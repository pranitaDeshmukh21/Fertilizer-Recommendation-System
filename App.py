# Front End Using Streamlit.

import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# ================= LOAD MODEL & ENCODERS =================
model = joblib.load("fertilizer_model.pkl")
soil_encoder = joblib.load("soil_encoder.pkl")
crop_encoder = joblib.load("crop_encoder.pkl")
fertilizer_encoder = joblib.load("fertilizer_encoder.pkl")

# 🔐 EXACT feature names from trained model
FEATURE_NAMES = list(model.feature_names_in_)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
body { background-color: #f4f9f4; }
.main {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
}
h1 { color: #2e7d32; text-align: center; }
.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-size: 18px;
}
.stButton>button:hover { background-color: #1b5e20; }
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.title("🌾 Fertilizer Recommendation System")
st.write("Provide soil and environmental details to get the best fertilizer recommendation.")
st.divider()

# ================= INPUT FIELDS =================
col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("🌡 Temperature (°C)", 0, 50, 25)
    humidity = st.slider("💧 Humidity (%)", 0, 100, 50)
    moisture = st.slider("🌱 Soil Moisture (%)", 0, 100, 40)
    nitrogen = st.number_input("🧪 Nitrogen (N)", 0, 200, 90)

with col2:
    potassium = st.number_input("🧪 Potassium (K)", 0, 200, 45)
    phosphorous = st.number_input("🧪 Phosphorous (P)", 0, 200, 38)
    soil_type = st.selectbox("🪨 Soil Type", soil_encoder.classes_)
    crop_type = st.selectbox("🌿 Crop Type", crop_encoder.classes_)

# ================= PREDICTION =================
st.divider()

if st.button("🔍 Predict Fertilizer"):
    try:
        soil_encoded = soil_encoder.transform([soil_type])[0]
        crop_encoded = crop_encoder.transform([crop_type])[0]

        # Build input strictly using model feature order
        values = [
            temperature,
            humidity,
            moisture,
            soil_encoded,
            crop_encoded,
            nitrogen,
            potassium,
            phosphorous
        ]

        input_data = pd.DataFrame([values], columns=FEATURE_NAMES)

        prediction = model.predict(input_data)
        fertilizer = fertilizer_encoder.inverse_transform(prediction)[0]

        st.success(f"✅ Recommended Fertilizer: **{fertilizer}**")

    except Exception as e:
        st.error("⚠️ Error during prediction")
        st.exception(e)

# ================= FOOTER =================
st.markdown("""
<hr>
<center>
<b>Developed using Machine Learning & Streamlit</b><br>
🌱 Smart Agriculture Project
</center>
""", unsafe_allow_html=True)
