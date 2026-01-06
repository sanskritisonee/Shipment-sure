import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Shipment Delay Prediction",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Shipment Delay Prediction System")
st.write("Predict whether a shipment will be **On-Time** or **Delayed** based on logistics features.")

# -----------------------------
# Load Model & Features
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("shipment_delay_model.pkl")
    features = joblib.load("model_features(1).pkl")
    return model, features

model, feature_list = load_model()

# -----------------------------
# Input Form
# -----------------------------
st.subheader("📦 Enter Shipment Details")

input_data = {}

for feature in feature_list:
    if feature.startswith(("region_", "carrier_name_", "weather_condition_", "holiday_period_")):
        input_data[feature] = st.checkbox(feature)
    else:
        input_data[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=0.0,
            step=1.0
        )

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔮 Predict Delivery Status"):
    input_df = pd.DataFrame([input_data])

    # Ensure column order matches training
    input_df = input_df[feature_list]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("🚨 **Prediction: DELAYED**")
    else:
        st.success("✅ **Prediction: ON-TIME**")

    st.write(f"**Delay Probability:** `{probability:.2%}`")
