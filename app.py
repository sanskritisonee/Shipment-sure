import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Shipment Delay Prediction",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Shipment Delay Prediction System")
st.write("Predict whether a shipment will be **On-Time** or **Delayed** using logistics and supplier information.")

# -----------------------------
# Load Model & Features
# -----------------------------
@st.cache_resource
def load_model(_version="v3"):
    model = joblib.load("shipment_delay_model.pkl")
    features = joblib.load("model_features.pkl")
    return model, features

model, feature_list = load_model()

# -----------------------------
# User Input UI
# -----------------------------
st.subheader("📦 Shipment Details")

input_data = {}

# ---------- Supplier Info ----------
st.markdown("### 🏭 Supplier Information")

input_data["supplier_rating"] = st.slider(
    "Supplier Rating",
    1.0, 5.0, 4.0, 0.1
)

input_data["supplier_lead_time"] = st.slider(
    "Supplier Lead Time (days)",
    1, 30, 7
)

# ---------- Order Info ----------
st.markdown("### 📦 Order Information")

input_data["order_quantity"] = st.number_input(
    "Order Quantity",
    min_value=1,
    value=10
)

input_data["unit_price"] = st.number_input(
    "Unit Price",
    min_value=1.0,
    value=100.0
)

input_data["total_order_value"] = input_data["order_quantity"] * input_data["unit_price"]

# ---------- Logistics ----------
st.markdown("### 🚚 Logistics")

input_data["shipping_distance_km"] = st.slider(
    "Shipping Distance (km)",
    10, 3000, 200
)

distance_type = st.selectbox(
    "Distance Category",
    ["Short", "Long"]
)
input_data["long_distance"] = 1 if distance_type == "Long" else 0

shipment_mode = st.selectbox(
    "Shipment Mode",
    ["Road", "Sea"]
)
input_data["shipment_mode_Road"] = 1 if shipment_mode == "Road" else 0
input_data["shipment_mode_Sea"] = 1 if shipment_mode == "Sea" else 0

# ---------- Delivery Speed ----------
st.markdown("### ⏱ Delivery Speed")

speed = st.selectbox(
    "Expected Delivery Speed",
    ["Normal", "Slow", "Very Slow"]
)

input_data["delivery_speed_Normal"] = 1 if speed == "Normal" else 0
input_data["delivery_speed_Slow"] = 1 if speed == "Slow" else 0
input_data["delivery_speed_Very_Slow"] = 1 if speed == "Very Slow" else 0

# ---------- Quality Signals ----------
st.markdown("### ⭐ Quality Indicators")

input_data["high_rating"] = 1 if input_data["supplier_rating"] >= 4.0 else 0

# -----------------------------
# Fill Missing Features with 0
# -----------------------------
for feature in feature_list:
    if feature not in input_data:
        input_data[feature] = 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Delivery Status"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_list]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.error("🚨 **Prediction: DELAYED**")
    else:
        st.success("✅ **Prediction: ON-TIME**")

    st.metric("Delay Probability", f"{probability:.2%}")
