import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# Optional SHAP import (safe)
# -----------------------------
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Shipment Delay Prediction",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Shipment Delay Prediction System")
st.write(
    "Predict whether a shipment will be **On-Time** or **Delayed**, "
    "understand **why**, and assess **risk level**."
)

# -----------------------------
# Load Model & Features
# -----------------------------
@st.cache_resource
def load_model(_version="v5"):
    model_path = "shipment_delay_model.pkl"
    feature_path = "model_features.pkl"

    if not os.path.exists(model_path):
        st.error("❌ shipment_delay_model.pkl not found")
        st.stop()

    if not os.path.exists(feature_path):
        st.error("❌ model_features.pkl not found")
        st.stop()

    model = joblib.load(model_path)
    features = joblib.load(feature_path)
    return model, features

model, feature_list = load_model()

# -----------------------------
# Demo Shipment
# -----------------------------
st.markdown("### 🧪 Demo Shipment")

if st.button("⚡ Load Demo Shipment"):
    st.session_state.demo = {
        "supplier_rating": 4.3,
        "supplier_lead_time": 6,
        "order_quantity": 20,
        "unit_price": 120.0,
        "shipping_distance_km": 180,
        "distance_type": "Long",
        "shipment_mode": "Road",
        "delivery_speed": "Slow"
    }
else:
    st.session_state.demo = {}

# -----------------------------
# Input UI
# -----------------------------
st.markdown("### 📦 Shipment Details")

input_data = {}

# Supplier info
input_data["supplier_rating"] = st.slider(
    "Supplier Rating",
    1.0, 5.0,
    float(st.session_state.demo.get("supplier_rating", 4.0)),
    0.1
)

input_data["supplier_lead_time"] = st.slider(
    "Supplier Lead Time (days)",
    1, 30,
    int(st.session_state.demo.get("supplier_lead_time", 7))
)

# Order info
input_data["order_quantity"] = st.number_input(
    "Order Quantity",
    min_value=1,
    value=int(st.session_state.demo.get("order_quantity", 10)),
    step=1
)

input_data["unit_price"] = st.number_input(
    "Unit Price",
    min_value=1.0,
    value=float(st.session_state.demo.get("unit_price", 100.0)),
    step=1.0
)

input_data["total_order_value"] = (
    input_data["order_quantity"] * input_data["unit_price"]
)

# Logistics
input_data["shipping_distance_km"] = st.slider(
    "Shipping Distance (km)",
    10, 3000,
    int(st.session_state.demo.get("shipping_distance_km", 200))
)

distance_type = st.selectbox(
    "Distance Category",
    ["Short", "Long"],
    index=1 if st.session_state.demo.get("distance_type") == "Long" else 0
)
input_data["long_distance"] = 1 if distance_type == "Long" else 0

shipment_mode = st.selectbox(
    "Shipment Mode",
    ["Road", "Sea"],
    index=0 if st.session_state.demo.get("shipment_mode") == "Road" else 1
)
input_data["shipment_mode_Road"] = 1 if shipment_mode == "Road" else 0
input_data["shipment_mode_Sea"] = 1 if shipment_mode == "Sea" else 0

# Delivery speed
speed = st.selectbox(
    "Expected Delivery Speed",
    ["Normal", "Slow", "Very Slow"],
    index=["Normal", "Slow", "Very Slow"].index(
        st.session_state.demo.get("delivery_speed", "Normal")
    )
)
input_data["delivery_speed_Normal"] = 1 if speed == "Normal" else 0
input_data["delivery_speed_Slow"] = 1 if speed == "Slow" else 0
input_data["delivery_speed_Very_Slow"] = 1 if speed == "Very Slow" else 0

# Derived feature
input_data["high_rating"] = 1 if input_data["supplier_rating"] >= 4.0 else 0

# Fill missing model features with 0
for f in feature_list:
    if f not in input_data:
        input_data[f] = 0

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔮 Predict Delivery Status"):
    input_df = pd.DataFrame([input_data])[feature_list]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")

    # Risk bands
    if probability < 0.4:
        st.success("🟢 LOW RISK — Likely On-Time")
    elif probability < 0.7:
        st.warning("🟡 MEDIUM RISK — Monitor Shipment")
    else:
        st.error("🔴 HIGH RISK — Likely Delayed")

    st.metric("Delay Probability", f"{probability:.2%}")

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    if SHAP_AVAILABLE:
        st.markdown("### 🔍 Why this prediction?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap_df = (
            pd.DataFrame({
                "Feature": feature_list,
                "Impact": shap_values[0]
            })
            .assign(AbsImpact=lambda x: x["Impact"].abs())
            .sort_values(by="AbsImpact", ascending=False)
            .head(5)
            .drop(columns="AbsImpact")
        )

        st.dataframe(shap_df, use_container_width=True)
    else:
        st.info("ℹ️ SHAP explanations unavailable (library not installed).")

# -----------------------------
# Batch CSV Upload
# -----------------------------
st.markdown("### 📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader(
    "Upload CSV with model feature columns",
    type=["csv"]
)

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    missing = [c for c in feature_list if c not in batch_df.columns]
    if missing:
        st.error(f"Missing columns in CSV: {missing}")
    else:
        preds = model.predict(batch_df[feature_list])
        probs = model.predict_proba(batch_df[feature_list])[:, 1]

        batch_df["Prediction"] = np.where(preds == 1, "Delayed", "On-Time")
        batch_df["Delay_Probability"] = probs

        st.success("✅ Batch prediction completed")
        st.dataframe(batch_df, use_container_width=True)
