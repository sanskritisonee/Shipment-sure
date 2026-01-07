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
    page_title="Shipment Sure",
    layout="centered"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 16px;
    color: #6c757d;
    margin-bottom: 2rem;
}
.section {
    font-size: 20px;
    font-weight: 600;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.divider {
    border-top: 1px solid #e6e6e6;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Shipment Sure</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Predict shipment delays, understand contributing factors, and assess delivery risk.'
    '</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Load Model & Features
# -----------------------------
@st.cache_resource
def load_model(_version="v5"):
    model_path = "shipment_delay_model.pkl"
    feature_path = "model_features.pkl"

    if not os.path.exists(model_path):
        st.error("Model file not found.")
        st.stop()

    if not os.path.exists(feature_path):
        st.error("Feature schema file not found.")
        st.stop()

    model = joblib.load(model_path)
    features = joblib.load(feature_path)
    return model, features

model, feature_list = load_model()

# -----------------------------
# Demo Shipment
# -----------------------------
st.markdown('<div class="section">Demo Shipment</div>', unsafe_allow_html=True)

if st.button("Load Demo Shipment"):
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
st.markdown('<div class="section">Shipment Details</div>', unsafe_allow_html=True)

input_data = {}

# Supplier Information
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

# Order Information
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

# Logistics Information
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

delivery_speed = st.selectbox(
    "Expected Delivery Speed",
    ["Normal", "Slow", "Very Slow"],
    index=["Normal", "Slow", "Very Slow"].index(
        st.session_state.demo.get("delivery_speed", "Normal")
    )
)
input_data["delivery_speed_Normal"] = 1 if delivery_speed == "Normal" else 0
input_data["delivery_speed_Slow"] = 1 if delivery_speed == "Slow" else 0
input_data["delivery_speed_Very_Slow"] = 1 if delivery_speed == "Very Slow" else 0

# Derived Feature
input_data["high_rating"] = 1 if input_data["supplier_rating"] >= 4.0 else 0

# Fill Missing Features
for f in feature_list:
    if f not in input_data:
        input_data[f] = 0

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Delivery Status"):
    input_df = pd.DataFrame([input_data])[feature_list]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if probability < 0.4:
        st.success("Low Risk — Shipment likely to be delivered on time.")
    elif probability < 0.7:
        st.warning("Medium Risk — Shipment requires monitoring.")
    else:
        st.error("High Risk — Shipment is likely to be delayed.")

    st.metric("Estimated Delay Probability", f"{probability:.2%}")

    # -----------------------------
    # SHAP Explanation
    # -----------------------------
    if SHAP_AVAILABLE:
        st.markdown('<div class="section">Prediction Explanation</div>', unsafe_allow_html=True)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        shap_df = (
            pd.DataFrame({
                "Feature": feature_list,
                "Impact": shap_values[0]
            })
            .assign(AbsoluteImpact=lambda x: x["Impact"].abs())
            .sort_values(by="AbsoluteImpact", ascending=False)
            .head(5)
            .drop(columns="AbsoluteImpact")
        )

        st.dataframe(shap_df, use_container_width=True)

# -----------------------------
# Batch CSV Upload
# -----------------------------
st.markdown('<div class="section">Batch Prediction</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a CSV file with required feature columns",
    type=["csv"]
)

if uploaded_file:
    batch_df = pd.read_csv(uploaded_file)

    missing_cols = [c for c in feature_list if c not in batch_df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
    else:
        preds = model.predict(batch_df[feature_list])
        probs = model.predict_proba(batch_df[feature_list])[:, 1]

        batch_df["Prediction"] = np.where(preds == 1, "Delayed", "On-Time")
        batch_df["Delay_Probability"] = probs

        st.success("Batch prediction completed successfully.")
        st.dataframe(batch_df, use_container_width=True)
