import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="ShipmentSure", layout="wide")
st.title("🚚 ShipmentSure – On-Time Delivery Prediction")

# =====================================================
# LOAD MODEL + FEATURES
# =====================================================
@st.cache_resource
def load_model():
    ROOT = Path(__file__).parent

    model_path = ROOT / "best_model.pkl"
    feature_path = ROOT / "model_features.pkl"

    if not model_path.exists():
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()

    if not feature_path.exists():
        st.error(f"❌ Feature file not found at: {feature_path}")
        st.stop()

    model = joblib.load(model_path)
    features = joblib.load(feature_path)

    return model, features

model, features = load_model()

# =====================================================
# SIDEBAR
# =====================================================
page = st.sidebar.radio("Menu", ["Predict Delivery", "Model Info"])

# =====================================================
# 1️⃣ PREDICT DELIVERY PAGE
# =====================================================
if page == "Predict Delivery":

    st.header("📦 Enter Shipment Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        order_id = st.number_input("Order ID", 1, 999999, 1001)
        supplier_id = st.number_input("Supplier ID", 1, 9999, 10)
        supplier_rating = st.slider("Supplier Rating", 1, 5, 4)

    with col2:
        supplier_lead_time = st.number_input("Supplier Lead Time (days)", 1, 60, 7)
        shipping_distance_km = st.number_input("Shipping Distance (km)", 1, 50000, 1000)
        order_quantity = st.number_input("Order Quantity", 1, 10000, 100)

    with col3:
        unit_price = st.number_input("Unit Price", 1.0, 10000.0, 50.0)
        previous_on_time_rate = st.slider("Previous On-Time Rate (%)", 0, 100, 85)
        shipment_mode = st.selectbox("Shipment Mode", ["Road", "Sea"])

    weather = st.selectbox("Weather Condition", ["Cloudy", "Rainy", "Storm"])
    region = st.selectbox("Region", ["East", "North", "South", "West"])
    holiday = st.selectbox("Holiday Period", ["No", "Yes"])
    carrier = st.selectbox("Carrier", ["DHL", "Delhivery", "EcomExpress", "FedEx"])

    # =====================================================
    # FEATURE ENGINEERING (MATCH TRAINING)
    # =====================================================
    total_order_value = order_quantity * unit_price
    long_distance = int(shipping_distance_km > 1000)
    high_rating = int(supplier_rating >= 4)

    # ❗ IDs are NOT part of model input
    model_df = pd.DataFrame([{
        "supplier_rating": supplier_rating,
        "supplier_lead_time": supplier_lead_time,
        "shipping_distance_km": shipping_distance_km,
        "order_quantity": order_quantity,
        "unit_price": unit_price,
        "total_order_value": total_order_value,
        "previous_on_time_rate": previous_on_time_rate,
        "long_distance": long_distance,
        "high_rating": high_rating,
        "shipment_mode": shipment_mode,
        "weather_condition": weather,
        "region": region,
        "holiday_period": holiday,
        "carrier_name": carrier
    }])

    # =====================================================
    # ONE-HOT ENCODING (SAFE)
    # =====================================================
    category_map = {
        "shipment_mode": ["Road", "Sea"],
        "weather_condition": ["Cloudy", "Rainy", "Storm"],
        "region": ["East", "North", "South", "West"],
        "holiday_period": ["Yes"],
        "carrier_name": ["DHL", "Delhivery", "EcomExpress", "FedEx"]
    }

    for col, values in category_map.items():
        if col not in model_df.columns:
            continue

        col_value = model_df[col].iloc[0]

        for v in values:
            model_df[f"{col}_{v}"] = int(col_value == v)

        model_df.drop(columns=[col], inplace=True)

    # =====================================================
    # ALIGN FEATURES WITH TRAINING
    # =====================================================
    model_df = model_df.reindex(columns=features, fill_value=0)

    # =====================================================
    # PREDICTION
    # =====================================================
    if st.button("🚀 Predict Delivery"):
        prob = model.predict_proba(model_df)[0][1]
        label = "✅ On-Time Delivery" if prob >= 0.5 else "❌ Delayed Delivery"

        st.subheader("📊 Prediction Result")
        st.write(f"**Order ID:** {order_id}")
        st.write(f"**Supplier ID:** {supplier_id}")

        st.success(label)
        st.metric("On-Time Delivery Probability", f"{prob*100:.2f}%")

# =====================================================
# 2️⃣ MODEL INFO PAGE
# =====================================================
elif page == "Model Info":

    st.header("📊 Model Information")
    st.write(f"**Model Type:** `{model.__class__.__name__}`")
    st.write(f"**Total Features Used:** `{len(features)}`")

    st.subheader("Feature List")
    st.dataframe(pd.DataFrame(features, columns=["Feature"]), height=500)
