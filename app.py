import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==============================
# App Title
# ==============================
st.set_page_config(page_title="ShipmentSure", layout="wide")
st.title("📦 ShipmentSure – On-Time Delivery Prediction")

# ==============================
# Load Model & Features
# ==============================
ROOT = Path(__file__).resolve().parents[1]

@st.cache_resource
def load_model():
    obj = joblib.load(ROOT / "best_model.pkl")
    return obj["model"], obj["features"]

model, features = load_model()

# ==============================
# Sidebar Navigation
# ==============================
page = st.sidebar.radio(
    "Navigation",
    ["Predict Delivery", "Model Info", "EDA Preview"]
)

# =====================================================
# 1️⃣ PREDICT DELIVERY PAGE
# =====================================================
if page == "Predict Delivery":

    st.header("🔹 Enter Shipment Details")

    col1, col2 = st.columns(2)

    with col1:
        order_id = st.number_input("Order ID", min_value=1, value=1001)
        supplier_id = st.number_input("Supplier ID", min_value=1, value=10)
        supplier_rating = st.slider("Supplier Rating", 1, 5, 4)
        supplier_lead_time = st.number_input("Supplier Lead Time (days)", 1, 60, 7)
        shipping_distance_km = st.number_input("Shipping Distance (km)", 1, 50000, 1000)

    with col2:
        order_quantity = st.number_input("Order Quantity", 1, 10000, 100)
        unit_price = st.number_input("Unit Price", 1.0, 10000.0, 50.0)
        previous_on_time_rate = st.slider("Previous On-Time Rate (%)", 0, 100, 85)

        delivery_speed = st.selectbox("Delivery Speed", ["Normal", "Slow", "Very_Slow"])
        shipment_mode = st.selectbox("Shipment Mode", ["Road", "Sea"])
        weather = st.selectbox("Weather Condition", ["Cloudy", "Rainy", "Storm"])
        region = st.selectbox("Region", ["East", "North", "South", "West"])
        holiday = st.selectbox("Holiday Period", ["No", "Yes"])
        carrier = st.selectbox("Carrier", ["DHL", "Delhivery", "EcomExpress", "FedEx"])
        delay_reason = st.selectbox("Delay Reason", ["Operational", "Traffic", "Weather"])

    # ==============================
    # Feature Engineering
    # ==============================
    total_order_value = order_quantity * unit_price
    long_distance = int(shipping_distance_km > 1000)
    high_rating = int(supplier_rating >= 4)

    base_df = pd.DataFrame([{
        "order_id": order_id,
        "supplier_id": supplier_id,
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
        "carrier_name": carrier,
        "delayed_reason_code": delay_reason,
        "delivery_speed": delivery_speed
    }])

    # ==============================
    # One-Hot Encoding (Manual – SAME AS TRAINING)
    # ==============================
    category_map = {
        "shipment_mode": ["Road", "Sea"],
        "weather_condition": ["Cloudy", "Rainy", "Storm"],
        "region": ["East", "North", "South", "West"],
        "holiday_period": ["Yes"],
        "carrier_name": ["DHL", "Delhivery", "EcomExpress", "FedEx"],
        "delayed_reason_code": ["Operational", "Traffic", "Weather"],
        "delivery_speed": ["Normal", "Slow", "Very_Slow"]
    }

    for col, values in category_map.items():
        for val in values:
            base_df[f"{col}_{val}"] = (base_df[col] == val).astype(int)
        base_df.drop(columns=col, inplace=True)

    # ==============================
    # Align With Model Features
    # ==============================
    input_df = base_df.reindex(columns=features, fill_value=0)

    # ==============================
    # Prediction
    # ==============================
    if st.button("🔍 Predict Delivery"):
        prob_on_time = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        st.subheader("📊 Prediction Result")

        if prediction == 1:
            st.success(f"✅ **On-Time Delivery**\n\nConfidence: **{prob_on_time*100:.2f}%**")
        else:
            st.error(f"🚨 **Delayed Delivery**\n\nConfidence: **{(1-prob_on_time)*100:.2f}%**")

# =====================================================
# 2️⃣ MODEL INFO PAGE
# =====================================================
elif page == "Model Info":
    st.header("ℹ️ Model Information")

    st.write(f"**Model Type:** `{model.__class__.__name__}`")
    st.write(f"**Total Features Used:** `{len(features)}`")

    st.subheader("Feature List")
    st.dataframe(pd.DataFrame(features, columns=["Feature Name"]), height=500)

# =====================================================
# 3️⃣ EDA PREVIEW PAGE
# =====================================================
elif page == "EDA Preview":
    st.header("📈 Dataset Preview")

    data_path = ROOT / "data" / "processed_milestone2_dataset.xlsx"

    if data_path.exists():
        df = pd.read_excel(data_path)

        st.subheader("Sample Records")
        st.dataframe(df.head(50), height=400)

        st.subheader("Target Variable Distribution")
        st.bar_chart(df["on_time_delivery"].value_counts())

        st.subheader("Numerical Feature Summary")
        num_cols = [
            "supplier_rating", "supplier_lead_time",
            "shipping_distance_km", "order_quantity",
            "unit_price", "previous_on_time_rate"
        ]
        st.dataframe(df[num_cols].describe().T)
    else:
        st.warning("Processed dataset not found.")
