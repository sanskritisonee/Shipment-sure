import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="ShipmentSure", layout="wide")
st.title("🚚 ShipmentSure – On-Time Delivery Prediction")

# =====================================================
# LOAD MODEL SAFELY
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
page = st.sidebar.radio(
    "Menu",
    ["Predict Delivery", "Model Info", "EDA Preview"]
)

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
        supplier_lead_time = st.number_input("Supplier Lead Time (days)", 1, 60, 7)

    with col2:
        shipping_distance_km = st.number_input("Shipping Distance (km)", 1, 50000, 1000)
        order_quantity = st.number_input("Order Quantity", 1, 10000, 100)
        unit_price = st.number_input("Unit Price", 1.0, 10000.0, 50.0)

    with col3:
        previous_on_time_rate = st.slider("Previous On-Time Rate (%)", 0, 100, 85)
        delivery_speed = st.selectbox("Delivery Speed", ["Normal", "Slow", "Very_Slow"])
        shipment_mode = st.selectbox("Shipment Mode", ["Road", "Sea"])

    weather = st.selectbox("Weather", ["Cloudy", "Rainy", "Storm"])
    region = st.selectbox("Region", ["East", "North", "South", "West"])
    holiday = st.selectbox("Holiday Period", ["No", "Yes"])
    carrier = st.selectbox("Carrier", ["DHL", "Delhivery", "EcomExpress", "FedEx"])
    delay_reason = st.selectbox("Delay Reason", ["Operational", "Traffic", "Weather"])

    total_order_value = order_quantity * unit_price
    long_distance = int(shipping_distance_km > 1000)
    high_rating = int(supplier_rating >= 4)

    # -------- Base DataFrame --------
    df = pd.DataFrame([{
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

    # -------- One-Hot Encoding --------
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
        for v in values:
            df[f"{col}_{v}"] = int(df[col].iloc[0] == v)
        df.drop(columns=[col], inplace=True)

    # -------- Align with training features --------
    df = df.reindex(columns=features, fill_value=0)

    # -------- Predict --------
    if st.button("🚀 Predict Delivery"):
        prob = model.predict_proba(df)[0][1]
        label = "✅ On-Time Delivery" if prob >= 0.5 else "❌ Delayed Delivery"

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
    st.dataframe(pd.DataFrame(features, columns=["Feature"]), height=450)

# =====================================================
# 3️⃣ EDA PREVIEW PAGE
# =====================================================
elif page == "EDA Preview":

    st.header("📈 Dataset Preview")

    data_path = Path(__file__).parent / "data" / "processed_milestone2_dataset.xlsx"

    if not data_path.exists():
        st.warning("Dataset file not found.")
    else:
        df_data = pd.read_excel(data_path)

        st.subheader("Sample Dataset (First 50 Rows)")
        st.dataframe(df_data.head(50), height=400)

        st.subheader("Target Variable Distribution")
        st.bar_chart(df_data["on_time_delivery"].value_counts())

        st.subheader("Summary Statistics")
        num_cols = [
            "supplier_rating", "supplier_lead_time", "shipping_distance_km",
            "order_quantity", "unit_price", "previous_on_time_rate"
        ]
        st.dataframe(df_data[num_cols].describe().T)
