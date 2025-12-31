import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ShipmentSure", layout="wide")
st.title("🚚 ShipmentSure – On-Time Delivery Prediction")

# =====================================================
# LOAD MODEL + SCALER + FEATURES
# =====================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    features = joblib.load("model_features.pkl")
    return model, scaler, features

model, scaler, features = load_artifacts()

# =====================================================
# USER INPUT UI
# =====================================================
st.header("📦 Enter Shipment Details")

col1, col2, col3 = st.columns(3)

with col1:
    order_id = st.number_input("Order ID", 1, 999999, 1001)
    supplier_id = st.number_input("Supplier ID", 1, 9999, 101)
    supplier_rating = st.slider("Supplier Rating", 1, 5, 4)

with col2:
    supplier_lead_time = st.number_input("Supplier Lead Time (days)", 1, 60, 7)
    shipping_distance_km = st.number_input("Shipping Distance (km)", 1, 50000, 500)
    order_quantity = st.number_input("Order Quantity", 1, 10000, 50)

with col3:
    unit_price = st.number_input("Unit Price", 1.0, 10000.0, 100.0)
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

input_df = pd.DataFrame([{
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
# ONE-HOT ENCODING (EXPLICIT & SAFE)
# =====================================================
category_map = {
    "shipment_mode": ["Road", "Sea"],
    "weather_condition": ["Cloudy", "Rainy", "Storm"],
    "region": ["East", "North", "South", "West"],
    "holiday_period": ["Yes"],
    "carrier_name": ["DHL", "Delhivery", "EcomExpress", "FedEx"]
}

for col, values in category_map.items():
    val = input_df[col].iloc[0]
    for v in values:
        input_df[f"{col}_{v}"] = int(val == v)
    input_df.drop(columns=[col], inplace=True)

# =====================================================
# SCALING (MATCH TRAINING)
# =====================================================
scaler_features = list(scaler.feature_names_in_)

for col in scaler_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df[scaler_features] = scaler.transform(input_df[scaler_features])

# =====================================================
# ALIGN FEATURES
# =====================================================
input_df = input_df.reindex(columns=features, fill_value=0)

# =====================================================
# PREDICTION
# =====================================================
if st.button("🔮 Predict Delivery Status"):
    X_np = input_df.values

    prediction = model.predict(X_np)[0]
    probability = model.predict_proba(X_np)[0][1]

    st.subheader("📊 Prediction Result")

    colA, colB = st.columns(2)

    with colA:
        st.write(f"**Order ID:** {order_id}")
        st.write(f"**Supplier ID:** {supplier_id}")

    with colB:
        if prediction == 1:
            st.success("✅ On-Time Delivery Expected")
        else:
            st.error("⏰ Delivery Likely to be Delayed")

        st.metric("On-Time Probability", f"{probability*100:.2f}%")

    with st.expander("🔍 View Processed Model Input"):
        st.dataframe(input_df.loc[:, ~input_df.columns.duplicated()])
