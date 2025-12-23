import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================
#  Load Model & Preprocessing
# ============================

MODEL_PATH = "best_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except:
    st.error("⚠️ Could not load best_model.pkl. Please place it in the same folder as this app.")
    st.stop()

st.set_page_config(page_title="Shipment Delivery Prediction", layout="centered")

# App Title
st.title("📦 Shipment On-Time Delivery Prediction App")
st.write("Enter shipment details below to check whether the delivery will be **On-Time** or **Delayed**.")

# ============================
#   User Input Section
# ============================

st.header("🔹 Enter Shipment Details")

col1, col2 = st.columns(2)

with col1:
    warehouse_block = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "E", "F"])
    mode = st.selectbox("Mode of Shipment", ["Flight", "Ship", "Road"])
    product_importance = st.selectbox("Product Importance", ["low", "medium", "high"])

with col2:
    customer_rating = st.slider("Customer Rating (1–5)", 1, 5)
    cost_of_the_product = st.number_input("Cost of Product", min_value=1)
    weight_in_gms = st.number_input("Weight (grams)", min_value=1)

discount = st.number_input("Discount Offered", min_value=0)
prior_purchases = st.number_input("Prior Purchases", min_value=0)
gender = st.selectbox("Customer Gender", ["F", "M"])

st.write("---")

# ============================
#  Prepare Input for Model
# ============================

def preprocess_input():
    input_dict = {
        "Warehouse_block": warehouse_block,
        "Mode_of_Shipment": mode,
        "Customer_rating": customer_rating,
        "Cost_of_the_Product": cost_of_the_product,
        "Prior_purchases": prior_purchases,
        "Product_importance": product_importance,
        "Gender": gender,
        "Discount_offered": discount,
        "Weight_in_gms": weight_in_gms,
    }

    df = pd.DataFrame([input_dict])

    # One-hot encoding (use SAME columns as training)
    df = pd.get_dummies(df)

    # Load training columns to align properly
    try:
        train_cols = joblib.load("model_columns.pkl")
        df = df.reindex(columns=train_cols, fill_value=0)
    except:
        st.warning("model_columns.pkl missing — making best guess based on model input.")
        # If you don't have model_columns.pkl, model must accept raw dummies.

    return df


# ============================
#  Predict
# ============================

if st.button("🔍 Predict Delivery Status"):
    processed = preprocess_input()

    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1] * 100

    st.subheader("📌 Prediction Result")
    if prediction == 1:
        st.error(f"🚨 **Delayed Delivery** ({probability:.2f}% probability)")
    else:
        st.success(f"📦 **On-Time Delivery** ({100-probability:.2f}% probability)")

    st.info("Model: XGBoost Classifier (Best Performing Model)")

