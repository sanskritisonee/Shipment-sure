# Shipment-sure
# 🚚 Shipment Sure – Shipment Delay Prediction System

Shipment Sure is an **end-to-end machine learning project** that predicts whether a shipment will be **On-Time** or **Delayed**, explains the reasons behind delays, and assesses operational risk using an interactive web application.

The project follows a **structured milestone-based approach**, covering the full ML lifecycle:
**data understanding → preprocessing → modeling → deployment**.

---

## 📌 Project Motivation

Shipment delays lead to:

* Increased logistics costs
* Poor customer satisfaction
* Supply chain inefficiencies

This project aims to help logistics planners and decision-makers:

* Predict shipment delays **before dispatch**
* Identify **key risk-driving factors**
* Perform **single and batch-level predictions**
* Make **data-driven operational decisions**

---

## 🧩 Project Milestones Overview

| Milestone   | Focus Area                                        |
| ----------- | ------------------------------------------------- |
| Milestone 1 | Data Understanding, Anomaly Generation & Cleaning |
| Milestone 2 | Exploratory Data Analysis & Feature Engineering   |
| Milestone 3 | Model Training & Evaluation                       |
| Milestone 4 | Model Deployment & Application Integration        |

---

## 📘 Milestone 1 – Data Understanding, Anomaly Generation & Data Cleaning

### Objective

To build a **reliable and consistent dataset** by understanding the data structure, simulating real-world anomalies, and cleaning the dataset for downstream analysis.

### Key Activities

* Dataset schema and datatype verification
* Validation of logical consistency in date fields
* Artificial anomaly generation:

  * Missing values
  * Duplicate records
  * Datatype errors
  * Outliers
  * Invalid dates
* Systematic anomaly cleaning:

  * Duplicate removal
  * Median/mode imputation
  * Datatype correction
  * Outlier capping using IQR
  * Invalid date removal

### Output

* `simple_cleaned_dataset.xlsx`
* Clean, consistent dataset ready for EDA

---

## 📗 Milestone 2 – Exploratory Data Analysis & Feature Engineering

### Objective

To extract insights from the cleaned data and transform it into a **machine-learning-ready dataset**.

### Key Activities

* Target variable distribution analysis
* Univariate, bivariate, and correlation analysis
* Identification of delay-driving patterns
* One-hot encoding of categorical variables
* Feature scaling using StandardScaler
* Feature engineering:

  * Delivery speed category
  * Long-distance indicator
  * High supplier rating flag

### Key Insights

* Longer delivery duration increases delay probability
* Lower supplier ratings correlate with delays
* Long-distance shipments are more delay-prone
* Shipment mode impacts delivery performance

### Output

* `processed_milestone2_dataset.xlsx`
* Fully encoded and engineered dataset

---

## 📌 Milestone 3 – Model Training & Evaluation

### Objective

To train a robust machine learning model capable of predicting shipment delays using pre-dispatch features.

### Model Details

* **Algorithm**: XGBoost Classifier
* **Problem Type**: Binary Classification

### Target Variable

Derived using delivery dates:

* `1` → Delayed
* `0` → On-Time

### Key Techniques

* Strict data leakage prevention
* Stratified train–test split
* Class imbalance handling using `scale_pos_weight`
* Hyperparameter tuning

### Evaluation Metrics

* ROC-AUC Score
* Confusion Matrix
* Precision, Recall, F1-score

### Explainability

* Feature importance analysis
* SHAP-based explanations for transparency

### Output Artifacts

* `shipment_delay_model.pkl`
* `model_features.pkl`

---

## 📌 Milestone 4 – Model Deployment & Application Integration

### Objective

To deploy the trained model as a **production-ready, interactive web application**.

### Deployment Stack

* **Frontend**: Streamlit
* **Backend**: XGBoost model via Joblib
* **Explainability**: SHAP

### Application Features

* Human-friendly UI with sliders & dropdowns
* Real-time prediction with probability score
* Risk classification:

  * 🟢 Low Risk
  * 🟡 Medium Risk
  * 🔴 High Risk
* SHAP-based explanation of predictions
* Demo shipment for quick testing
* Batch prediction via CSV upload

### Reliability Measures

* Safe model loading
* Feature schema validation
* Graceful error handling
* Streamlit caching for performance

---

## 🖥️ Streamlit Application

The deployed application allows users to:

* Enter shipment details interactively
* Predict delivery status with one click
* Understand **why** a shipment is delayed
* Upload CSV files for batch prediction

The UI is designed for **non-technical users**, making it suitable for real-world operational use.

---

## 📁 Repository Structure

```
shipment-sure/
│
├── app.py                      # Streamlit application
├── shipment_delay_model.pkl    # Trained ML model
├── model_features.pkl          # Feature schema
├── requirements.txt            # Dependencies
├── modeltraining.ipynb         # Model training notebook
├── eda.ipynb                   # Exploratory Data Analysis
├── dataexploration.ipynb       # Dataset understanding
├── datasets/                   # Raw & processed data
├── docs/                       # Milestone documentation
└── README.md                   # Project documentation
```

---

## ⚙️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧰 Tech Stack

* Python
* XGBoost
* Scikit-learn
* Pandas & NumPy
* SHAP
* Streamlit

---

## 🏁 Project Status

| Component                 | Status       |
| ------------------------- | ------------ |
| Data Cleaning             | ✅ Completed  |
| EDA & Feature Engineering | ✅ Completed  |
| Model Training            | ✅ Completed  |
| Explainability            | ✅ Integrated |
| Deployment                | ✅ Live       |
| Submission Ready          | ⭐⭐⭐⭐⭐        |

---

## 📌 Conclusion

**Shipment Sure** demonstrates a complete **industry-grade machine learning pipeline**, from raw data to a deployed, explainable application.
The project emphasizes **data quality, model reliability, transparency, and usability**, making it suitable for academic evaluation, interviews, and real-world demonstrations.

---
