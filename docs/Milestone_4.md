📌 Milestone 4: Model Deployment and Application Integration
1. Objective

The objective of Milestone 4 is to deploy the trained shipment delay prediction model as an interactive, user-friendly web application that supports both real-time and batch predictions.

This milestone focuses on:

Model integration

User interface design

Explainability

Batch prediction support

Deployment readiness

2. Deployment Architecture

The deployment architecture consists of:

Frontend: Streamlit web application

Backend: Trained XGBoost model loaded via Joblib

Explainability: SHAP for feature-level insights

All components are integrated within a single, lightweight application suitable for cloud deployment.

3. Streamlit Application Design

The Streamlit interface was designed with non-technical users in mind.

Key design principles:

Human-readable feature labels

Sliders and dropdowns instead of raw numeric inputs

Clear sectioning for shipment details

Dark-mode friendly UI

4. Real-Time Prediction

Users can:

Enter shipment details

Select logistics parameters

Trigger prediction with a single click

The system outputs:

Prediction (On-Time / Delayed)

Delay probability

Risk category (Low / Medium / High)

5. Risk Classification

Delay probability is mapped to intuitive risk levels:

🟢 Low Risk – Likely On-Time

🟡 Medium Risk – Requires monitoring

🔴 High Risk – Likely Delayed

This helps stakeholders quickly interpret results.

6. Explainable AI Integration

SHAP (SHapley Additive Explanations) is integrated to explain predictions.

For each shipment:

Top contributing features are displayed

Feature impact magnitude is shown

Users can understand why a shipment is predicted to be delayed

This enhances transparency and trust.

7. Demo Shipment Feature

A Demo Shipment button was implemented to:

Instantly populate realistic input values

Support quick demonstrations

Aid viva and project presentations

8. Batch Prediction Support

The application supports CSV uploads for batch prediction.

Features:

Upload multiple shipments at once

Validate feature compatibility

Generate predictions and probabilities for each record

This simulates real-world operational usage.

9. Model Loading & Reliability

The application includes:

Safe model loading with file existence checks

Feature order validation

Graceful error handling

Streamlit caching for performance

This ensures reliable execution in cloud environments.

10. Deployment Readiness

The application is fully deployment-ready with:

requirements.txt for dependency management

Serialized model artifacts

Portable codebase

Cloud-compatible structure

It can be deployed on:

Streamlit Cloud

Local environments

Containerized platforms (future scope)

11. Outcome of Milestone 4

✔ Successfully deployed ML model as a web application
✔ Enabled real-time and batch predictions
✔ Integrated explainable AI
✔ Created a production-ready ML system

This milestone completes the end-to-end machine learning lifecycle, from data to deployment.
