📌 Milestone 3: Model Training and Evaluation
1. Objective

The objective of Milestone 3 is to train a robust machine learning model capable of predicting whether a shipment will be On-Time or Delayed, using logistics and supplier-related features derived during earlier milestones.

This milestone focuses on:

Target variable creation

Feature selection

Handling class imbalance

Model training

Performance evaluation

Model persistence for deployment

2. Target Variable Definition

Since the dataset does not explicitly contain a delay label, the target variable was engineered using delivery dates.

Definition:

delayed = 1 → Actual delivery date > Promised delivery date

delayed = 0 → Delivered on or before promised date

This approach mirrors real-world logistics delay definitions.

3. Feature Selection & Leakage Prevention

Only pre-dispatch features were used to train the model.
To prevent data leakage, the following columns were explicitly removed before training:

Order and supplier identifiers

Delivery dates

Delay reason codes

Historical aggregates derived post-delivery

This ensures the model predicts delays before shipment execution, not after outcomes are known.

4. Handling Class Imbalance

The dataset exhibited class imbalance, with delayed shipments being more frequent.

To address this:

Class weights were computed using compute_class_weight

The scale_pos_weight parameter was applied in XGBoost

This ensured balanced learning and prevented bias toward the majority class.

5. Model Selection

Algorithm Used:

XGBoost Classifier

Reasons for Selection:

Strong performance on tabular data

Handles non-linear relationships

Robust to feature scaling

Widely used in production-grade ML systems

6. Model Training

The dataset was split into training and testing sets using stratified sampling to preserve class distribution.

Key hyperparameters:

Number of estimators

Maximum tree depth

Learning rate

Subsampling and column sampling

Class imbalance correction

The model was trained using the training subset and validated on unseen test data.

7. Model Evaluation

The following evaluation metrics were used:

ROC-AUC Score

Confusion Matrix

Precision, Recall, and F1-Score

These metrics were chosen to ensure:

High recall for delayed shipments

Balanced precision for on-time predictions

Overall model reliability

The results demonstrated strong predictive performance with realistic, non-overfitted metrics.

8. Explainability Considerations

Feature importance and SHAP-based explanations were used to:

Interpret model predictions

Identify key delay-driving factors

Improve trust and transparency

This step ensures the model is explainable and audit-ready.

9. Model Persistence

The trained artifacts were saved for deployment:

shipment_delay_model.pkl – trained model

model_features.pkl – ordered feature schema

These files are used directly in the deployment phase.

10. Outcome of Milestone 3

✔ Trained a reliable shipment delay prediction model
✔ Addressed class imbalance and data leakage
✔ Achieved strong evaluation metrics
✔ Generated deployable model artifacts

This milestone successfully prepares the system for real-world deployment.
