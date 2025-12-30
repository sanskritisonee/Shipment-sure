📗 Milestone 2 – Exploratory Data Analysis & Feature Engineering

(Week 3–4)

1. Overview

Milestone 2 focuses on analyzing the cleaned dataset through Exploratory Data Analysis (EDA), extracting meaningful insights, and preparing the data for machine learning by applying preprocessing and feature engineering techniques.

The goal of this milestone is to transform the cleaned data into a machine-learning-ready dataset.

2. Data Loading & Initial Checks

Input File: simple_cleaned_dataset.xlsx

Performed:

Data type verification (df.info())

Statistical summary (df.describe())

Missing value verification (df.isnull().sum())

Observation

Dataset contained no missing values

Data types were consistent

Dataset was ready for analysis

3. Exploratory Data Analysis (EDA)
3.1 Target Variable Analysis

Analyzed the distribution of on_time_delivery

Observed class imbalance:

Delayed deliveries occurred more frequently than on-time deliveries

3.2 Univariate Analysis

Distribution analysis of numerical features such as:

Shipping distance

Order quantity

Supplier rating

3.3 Bivariate Analysis

Relationship between delivery_days and delivery status

Comparison of shipment modes with on-time delivery performance

3.4 Correlation Analysis

Generated correlation heatmap of numerical features

Identified strong relationships between delivery duration and delays

4. Key Insights from EDA

Longer delivery duration significantly increases delay probability

Lower supplier ratings correlate with delayed deliveries

Long-distance shipments are more prone to delays

Certain shipment modes perform better for on-time delivery

5. Data Preprocessing
5.1 Categorical Encoding

Applied One-Hot Encoding to categorical variables

Prevented dummy variable trap using drop_first=True

5.2 Feature Scaling

Normalized numerical features using StandardScaler

Ensured uniform feature contribution during model training

5.3 Train–Test Split

Split data into:

80% training

20% testing

Used stratified sampling to preserve class distribution

6. Feature Engineering
Engineered Features

Delivery Speed Category
Categorized delivery duration into meaningful speed groups

Long Distance Indicator
Binary flag indicating unusually long shipment distance

High Supplier Rating Flag
Binary indicator for highly rated suppliers

These features help capture non-linear delivery behavior and improve model performance.

7. Final Processed Dataset

Output File: processed_milestone2_dataset.xlsx

Contains:

Encoded categorical features

Scaled numerical features

Engineered features

Ready for machine learning modeling

8. Deliverables – Milestone 2
Deliverable	Status
EDA notebook	✔ Completed
EDA visualizations	✔ Completed
Encoded dataset	✔ Completed
Feature engineered dataset	✔ Completed
Final processed dataset	✔ Completed
9. Summary

Milestone 2 successfully transformed the cleaned dataset into a fully processed, machine-learning-ready format.
Through EDA and feature engineering, meaningful insights were extracted and predictive signals enhanced, enabling effective model development in Milestone 3.
