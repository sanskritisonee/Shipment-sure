📘 Milestone 1 – Data Understanding, Anomaly Generation & Data Cleaning

(Week 1–2)

1. Overview

Milestone 1 focuses on understanding the shipment dataset, analyzing its structure, simulating real-world data quality issues through artificial anomaly generation, and finally cleaning the dataset to prepare it for further analysis and machine learning.

The primary objective of this milestone is to ensure that the dataset is accurate, consistent, and reliable, which is essential before performing Exploratory Data Analysis (EDA), feature engineering, and model building.

2. Dataset Understanding
2.1 Dataset Description

Dataset Name: shipment_dataset_10000.xlsx

Number of Records: ~10,000

Domain: Supply Chain & Logistics

Data Type: Structured tabular dataset

2.2 Key Columns

Identifiers:
order_id, supplier_id

Supplier Information:
supplier_rating, supplier_lead_time

Order & Shipment Details:
shipment_mode, shipping_distance_km,
order_quantity, unit_price, total_order_value

Date Fields:
order_date, promised_delivery_date, actual_delivery_date

Target Variable:
on_time_delivery

1 → On-time delivery

0 → Delayed delivery

2.3 Tasks Performed

Reviewed dataset schema and column definitions

Verified data types using df.info()

Analyzed numerical and categorical distributions

Checked for missing values and duplicates

Validated logical consistency of date columns
(order_date → promised_delivery_date → actual_delivery_date)

Confirmed that the original dataset did not contain critical anomalies

3. Anomaly Generation

To simulate real-world data issues commonly encountered in production systems, artificial anomalies were intentionally introduced using a Python script (add_anomalies.py).

3.1 Types of Anomalies Introduced
a) Missing Values

Random NaN values introduced in:

supplier_rating

order_quantity

Other key numeric columns

b) Duplicate Records

Random duplicate rows inserted

Simulates data ingestion and merging errors

c) Datatype Errors

Invalid string values such as "error_value" injected into numeric columns

Represents incorrect data entry scenarios

d) Outliers

Extreme and unrealistic values introduced, such as:

shipping_distance_km = 99999

Abnormally large order quantities

e) Invalid Dates

Modified actual_delivery_date to occur before order_date

Violates real-world business logic

3.2 Output

Generated File: simple_anomalies_dataset.xlsx

4. Anomaly Cleaning

The anomalous dataset was cleaned using a systematic preprocessing pipeline implemented in clean_anomalies.py.

4.1 Cleaning Steps Performed

Duplicate Removal

Removed duplicate rows using drop_duplicates()

Datatype Correction

Converted invalid string values to NaN

Cast columns back to appropriate numeric types

Missing Value Handling

Numerical columns → Median imputation

Categorical columns → Mode imputation

Invalid Date Handling

Removed rows where:

actual_delivery_date < order_date


Outlier Treatment

Applied IQR-based capping to numerical columns such as shipping_distance_km

Feature Engineering

Created a new feature:

delivery_days = actual_delivery_date − order_date

4.2 Output

Cleaned File: simple_cleaned_dataset.xlsx

5. Validation of Cleaning

A before-and-after comparison was conducted to validate the effectiveness of the cleaning process.

Validation Results

Duplicates

Present after anomaly injection

Completely removed after cleaning

Missing Values

Increased after anomaly generation

Reduced to acceptable levels after cleaning

Datatype Errors

Invalid string values removed

Numeric columns restored

Date Consistency

No records contain invalid delivery dates

Outliers

Extreme values capped using IQR method

The final dataset is now clean, consistent, and ready for EDA.

6. Deliverables – Milestone 1
Deliverable	Status
Original dataset	✔ Completed
Dataset with anomalies	✔ Completed
Cleaned dataset	✔ Completed
Anomaly generation script (add_anomalies.py)	✔ Completed
Cleaning script (clean_anomalies.py)	✔ Completed
Validation of cleaning	✔ Completed
7. Summary

Milestone 1 successfully established a robust data foundation by performing dataset understanding, simulating real-world anomalies, and cleaning the data using systematic preprocessing techniques.
The cleaned dataset is now suitable for Exploratory Data Analysis and feature engineering in Milestone
