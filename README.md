# Credit Card Routing For Online Purchase via Predictive Modelling

## Overview
Welcome to the Credit Card Routing for Online Purchase project! This project aims to enhance the payment success rate for online credit card transactions by leveraging predictive modelling to select the optimal Payment Service Provider (PSP) for each transaction. By automating this process, I aim to reduce transaction fees and improve customer satisfaction.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Project Structure](#project-structure)
4. [Data Quality Assessment](#data-quality-assessment)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Feature Engineering](#feature-engineering)
7. [Feature Selection](#feature-selection)
8. [Imbalanced Data Handling](imbalanced-data-handling)
9. [Model Development](#model-development)
10. [Feature Importance and Model Interpretability](#feature-importance-and-model-interpretability)
11. [Error Analysis](#error-analysis)
12. [Streamlit Application](#streamlit-application)
13. [Conclusion](#conclusion)


## Introduction
In the dynamic world of online retail, ensuring seamless payment processes is crucial for customer satisfaction and financial efficiency. This project addresses the challenge of high failure rates in online credit card payments by developing a predictive model to automate the selection of the best PSP for each transaction. By optimizing the routing logic, I aim to increase payment success rates while minimizing transaction fees.

## Data Description
The dataset consists of transaction records from January to February 2019, including the following key features:
* `tmsp` (Timestamp of transaction)
* `country` (Transaction origin country)
* `amount` (Transaction amount)
* `success` (Binary outcome: 1 = success, 0 = failure)
* `PSP` (Payment service provider used)
* `3D_secured` (Binary indicator for secure payments)
* `card` (Credit card provider: Visa, Master, Diners)

**List of PSPs and service fees**:

|     Name     |    Fee on successful transactions     |    Fee on failed transactions    |
|--------------| --------------------------------------|----------------------------------|
|   Moneycard  |              5 Euro                   |             2 Euro               |
|   Goldcard   |             10 Euro                   |             5 Euro               |
|    UK_Card   |              3 Euro                   |             1 Euro               |
|  Simplecard  |              1 Euro                   |           0,5 Euro               |


## Project Structure
The project is structured using the CRISP-DM methodology, which includes the following phases:

1. **Business Understanding**: Identifying the need to improve payment success rates and reduce transaction fees.
2. **Data Understanding**: Assessing the quality and structure of the provided dataset.
3. **Data Preparation**: Cleaning and preprocessing the data for analysis.
4. **Modelling**: Developing baseline and advanced predictive models.
5. **Evaluation**: Assessing model performance and interpreting results.
6. **Deployment**: Preparing the model for implementation in the production environment.

### Git Repository Structure
The git repository for this project is organized as follows:

    Credit_Card_Routing_For_Online_Purchase/
    │
    ├── data/
    │   ├── raw/
    │   │   └── PSP_Jan_Feb_2019.xlsx
    │   └── processed/
    │       └── credit_card_routing_cleaned_data.csv
    │
    ├── src/
    │   ├── data_preprocessing.py
    │   ├── feature_engineering.py
    │   ├── model_training.py
    │   ├── model_evaluation.py
    │   └── visualization.py
    │
    ├── notebook/
    │   └── credit_card_routing_for_online_purchase_v1_1.ipynb
    │
    ├── models/
    │   ├── credit_card_routing_rf_psp_models.joblib
    │   └── routing_data_psp_models.joblib
    │
    ├── app/
    │   └── credit_card_routing_streamlit_app.py
    │
    ├── reports/
    │   └── project_report.pdf
    │
    ├── README.md
    └── requirements.txt

## Data Quality Assessment
The initial dataset contained 50,410 records with 8 columns. Key findings from the data quality assessment include:

* **Completeness**: No missing values were found.
* **Uniqueness**: 81 duplicate records were identified and removed.
* **Accuracy**: All categorical and numerical columns contained valid values.
* **Consistency**: Categorical variables were consistent across records.
* **Timeliness**: Timestamps were valid and in the correct range.

## Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) revealed insights into transaction patterns and PSP performance:

* **Transaction Distribution**: Germany had the highest number of transactions, followed by Switzerland and Austria.
* **PSP Performance**: UK_Card had the highest transaction volume but a lower success rate. Goldcard had the highest success rate but was underutilized.
* **3D-Secured Transactions**: Transactions with 3D-Secured authentication had a significantly higher success rate.

## Feature Engineering
Feature engineering was performed to create meaningful features for model training:

* **Purchase Grouping**: Identified repeated attempts within 1 minute for the same country and amount.
* **Interaction Features**: Created interaction features combining PSP, card, 3D-Secured status, and country.
* **Log Transform**: Applied a log transform to the transaction amount to reduce skewness.
* **Time-Based Features**: Extracted hour of the day and day of the week from timestamps.

## Feature Selection
Feature Selection is done by selecting the top 10 features using SelectKBest.

## Imbalanced Data Handling
Class Imbalance handled using SMOTE (Synthetic Minority Over-sampling Technique).

## Model Development
Both baseline and advanced predictive models were developed to optimize PSP selection:

* **Baseline Model**:
    * A simple rule-based model was used as a baseline.
    * The baseline strategy implemented: 
        * `Always pick UK_Card`: The most frequently used PSP in the dataset.
    * This strategy was evaluated based on its expected cost and success rate.

* **Advanced Predictive Model**:
    * **Random Forest Classifier** and **LightGBM** were implemented for improved predictive performance.
    * Hyperparameter Tuning was performed using RandomizedSearchCV.
    * **Evaluation Metrics**:
        * Precision, Recall, and F1-Score to measure classification performance.
        * ROC-AUC Score for assessing model discrimination ability.
        * Confusion Matrix to analyze misclassification errors.


## Feature Importance & Model Interpretability
Feature importance analysis highlighted the significance of interaction features and time-based features in predicting payment success. The model's interpretability was enhanced by visualizing feature importances and Permutation Importance.


## Error Analysis
A sophisticated segmented error analysis was conducted through classification_report to understand the model's limitations.


## Streamlit Application
A Streamlit application was developed to provide a user-friendly interface for testing the predictive model. The app allows users to input transaction details and simulate both single-attempt and multiple-attempt routing strategies.

* **Single-Attempt Strategy**: Selects the PSP with the lowest expected cost for a single transaction attempt.
* **Multiple-Attempt Strategy**: Simulates repeated attempts until success or a maximum number of attempts is reached, choosing the best PSP for each attempt.

The app provides insights into the chosen PSP, predicted success probability, expected cost, and detailed logs of each attempt.


## Conclusion
The predictive model developed in this project successfully automates the selection of the optimal PSP for each transaction, increasing payment success rates and reducing transaction fees. By leveraging advanced feature engineering and model interpretation techniques, I have provided a robust solution to enhance the online payment process.

For further details, please refer to the project report and notebooks in the repository.

