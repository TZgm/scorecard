Credit Scoring Scorecard Development
Postgraduate Thesis Project | Corvinus University of Budapest "Data Analyst" (Adatelemző szakirányú továbbképzés)

This repository showcases the development of a banking credit scorecard, comparing traditional statistical methods with machine learning approaches.

📖 Project Background
The project focuses on the end-to-end workflow of credit risk modeling. It demonstrates how raw banking data is transformed into a decision-making tool. 

🛠 Methodology & Workflow
1. Data Understanding & Preprocessing
Handling Missing Values and outliers: Strategy-based imputation for missing data.

Feature Engineering: Implementing necessary transformations to prepare the data for credit modeling.

Weight of Evidence (WoE) & Information Value (IV): * Performed optimal binning.

Calculated IV to identify the predictive power of each variable and handle non-linear relationships.

2. Model Development
I implemented and compared two distinct modeling techniques:

Logistic Regression: The industry-standard approach for "glass-box" scorecard development.

Recursive Partitioning (RPA): A tree-based approach (Classification and Regression Trees) to capture non-linear interactions.

3. Model Evaluation
The models were evaluated and compared based on:

ROC Curve (Receiver Operating Characteristic): Visualizing the trade-off between sensitivity and specificity.

AUC (Area Under the Curve): Quantifying the overall discriminative power.

Scorecard Scaling: Converting model coefficients into a point-based scoring system for easy business interpretation.

💻 Implementation
Language: R
