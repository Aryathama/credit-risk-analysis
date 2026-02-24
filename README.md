# Credit Risk Default Prediction: A Business-Centric Approach

## Overview

A machine learning project focused on minimizing financial risk in credit scoring. Rather than optimizing purely for statistical metrics like AUC, this project optimizes for **Net Protected Value**—balancing the actual financial cost of False Negatives (lost capital) against False Positives (lost interest opportunity).

## 🚀 Business Impact

Based on a test set of 1,992 customers (Avg. Loan: Rp 17.9M, Target Interest: 12.5%):

* **Capital Protected (True Positives):** Rp 1.597 Billion
* **Opportunity Cost (False Positives):** Rp 1.061 Billion
* **Net Protected Value:** **+Rp 536 Million**
* *Conclusion:* The model's conservative threshold effectively shields high-principal capital, heavily outweighing the cost of rejected interest margins.

## 🛠 Methodology & Tech Stack

* **Models Evaluated:** Logistic Regression, Random Forest, XGBoost.
* **Hyperparameter Tuning:** `GridSearchCV` utilized for ensemble models to find optimal depths, learning rates, and subsample ratios.
* **Class Imbalance Handling:** Implemented via `class_weight='balanced'`, `scale_pos_weight`, and custom probability threshold adjustments (Threshold tuned to 0.56 for minimal cost).
* **Primary Evaluation Metric:** Custom Cost Function & ROC-AUC.

## 📊 Key Findings

1. **Simplicity Wins:** **Logistic Regression** outperformed complex ensemble models (XGBoost & Random Forest) in both the custom financial Cost metric and slightly in AUC (0.72).
2. **White-Box Interpretability:** As a linear model, Logistic Regression provided 100% transparency into its decision-making, which is crucial for financial regulatory compliance.
3. **Core Risk Drivers:** Extracted feature importance (coefficients) revealed that `loan_amount` heavily drives default risk (Positive), while `income` and `credit_score` act as the strongest risk mitigators (Negative).

## 📁 Repository Contents

* `main.ipynb`: End-to-end Jupyter Notebook covering EDA, preprocessing, model training, and threshold optimization.
* `executive_summary.html`: A custom-built, interactive HTML/CSS dashboard designed to present model interpretability and business impact to non-technical stakeholders.
