# Customer Churn Prediction

Predicting which customers are likely to cancel their subscription/service, using the IBM Telco Customer Churn dataset (7,043 customers, 20 features).

## Problem
Customer churn is one of the most expensive problems for subscription-based businesses (telecom, SaaS, streaming, gyms, etc.) — acquiring a new customer costs far more than retaining an existing one. This project builds a model that flags at-risk customers *before* they leave, so a business can intervene (discounts, outreach, support) in time.

## Approach
1. **Data cleaning** — handled missing/blank values in `TotalCharges`, encoded categorical variables
2. **Exploratory analysis** — examined churn patterns across contract type, tenure, and monthly charges
3. **Model training** — compared Logistic Regression vs Random Forest
4. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix
5. **Feature importance** — identified the top drivers of churn

## Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 79.9% | 64.3% | 54.8% | 59.2% | 0.840 |
| Random Forest | 80.5% | 67.6% | 50.8% | 58.0% | **0.841** |

## Key Insight
Contract type and tenure are the strongest predictors of churn — customers on month-to-month contracts with short tenure churn at dramatically higher rates than those on annual contracts. This is exactly the kind of actionable insight a business can act on immediately (e.g., incentivizing longer-term contracts for new customers).

## Files
- `churn_analysis.py` — full pipeline (cleaning → EDA → modeling → evaluation)
- `eda_overview.png` — exploratory data visualizations
- `model_evaluation.png` — ROC curves, confusion matrix, feature importance
- `results_summary.txt` — text summary of all metrics
- `telco_churn.csv` — dataset used

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

## Note
This is a demo/portfolio project using a public dataset. For a real client project, this same pipeline is adapted to their actual customer data, with the model and features tuned specifically to their business.
