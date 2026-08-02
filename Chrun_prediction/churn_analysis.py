"""
Customer Churn Prediction - Portfolio Project
Author: Siddharth Sharma
Dataset: IBM Telco Customer Churn (public dataset)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ------------------------------------------------------------------
df = pd.read_csv("telco_churn.csv")

# TotalCharges has some blank strings -> convert to numeric, fill missing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

df.drop("customerID", axis=1, inplace=True)

print("Dataset shape:", df.shape)
print("Churn rate: {:.2f}%".format((df["Churn"] == "Yes").mean() * 100))

# ------------------------------------------------------------------
# 2. EDA VISUALIZATIONS
# ------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Churn distribution
churn_counts = df["Churn"].value_counts()
axes[0, 0].bar(churn_counts.index, churn_counts.values, color=["#4C72B0", "#DD8452"])
axes[0, 0].set_title("Customer Churn Distribution", fontsize=13, fontweight="bold")
axes[0, 0].set_ylabel("Number of Customers")

# Churn by contract type
contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
contract_churn.plot(kind="bar", ax=axes[0, 1], color=["#4C72B0", "#DD8452"])
axes[0, 1].set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
axes[0, 1].set_ylabel("% of Customers")
axes[0, 1].legend(title="Churn")
axes[0, 1].tick_params(axis="x", rotation=20)

# Monthly charges distribution by churn
for churn_val, color in zip(["Yes", "No"], ["#DD8452", "#4C72B0"]):
    subset = df[df["Churn"] == churn_val]["MonthlyCharges"]
    axes[1, 0].hist(subset, bins=30, alpha=0.6, label=f"Churn={churn_val}", color=color)
axes[1, 0].set_title("Monthly Charges by Churn Status", fontsize=13, fontweight="bold")
axes[1, 0].set_xlabel("Monthly Charges ($)")
axes[1, 0].legend()

# Tenure vs churn
for churn_val, color in zip(["Yes", "No"], ["#DD8452", "#4C72B0"]):
    subset = df[df["Churn"] == churn_val]["tenure"]
    axes[1, 1].hist(subset, bins=30, alpha=0.6, label=f"Churn={churn_val}", color=color)
axes[1, 1].set_title("Customer Tenure by Churn Status", fontsize=13, fontweight="bold")
axes[1, 1].set_xlabel("Tenure (months)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("eda_overview.png", bbox_inches="tight")
plt.close()
print("Saved eda_overview.png")

# ------------------------------------------------------------------
# 3. PREPROCESSING FOR MODELING
# ------------------------------------------------------------------
model_df = df.copy()
target = model_df.pop("Churn")
target = target.map({"Yes": 1, "No": 0})

cat_cols = model_df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    model_df[col] = le.fit_transform(model_df[col])

X_train, X_test, y_train, y_test = train_test_split(
    model_df, target, test_size=0.2, random_state=42, stratify=target
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. TRAIN MODELS
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
}

results = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model": model,
        "preds": preds,
        "probs": probs,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "auc": roc_auc_score(y_test, probs),
    }
    print(f"\n{name}:")
    print(f"  Accuracy:  {results[name]['accuracy']:.3f}")
    print(f"  Precision: {results[name]['precision']:.3f}")
    print(f"  Recall:    {results[name]['recall']:.3f}")
    print(f"  F1 Score:  {results[name]['f1']:.3f}")
    print(f"  ROC-AUC:   {results[name]['auc']:.3f}")

best_model_name = max(results, key=lambda k: results[k]["auc"])
print(f"\nBest model: {best_model_name}")

# ------------------------------------------------------------------
# 5. MODEL EVALUATION VISUALIZATIONS
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["probs"])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves", fontsize=13, fontweight="bold")
axes[0].legend()

# Confusion matrix for best model
cm = confusion_matrix(y_test, results[best_model_name]["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
axes[1].set_title(f"Confusion Matrix ({best_model_name})", fontsize=13, fontweight="bold")
axes[1].set_ylabel("Actual")
axes[1].set_xlabel("Predicted")

# Feature importance (Random Forest)
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=model_df.columns)
top_features = importances.sort_values(ascending=False).head(10)
axes[2].barh(top_features.index[::-1], top_features.values[::-1], color="#55A868")
axes[2].set_title("Top 10 Predictive Features", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Importance")

plt.tight_layout()
plt.savefig("model_evaluation.png", bbox_inches="tight")
plt.close()
print("Saved model_evaluation.png")

# ------------------------------------------------------------------
# 6. SAVE SUMMARY REPORT
# ------------------------------------------------------------------
with open("results_summary.txt", "w") as f:
    f.write("CUSTOMER CHURN PREDICTION - RESULTS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset: {df.shape[0]} customers, {df.shape[1]} features\n")
    f.write(f"Overall churn rate: {(target.mean() * 100):.2f}%\n\n")
    for name, res in results.items():
        f.write(f"{name}:\n")
        f.write(f"  Accuracy:  {res['accuracy']:.3f}\n")
        f.write(f"  Precision: {res['precision']:.3f}\n")
        f.write(f"  Recall:    {res['recall']:.3f}\n")
        f.write(f"  F1 Score:  {res['f1']:.3f}\n")
        f.write(f"  ROC-AUC:   {res['auc']:.3f}\n\n")
    f.write(f"Best model: {best_model_name}\n\n")
    f.write("Top predictive features:\n")
    for feat, imp in top_features.items():
        f.write(f"  {feat}: {imp:.4f}\n")

print("\nAll done. Files created: eda_overview.png, model_evaluation.png, results_summary.txt")
