"""
train.py
--------
Full training pipeline:
  - Load data
  - Feature engineering
  - Feature selection
  - Train 10 models with cross-validation
  - Hyperparameter tuning of the best model
  - Save results and best model

Run
---
    python src/train.py --data_path data/train.csv --output_dir outputs/
"""

import argparse
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

from src.config import (
    COVER_NAMES,
    CV_FOLDS,
    KAGGLE_DATA_PATH,
    LGBM_PARAM_DIST,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    TUNING_N_ITER,
    XGB_PARAM_DIST,
)
from src.feature_engineering import engineer_features
from src.feature_selection import select_features


# ── Model zoo ─────────────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Naive Bayes":         GaussianNB(),
        "Decision Tree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbours":KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "Extra Trees":         ExtraTreesClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "AdaBoost":            AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost":             XGBClassifier(n_estimators=200, eval_metric="mlogloss",
                                              random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        "LightGBM":            LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                               n_jobs=-1, verbose=-1),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test,
                    X_train_scaled, X_test_scaled, cv):
    """Train and evaluate all models, return results dict and DataFrame."""
    results = {}
    cv_scores_dict = {}
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    for name, model in models.items():
        print(f"  Training {name} ...", end="  ", flush=True)
        start = time.time()

        needs_scaling = name in ["Logistic Regression", "K-Nearest Neighbours"]
        Xtr = X_train_scaled if needs_scaling else X_train.values
        Xte = X_test_scaled  if needs_scaling else X_test.values

        cv_acc = cross_val_score(model, Xtr, y_train_enc, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        model.fit(Xtr, y_train_enc)
        y_pred_enc = model.predict(Xte)
        y_pred = le.inverse_transform(y_pred_enc)

        elapsed  = time.time() - start
        test_acc = accuracy_score(y_test_enc, y_pred_enc)
        f1_mac   = f1_score(y_test_enc, y_pred_enc, average="macro")
        f1_wt    = f1_score(y_test_enc, y_pred_enc, average="weighted")

        results[name] = {
            "CV_Accuracy_Mean": cv_acc.mean(),
            "CV_Accuracy_Std":  cv_acc.std(),
            "Test_Accuracy":    test_acc,
            "F1_Macro":         f1_mac,
            "F1_Weighted":      f1_wt,
            "Time_s":           elapsed,
            "y_pred":           y_pred,
            "model_object":     model,
        }
        cv_scores_dict[name] = cv_acc
        print(f"Acc={test_acc:.4f}  F1={f1_mac:.4f}  ({elapsed:.1f}s)")

    results_df = pd.DataFrame(
        {k: {kk: vv for kk, vv in v.items() if kk not in ("y_pred", "model_object")}
         for k, v in results.items()}
    ).T.astype(float)
    results_df.sort_values("Test_Accuracy", ascending=False, inplace=True)

    return results, cv_scores_dict, results_df


def tune_best_model(best_name, X_train, y_train, cv):
    """Run RandomizedSearchCV on the best model."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    if "LightGBM" in best_name:
        p = LGBM_PARAM_DIST
        param_dist = {
            "n_estimators":    randint(*p["n_estimators"]),
            "max_depth":       randint(*p["max_depth"]),
            "num_leaves":      randint(*p["num_leaves"]),
            "learning_rate":   uniform(p["learning_rate"][0], p["learning_rate"][1] - p["learning_rate"][0]),
            "subsample":       uniform(*p["subsample"]),
            "colsample_bytree":uniform(*p["colsample_bytree"]),
            "reg_alpha":       uniform(*p["reg_alpha"]),
            "reg_lambda":      uniform(*p["reg_lambda"]),
        }
        base = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    elif "XGBoost" in best_name:
        p = XGB_PARAM_DIST
        param_dist = {
            "n_estimators":    randint(*p["n_estimators"]),
            "max_depth":       randint(*p["max_depth"]),
            "learning_rate":   uniform(p["learning_rate"][0], p["learning_rate"][1] - p["learning_rate"][0]),
            "subsample":       uniform(*p["subsample"]),
            "colsample_bytree":uniform(*p["colsample_bytree"]),
            "reg_alpha":       uniform(*p["reg_alpha"]),
            "reg_lambda":      uniform(*p["reg_lambda"]),
        }
        base = XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE,
                             n_jobs=-1, verbosity=0)
    else:
        param_dist = {
            "n_estimators":    randint(100, 500),
            "max_depth":       [None, 10, 20, 30],
            "min_samples_split": randint(2, 10),
            "min_samples_leaf":  randint(1, 5),
            "max_features":    ["sqrt", "log2", None],
        }
        base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    rscv = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=TUNING_N_ITER,
        cv=cv,
        scoring="accuracy",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    rscv.fit(X_train.values, y_enc)
    return rscv, le


# ── Main ──────────────────────────────────────────────────────────────────────
def main(data_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load
    print("\n[1/6] Loading data...")
    if not os.path.exists(data_path):
        data_path = KAGGLE_DATA_PATH
    df = pd.read_csv(data_path)
    print(f"      Shape: {df.shape}")

    # 2. Feature engineering
    print("\n[2/6] Engineering features...")
    df_eng = engineer_features(df)
    X_all = df_eng.drop(TARGET_COL, axis=1)
    y_all = df_eng[TARGET_COL]

    # 3. Feature selection
    print("\n[3/6] Selecting features...")
    final_features, importance_df = select_features(X_all, y_all)
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"))
    X = df_eng[final_features]
    y = df_eng[TARGET_COL]

    # 4. Split & scale
    print("\n[4/6] Splitting and scaling...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 5. Train all models
    print("\n[5/6] Training all models...")
    models = get_models()
    results, cv_scores_dict, results_df = evaluate_models(
        models, X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, cv
    )

    print("\n── Model Summary ──")
    print(results_df[["CV_Accuracy_Mean", "Test_Accuracy", "F1_Macro", "F1_Weighted"]].to_string())
    results_df.to_csv(os.path.join(output_dir, "model_results.csv"))

    # 6. Tune best model
    best_name = results_df.index[0]
    print(f"\n[6/6] Tuning best model: {best_name}...")
    rscv, le = tune_best_model(best_name, X_train, y_train, cv)

    y_test_enc   = le.transform(y_test)
    y_pred_tuned = rscv.best_estimator_.predict(X_test.values)
    tuned_acc = accuracy_score(y_test_enc, y_pred_tuned)
    tuned_f1  = f1_score(y_test_enc, y_pred_tuned, average="macro")

    print(f"\n✅  Tuned {best_name}:")
    print(f"   CV Accuracy  : {rscv.best_score_:.4f}")
    print(f"   Test Accuracy: {tuned_acc:.4f}")
    print(f"   F1 Macro     : {tuned_f1:.4f}")
    print(f"\n   Best Params  : {rscv.best_params_}")

    # Save best model
    joblib.dump(rscv.best_estimator_, os.path.join(output_dir, "best_model.pkl"))
    joblib.dump(le,                   os.path.join(output_dir, "label_encoder.pkl"))
    joblib.dump(final_features,       os.path.join(output_dir, "selected_features.pkl"))
    print(f"\n💾 Artifacts saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Forest Cover Type models")
    parser.add_argument("--data_path",  default="data/train.csv")
    parser.add_argument("--output_dir", default="outputs/")
    args = parser.parse_args()
    main(args.data_path, args.output_dir)
