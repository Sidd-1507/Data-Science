"""
tests/test_train.py
-------------------
Unit tests for the training pipeline utilities.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.train import evaluate_models, get_models
from sklearn.preprocessing import StandardScaler


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_dataset():
    """Tiny 7-class dataset to test pipeline without real data."""
    X, y = make_classification(
        n_samples=350,
        n_features=20,
        n_informative=10,
        n_classes=7,
        n_clusters_per_class=1,
        random_state=42,
    )
    y = y + 1  # shift to 1-indexed like real dataset
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_s  = pd.Series(y, name="Cover_Type")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_s, test_size=0.2, random_state=42, stratify=y_s
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc


@pytest.fixture
def cv():
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_get_models_returns_dict():
    models = get_models()
    assert isinstance(models, dict)
    assert len(models) == 10


def test_get_models_keys():
    models = get_models()
    expected = {
        "Logistic Regression", "Naive Bayes", "Decision Tree",
        "K-Nearest Neighbours", "Random Forest", "Extra Trees",
        "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM",
    }
    assert set(models.keys()) == expected


def test_evaluate_models_returns_correct_types(small_dataset, cv):
    X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc = small_dataset
    # Use only 2 fast models for speed
    fast_models = {
        k: v for k, v in get_models().items()
        if k in ["Naive Bayes", "Decision Tree"]
    }
    results, cv_scores, results_df = evaluate_models(
        fast_models, X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc, cv
    )
    assert isinstance(results, dict)
    assert isinstance(cv_scores, dict)
    assert isinstance(results_df, pd.DataFrame)


def test_evaluate_models_result_keys(small_dataset, cv):
    X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc = small_dataset
    fast_models = {"Naive Bayes": get_models()["Naive Bayes"]}
    results, _, _ = evaluate_models(
        fast_models, X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc, cv
    )
    expected_keys = {
        "CV_Accuracy_Mean", "CV_Accuracy_Std", "Test_Accuracy",
        "F1_Macro", "F1_Weighted", "Time_s", "y_pred", "model_object",
    }
    assert expected_keys.issubset(set(results["Naive Bayes"].keys()))


def test_results_df_sorted_by_accuracy(small_dataset, cv):
    X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc = small_dataset
    fast_models = {
        k: v for k, v in get_models().items()
        if k in ["Naive Bayes", "Decision Tree", "K-Nearest Neighbours"]
    }
    _, _, results_df = evaluate_models(
        fast_models, X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc, cv
    )
    accs = results_df["Test_Accuracy"].values
    assert all(accs[i] >= accs[i + 1] for i in range(len(accs) - 1))


def test_accuracy_in_valid_range(small_dataset, cv):
    X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc = small_dataset
    fast_models = {"Decision Tree": get_models()["Decision Tree"]}
    results, _, _ = evaluate_models(
        fast_models, X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc, cv
    )
    acc = results["Decision Tree"]["Test_Accuracy"]
    assert 0.0 <= acc <= 1.0
