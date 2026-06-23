"""
feature_selection.py
---------------------
Three-method feature selection pipeline:
  1. ANOVA F-test
  2. Mutual Information
  3. Random Forest Importance
  4. RFECV (validation / cross-check)

Usage
-----
    from src.feature_selection import select_features
    selected_cols, importance_df = select_features(X, y)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFECV,
    SelectKBest,
    f_classif,
    mutual_info_classif,
)
from sklearn.model_selection import StratifiedKFold

from src.config import FEATURE_IMPORTANCE_THRESHOLD, RANDOM_STATE


def compute_importance_scores(
    X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """
    Compute normalised feature importance from three methods and combine.

    Returns
    -------
    pd.DataFrame
        Columns: F_Score, MI_Score, RF_Importance, Combined_Score
        Sorted descending by Combined_Score.
    """
    feature_names = X.columns.tolist()

    # 1. ANOVA F-test
    f_scores, _ = f_classif(X, y)
    f_series = pd.Series(f_scores, index=feature_names, name="F_Score")

    # 2. Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    mi_series = pd.Series(mi_scores, index=feature_names, name="MI_Score")

    # 3. Random Forest importance
    rf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X, y)
    rf_series = pd.Series(
        rf.feature_importances_, index=feature_names, name="RF_Importance"
    )

    # Normalise each to [0, 1] and average
    importance_df = pd.concat(
        [
            f_series / f_series.max(),
            mi_series / mi_series.max(),
            rf_series / rf_series.max(),
        ],
        axis=1,
    )
    importance_df["Combined_Score"] = importance_df.mean(axis=1)
    importance_df.sort_values("Combined_Score", ascending=False, inplace=True)

    return importance_df


def run_rfecv(X: pd.DataFrame, y: pd.Series, cv_folds: int = 3) -> list:
    """
    Run Recursive Feature Elimination with Cross-Validation.

    Returns
    -------
    list of selected feature names
    """
    rfecv = RFECV(
        estimator=RandomForestClassifier(
            n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1
        ),
        step=5,
        cv=StratifiedKFold(cv_folds),
        scoring="accuracy",
        n_jobs=-1,
    )
    rfecv.fit(X, y)
    return X.columns[rfecv.support_].tolist()


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = FEATURE_IMPORTANCE_THRESHOLD,
    use_rfecv: bool = True,
) -> tuple[list, pd.DataFrame]:
    """
    Full feature selection pipeline.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    threshold : float
        Minimum Combined_Score to include a feature (filter method).
    use_rfecv : bool
        Whether to also run RFECV and take the union.

    Returns
    -------
    (selected_feature_names, importance_df)
    """
    print("Computing importance scores...")
    importance_df = compute_importance_scores(X, y)

    filter_features = importance_df[
        importance_df["Combined_Score"] >= threshold
    ].index.tolist()
    print(f"  Filter method  → {len(filter_features)} features (threshold={threshold})")

    if use_rfecv:
        print("Running RFECV (this may take a few minutes)...")
        rfecv_features = run_rfecv(X, y)
        print(f"  RFECV optimal  → {len(rfecv_features)} features")
        final_features = list(set(filter_features) | set(rfecv_features))
    else:
        final_features = filter_features

    print(f"  Final union    → {len(final_features)} features selected")
    return final_features, importance_df
