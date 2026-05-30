"""
features.py
-----------
Leakage-free feature engineering for the Tesla EV Deliveries forecasting project.

All transformations use only information that would be available *before* a
quarter's delivery figure is published. Import this module in the notebook or
any downstream script to keep engineering logic DRY.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# Columns confirmed to cause target leakage — drop immediately after loading
LEAKY_COLS: list[str] = ["Production_Units"]

# Raw columns that encode the target if engineered — never create these
# "Delivery_Efficiency" = Deliveries / Production
# "Production_Surplus"  = Production − Deliveries
FORBIDDEN_DERIVED: list[str] = ["Delivery_Efficiency", "Production_Surplus"]

TARGET: str = "Estimated_Deliveries"
CATEGORICAL: list[str] = ["Region", "Model", "Source_Type"]

# Dropped before modelling (encoded via Quarter / Is_Q4 instead)
DROP_BEFORE_MODEL: list[str] = [TARGET, "Year", "Month"]


# ── Leakage audit ─────────────────────────────────────────────────────────────

def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that constitute target leakage.

    Parameters
    ----------
    df : raw DataFrame straight from pd.read_csv

    Returns
    -------
    DataFrame with leaky columns removed.
    """
    present = [c for c in LEAKY_COLS if c in df.columns]
    if present:
        print(f"[leakage audit] Dropping: {present}")
    else:
        print("[leakage audit] No known leaky columns found.")
    return df.drop(columns=present)


# ── Safe feature engineering ──────────────────────────────────────────────────

def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-per-km and charging impact — derived from spec / public data
    that is available before deliveries are reported.
    """
    df = df.copy()
    df["Price_Per_KM"]    = df["Avg_Price_USD"] / df["Range_km"].replace(0, np.nan)
    df["Charging_Impact"] = df["Charging_Stations"] / 1_000
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Quarter and Is_Q4 from the Month column."""
    df = df.copy()
    df["Quarter"] = ((df["Month"] - 1) // 3) + 1
    df["Is_Q4"]   = (df["Quarter"] == 4).astype(int)
    return df


def add_lag_features(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Add per-series lag and rolling features.

    Lags are computed **per (Region, Model) group** using .shift(1) so that:
      - No future delivery values leak into training rows.
      - Lags from one series (e.g. "Model 3 / China") never bleed into
        another (e.g. "Model Y / USA").

    The DataFrame MUST be sorted chronologically before calling this function.

    Parameters
    ----------
    df         : DataFrame sorted by ['Year', 'Month']
    group_cols : columns defining each independent series (default: Region + Model)
    """
    if group_cols is None:
        group_cols = ["Region", "Model"]

    df = df.copy()
    grp = df.groupby(group_cols)[TARGET]

    df["Lag1_Deliveries"]     = grp.shift(1)
    df["Lag2_Deliveries"]     = grp.shift(2)
    df["Rolling3_Deliveries"] = grp.shift(1).transform(lambda s: s.rolling(3).mean())
    df["Rolling6_Deliveries"] = grp.shift(1).transform(lambda s: s.rolling(6).mean())

    lag_cols = [
        "Lag1_Deliveries", "Lag2_Deliveries",
        "Rolling3_Deliveries", "Rolling6_Deliveries",
    ]
    for col in lag_cols:
        df[col] = df.groupby(group_cols)[col].transform(
            lambda s: s.fillna(s.median())
        )

    return df


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full leakage-free feature engineering pipeline.

    Steps
    -----
    1. Drop leaky columns
    2. Deduplicate
    3. Sort chronologically (required for correct lags)
    4. Add ratio features
    5. Add temporal features
    6. Add lag / rolling features

    Parameters
    ----------
    df_raw : raw DataFrame from pd.read_csv

    Returns
    -------
    Feature-engineered DataFrame ready for train/test split.
    """
    df = df_raw.copy()

    # 1. Drop leaky columns
    df = drop_leaky_columns(df)

    # 2. Deduplicate
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[preprocessing] Removed {before - len(df)} duplicates  |  rows: {len(df):,}")

    # 3. Sort chronologically — MUST happen before lag computation
    df = df.sort_values(["Year", "Month"]).reset_index(drop=True)

    # 4–6. Feature engineering
    df = add_ratio_features(df)
    df = add_temporal_features(df)
    df = add_lag_features(df)

    print(f"[features] Done  |  shape: {df.shape}")
    return df


# ── Temporal train/test split ─────────────────────────────────────────────────

def temporal_split(
    df: pd.DataFrame,
    split_year: int = 2023,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Chronological train / test split.

    Train: Year <= split_year
    Test : Year >  split_year

    Returns X_train, y_train, X_test, y_test.
    """
    X = df.drop(columns=DROP_BEFORE_MODEL)
    y = df[TARGET]

    mask = df["Year"] <= split_year
    X_train, y_train = X[mask],  y[mask]
    X_test,  y_test  = X[~mask], y[~mask]

    print(
        f"[split] Train: {X_train.shape[0]:,} rows "
        f"({df.loc[mask, 'Year'].min()}–{df.loc[mask, 'Year'].max()})  |  "
        f"Test: {X_test.shape[0]:,} rows "
        f"({df.loc[~mask, 'Year'].min()}–{df.loc[~mask, 'Year'].max()})"
    )
    return X_train, y_train, X_test, y_test


# ── 2026 forecast helper ──────────────────────────────────────────────────────

def build_forecast_frame(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build a one-row-per-(Region, Model) forecast frame for the next quarter
    after the last available data point.

    Lag features are rolled forward using the last known delivery value —
    no 2026 target information is used.
    """
    last_rows = (
        df_clean.sort_values(["Year", "Month"])
        .groupby(["Region", "Model"])
        .last()
        .reset_index()
    )

    future = last_rows.copy()
    future["Year"]    = 2026
    future["Quarter"] = (future["Quarter"] % 4) + 1
    future["Is_Q4"]   = (future["Quarter"] == 4).astype(int)

    # Roll lags forward
    future["Lag2_Deliveries"]     = future["Lag1_Deliveries"]
    future["Lag1_Deliveries"]     = future[TARGET]          # last known actual
    future["Rolling3_Deliveries"] = (
        future["Lag1_Deliveries"] * 0.5 + future["Rolling3_Deliveries"] * 0.5
    )
    future["Rolling6_Deliveries"] = (
        future["Lag1_Deliveries"] * 0.3 + future["Rolling6_Deliveries"] * 0.7
    )

    return future
