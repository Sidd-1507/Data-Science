"""
feature_engineering.py
-----------------------
All feature creation logic for the Forest Cover Type dataset.
Call `engineer_features(df)` to get the enriched DataFrame.
"""

import numpy as np
import pandas as pd


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply domain-informed feature engineering to the raw dataset.

    New features created
    --------------------
    - Log-transformed distance columns (reduce right skew)
    - Euclidean distance to hydrology (combining H & V distances)
    - Mean distance to amenities
    - Elevation × Aspect / Slope trigonometric interactions
    - Hillshade summary statistics (mean, diurnal range)
    - Soil type & wilderness area counts

    Parameters
    ----------
    data : pd.DataFrame
        Raw input dataframe containing original dataset columns.

    Returns
    -------
    pd.DataFrame
        Copy of input with 12 additional engineered features.
    """
    df = data.copy()

    # ── Log-transformed distances (reduce positive skew) ──────────────────────
    for col in [
        "Horizontal_Distance_To_Fire_Points",
        "Horizontal_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
    ]:
        df[f"{col}_Log"] = np.log1p(df[col])

    # ── Euclidean distance to nearest hydrology ────────────────────────────────
    df["Euclidean_Distance_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"] ** 2
        + df["Vertical_Distance_To_Hydrology"] ** 2
    )

    # ── Mean of all horizontal distances ──────────────────────────────────────
    df["Mean_Distance_To_Amenities"] = (
        df["Horizontal_Distance_To_Fire_Points"]
        + df["Horizontal_Distance_To_Hydrology"]
        + df["Horizontal_Distance_To_Roadways"]
    ) / 3

    # ── Elevation interaction with aspect (cosine component) ──────────────────
    df["Elevation_Aspect_Interaction"] = df["Elevation"] * np.cos(
        np.radians(df["Aspect"])
    )

    # ── Elevation interaction with slope (sine component) ─────────────────────
    df["Elevation_Slope_Interaction"] = df["Elevation"] * np.sin(
        np.radians(df["Slope"])
    )

    # ── Hillshade summary statistics ──────────────────────────────────────────
    hs_cols = ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    df["Hillshade_Mean"] = df[hs_cols].mean(axis=1)
    df["Hillshade_Range"] = df[hs_cols].max(axis=1) - df[hs_cols].min(axis=1)

    # ── Soil richness: number of active soil types ─────────────────────────────
    soil_cols = [c for c in df.columns if c.startswith("Soil")]
    df["Soil_Type_Count"] = df[soil_cols].sum(axis=1)

    # ── Wilderness area count ─────────────────────────────────────────────────
    wild_cols = [c for c in df.columns if c.startswith("Wilderness")]
    df["Wilderness_Area_Count"] = df[wild_cols].sum(axis=1)

    return df


def get_feature_groups(df: pd.DataFrame):
    """
    Return lists of column names grouped by type.

    Returns
    -------
    dict with keys: 'numerical', 'binary', 'engineered', 'all_features'
    """
    target = "Cover_Type"
    binary_cols = [
        c for c in df.columns if df[c].nunique() == 2 and c != target
    ]
    num_cols = [
        c
        for c in df.select_dtypes(include=np.number).columns
        if c not in binary_cols and c != target
    ]
    engineered = [
        "Horizontal_Distance_To_Fire_Points_Log",
        "Horizontal_Distance_To_Hydrology_Log",
        "Horizontal_Distance_To_Roadways_Log",
        "Euclidean_Distance_Hydrology",
        "Mean_Distance_To_Amenities",
        "Elevation_Aspect_Interaction",
        "Elevation_Slope_Interaction",
        "Hillshade_Mean",
        "Hillshade_Range",
        "Soil_Type_Count",
        "Wilderness_Area_Count",
    ]
    all_features = [c for c in df.columns if c != target]

    return {
        "numerical": num_cols,
        "binary": binary_cols,
        "engineered": [e for e in engineered if e in df.columns],
        "all_features": all_features,
    }
