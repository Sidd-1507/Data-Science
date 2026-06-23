"""
tests/test_feature_engineering.py
----------------------------------
Unit tests for feature_engineering.py
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_engineering import engineer_features, get_feature_groups


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe matching dataset column schema."""
    np.random.seed(42)
    n = 50
    soil_cols = {f"Soil_Type{i}": np.random.randint(0, 2, n) for i in range(1, 41)}
    wild_cols = {f"Wilderness_Area{i}": np.random.randint(0, 2, n) for i in range(1, 5)}

    base = {
        "Elevation":                          np.random.randint(1800, 3800, n),
        "Aspect":                             np.random.randint(0, 360, n),
        "Slope":                              np.random.randint(0, 60, n),
        "Horizontal_Distance_To_Hydrology":   np.random.randint(0, 1500, n),
        "Vertical_Distance_To_Hydrology":     np.random.randint(-200, 600, n),
        "Horizontal_Distance_To_Roadways":    np.random.randint(0, 7000, n),
        "Hillshade_9am":                      np.random.randint(0, 255, n),
        "Hillshade_Noon":                     np.random.randint(0, 255, n),
        "Hillshade_3pm":                      np.random.randint(0, 255, n),
        "Horizontal_Distance_To_Fire_Points": np.random.randint(0, 7000, n),
        "Cover_Type":                         np.random.randint(1, 8, n),
    }
    return pd.DataFrame({**base, **wild_cols, **soil_cols})


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_engineer_features_returns_dataframe(sample_df):
    result = engineer_features(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_engineer_features_adds_columns(sample_df):
    result = engineer_features(sample_df)
    expected_new = [
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
    for col in expected_new:
        assert col in result.columns, f"Missing column: {col}"


def test_engineer_features_no_original_columns_dropped(sample_df):
    result = engineer_features(sample_df)
    for col in sample_df.columns:
        assert col in result.columns, f"Original column dropped: {col}"


def test_engineer_features_does_not_mutate_input(sample_df):
    original_cols = list(sample_df.columns)
    _ = engineer_features(sample_df)
    assert list(sample_df.columns) == original_cols


def test_euclidean_distance_non_negative(sample_df):
    result = engineer_features(sample_df)
    assert (result["Euclidean_Distance_Hydrology"] >= 0).all()


def test_log_features_non_negative(sample_df):
    result = engineer_features(sample_df)
    for col in [
        "Horizontal_Distance_To_Fire_Points_Log",
        "Horizontal_Distance_To_Hydrology_Log",
        "Horizontal_Distance_To_Roadways_Log",
    ]:
        assert (result[col] >= 0).all(), f"{col} contains negative values"


def test_hillshade_range_non_negative(sample_df):
    result = engineer_features(sample_df)
    assert (result["Hillshade_Range"] >= 0).all()


def test_soil_type_count_valid_range(sample_df):
    result = engineer_features(sample_df)
    assert result["Soil_Type_Count"].between(0, 40).all()


def test_wilderness_area_count_valid_range(sample_df):
    result = engineer_features(sample_df)
    assert result["Wilderness_Area_Count"].between(0, 4).all()


def test_get_feature_groups_returns_dict(sample_df):
    df_eng = engineer_features(sample_df)
    groups = get_feature_groups(df_eng)
    assert isinstance(groups, dict)
    assert "numerical" in groups
    assert "binary" in groups
    assert "engineered" in groups
    assert "all_features" in groups


def test_get_feature_groups_no_target_in_features(sample_df):
    df_eng = engineer_features(sample_df)
    groups = get_feature_groups(df_eng)
    for key, cols in groups.items():
        assert "Cover_Type" not in cols, f"Target found in group '{key}'"
