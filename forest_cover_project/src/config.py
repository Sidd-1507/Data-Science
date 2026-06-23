"""
config.py
---------
Central configuration: paths, constants, model hyperparameters.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

TRAIN_PATH  = os.path.join(DATA_DIR, "train.csv")
TEST_PATH   = os.path.join(DATA_DIR, "test.csv")

# Kaggle path (used inside Kaggle kernels)
KAGGLE_DATA_PATH = "/kaggle/input/datasets/aswathimp/forest26-types/train.csv"

# ── Target ────────────────────────────────────────────────────────────────────
TARGET_COL = "Cover_Type"

COVER_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

# ── Split settings ────────────────────────────────────────────────────────────
TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

# ── Feature selection threshold ───────────────────────────────────────────────
FEATURE_IMPORTANCE_THRESHOLD = 0.05

# ── Model defaults ────────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "random_forest": {
        "n_estimators": 200,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
    "xgboost": {
        "n_estimators": 200,
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbosity": 0,
    },
    "lightgbm": {
        "n_estimators": 200,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    },
}

# ── Hyperparameter search spaces ──────────────────────────────────────────────
LGBM_PARAM_DIST = {
    "n_estimators":    (100, 500),   # randint range
    "max_depth":       (4, 12),
    "num_leaves":      (20, 150),
    "learning_rate":   (0.01, 0.21), # uniform range
    "subsample":       (0.6, 1.0),
    "colsample_bytree":(0.6, 1.0),
    "reg_alpha":       (0.0, 1.0),
    "reg_lambda":      (0.0, 1.0),
}

XGB_PARAM_DIST = {
    "n_estimators":    (100, 400),
    "max_depth":       (3, 10),
    "learning_rate":   (0.01, 0.21),
    "subsample":       (0.6, 1.0),
    "colsample_bytree":(0.6, 1.0),
    "reg_alpha":       (0.0, 1.0),
    "reg_lambda":      (0.0, 1.0),
}

TUNING_N_ITER = 30
