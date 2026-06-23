"""
predict.py
----------
Run inference on new / unseen data using saved model artifacts.

Usage
-----
    python src/predict.py --data_path data/test.csv --output_dir outputs/
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import COVER_NAMES, TARGET_COL
from src.feature_engineering import engineer_features


def load_artifacts(output_dir: str):
    """Load all saved model artifacts from disk."""
    model     = joblib.load(os.path.join(output_dir, "best_model.pkl"))
    le        = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))
    scaler    = joblib.load(os.path.join(output_dir, "scaler.pkl"))
    features  = joblib.load(os.path.join(output_dir, "selected_features.pkl"))
    return model, le, scaler, features


def predict(data_path: str, output_dir: str, save_csv: bool = True) -> pd.DataFrame:
    """
    Generate predictions for new data.

    Parameters
    ----------
    data_path  : path to CSV file (no Cover_Type column required)
    output_dir : directory containing saved model artifacts
    save_csv   : if True, write predictions to outputs/predictions.csv

    Returns
    -------
    pd.DataFrame with columns [Cover_Type_Pred, Cover_Type_Name]
    """
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Input shape: {df.shape}")

    # Drop target if accidentally included
    if TARGET_COL in df.columns:
        df = df.drop(TARGET_COL, axis=1)

    # Feature engineering
    df_eng = engineer_features(df)

    # Load artifacts
    print(f"Loading artifacts from: {output_dir}")
    model, le, scaler, selected_features = load_artifacts(output_dir)

    # Align columns
    missing = [f for f in selected_features if f not in df_eng.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    X = df_eng[selected_features].values

    # Predict
    y_pred_enc = model.predict(X)
    y_pred     = le.inverse_transform(y_pred_enc)

    results = pd.DataFrame({
        "Cover_Type_Pred": y_pred,
        "Cover_Type_Name": [COVER_NAMES.get(p, "Unknown") for p in y_pred],
    })

    if save_csv:
        out_path = os.path.join(output_dir, "predictions.csv")
        results.to_csv(out_path, index=False)
        print(f"✅  Predictions saved to: {out_path}")

    print("\nPrediction Distribution:")
    vc = results["Cover_Type_Name"].value_counts()
    for name, count in vc.items():
        print(f"  {name:<25} : {count}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Forest Cover Type")
    parser.add_argument("--data_path",  required=True, help="Path to input CSV")
    parser.add_argument("--output_dir", default="outputs/", help="Directory with model artifacts")
    args = parser.parse_args()
    predict(args.data_path, args.output_dir)
