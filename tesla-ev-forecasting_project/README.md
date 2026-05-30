# 🚗 Tesla EV Deliveries & Production Forecasting

> A leakage-free, production-grade ML forecasting pipeline — 2015 → 2026

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-F7931E)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## Overview

This project predicts **quarterly Tesla EV delivery volumes** by region and model using
historical production and market data. It is built around a single, non-negotiable principle:
**zero data leakage**.

A naive version of this problem using `Production_Units` as a feature yields **R² = 1.0000**
because production and delivery figures are reported together in Tesla's earnings releases — making
`Production_Units` a near-perfect proxy for the target. This project identifies and eliminates
every such source of leakage before a single model is trained.

---

## Project Structure

```
tesla-ev-forecasting/
│
├── notebooks/
│   └── tesla_ev_forecasting.ipynb   # Main notebook — full pipeline end to end
│
├── src/
│   └── features.py                  # Leakage-free feature engineering (importable)
│
├── data/
│   └── .gitkeep                     # Dataset downloaded via kagglehub at runtime
│
├── outputs/
│   └── .gitkeep                     # Saved plots / model artefacts (gitignored)
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/tesla-ev-forecasting.git
cd tesla-ev-forecasting
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up Kaggle credentials
Place your `kaggle.json` in `~/.kaggle/` (or export env vars):
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

### 4. Run the notebook
```bash
jupyter notebook notebooks/tesla_ev_forecasting.ipynb
```

---

## Dataset

**[Tesla EA Deliveries and Production Data 2015-2025](https://www.kaggle.com/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025)**
on Kaggle — downloaded automatically via `kagglehub` when the notebook runs.

---

## Pipeline Summary

| Stage | What happens |
|---|---|
| **Leakage Audit** | `Production_Units` dropped; `Delivery_Efficiency` and `Production_Surplus` never engineered |
| **Feature Engineering** | Price-per-km, charging impact, quarter, Is_Q4, per-group lag (shift 1 & 2), rolling averages |
| **Train / Test Split** | Chronological cutoff — train ≤ 2023, test 2024–2025 (no random shuffle) |
| **Cross-Validation** | `TimeSeriesSplit(n_splits=5)` — temporal ordering preserved |
| **Models** | Ridge (baseline), Random Forest, XGBoost |
| **Tuning** | `GridSearchCV` over XGBoost depth, learning rate, ensemble size, subsample |
| **Forecast** | Last-known-state per (Region, Model) series → lag rollforward → out-of-sample 2026 |

---

## Data Leakage — Explained

### What was wrong in the naive version

| Problem | Effect |
|---|---|
| `Production_Units` included as a feature | R² jumps to 1.0000 — model is just reading the answer |
| `Delivery_Efficiency = Deliveries / Production` | Encodes target in numerator |
| `Production_Surplus = Production − Deliveries` | Encodes target directly |
| Lag features computed on full dataset before split | Future rows visible in training lags |
| Random `train_test_split` on time-series data | 2024 data trains the model, 2022 data tests it |
| Standard `KFold` cross-validation | Folds allow future data in training windows |
| 2026 forecast = `df.copy()` with `Year=2026` | Not a forecast — just re-predicts known history |

### What this project does instead

- Drops `Production_Units` immediately after loading
- Never engineers `Delivery_Efficiency` or `Production_Surplus`
- Computes all lags **per `(Region, Model)` group** using `.shift(1)` after chronological sort
- Uses a **hard chronological split** at year 2023
- Uses `TimeSeriesSplit` for all cross-validation
- Builds the 2026 forecast by rolling the last known state forward

---

## Results

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| Ridge Regression | *run notebook* | *run notebook* | *run notebook* | *run notebook* |
| Random Forest | *run notebook* | *run notebook* | *run notebook* | *run notebook* |
| XGBoost | *run notebook* | *run notebook* | *run notebook* | *run notebook* |
| **XGBoost (Tuned)** ✅ | **best** | **lowest** | **lowest** | **lowest** |

> Results depend on the dataset version downloaded from Kaggle. Run the notebook to get live numbers.

---

## Key Findings

- **Lag features** (`Lag1_Deliveries`, `Rolling3_Deliveries`) consistently top the feature
  importance chart — recent delivery momentum is the single strongest signal.
- **`Is_Q4`** confirms Tesla's well-documented end-of-year delivery surge.
- **Tuned XGBoost** outperforms Ridge and Random Forest across all four metrics.

---

## Potential Improvements

- Add macroeconomic features (interest rates, EV subsidies, fuel prices)
- Add competitor delivery data as a market-share signal
- Try LightGBM or CatBoost for better native categorical handling
- Replace grid search with Optuna / Bayesian optimisation
- Use walk-forward (expanding-window) validation for the strictest time-series evaluation
- Scrape Tesla shareholder letters for production guidance as a leading indicator

---

## Requirements

See `requirements.txt`. Key packages:

```
numpy, pandas, matplotlib, seaborn
scikit-learn, xgboost
kagglehub, jupyter
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

*Built by Siddharth Sharma · May 2026*
