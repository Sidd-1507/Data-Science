# 🌲 Forest Cover Type Prediction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

> **Multi-class classification of forest cover types using cartographic variables from the Roosevelt National Forest, Colorado.**

---

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Key Insights](#-key-insights)
- [Author](#-author)

---

## 🎯 Problem Statement

Predict the **forest cover type** (1 of 7 classes) for 30×30 meter patches of land in the Roosevelt National Forest using purely cartographic features — no remote sensing imagery involved.

| Cover Type | Forest Name |
|:---:|---|
| 1 | Spruce / Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood / Willow |
| 5 | Aspen |
| 6 | Douglas-fir |
| 7 | Krummholz |

---

## 📦 Dataset

- **Source:** [Kaggle — Forest Cover Type](https://www.kaggle.com/datasets/aswathimp/forest26-types)
- **Rows:** ~15,000 (train split)
- **Features:** 54 original + 12 engineered = **66 total**
  - 10 continuous numerical features (elevation, slope, hillshade, distances)
  - 4 binary wilderness area indicators
  - 40 binary soil type indicators

---

## 📁 Project Structure

```
forest-cover-type-prediction/
│
├── notebooks/
│   └── forest_cover_type_prediction.ipynb   # Full end-to-end notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py                             # Paths & hyperparameters
│   ├── feature_engineering.py               # Feature creation functions
│   ├── feature_selection.py                 # Selection pipeline
│   ├── train.py                             # Model training & evaluation
│   └── predict.py                           # Inference on new data
│
├── tests/
│   ├── test_feature_engineering.py
│   └── test_train.py
│
├── outputs/                                 # Saved plots & model artifacts
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Target distribution analysis (class imbalance detected for types 4 & 5)
- Per-class feature distributions (histograms, box plots)
- Correlation heatmap
- Wilderness area & soil type frequency analysis
- Pair plots of key numerical features

### 2. Feature Engineering (12 new features)
| Feature | Description |
|---|---|
| `Euclidean_Distance_Hydrology` | √(H² + V²) — true distance to water |
| `Horizontal_Distance_To_*_Log` | Log-transformed distance features |
| `Mean_Distance_To_Amenities` | Average of 3 horizontal distances |
| `Elevation_Aspect_Interaction` | Elevation × cos(Aspect) |
| `Elevation_Slope_Interaction` | Elevation × sin(Slope) |
| `Hillshade_Mean` | Average of 9am, Noon, 3pm hillshade |
| `Hillshade_Range` | Max − Min hillshade (diurnal range) |
| `Soil_Type_Count` | Number of active soil types |
| `Wilderness_Area_Count` | Number of active wilderness areas |

### 3. Feature Selection
Three complementary methods combined into a unified score:
- **ANOVA F-test** — linear separability per feature
- **Mutual Information** — non-linear relationships
- **Random Forest Importance** — embedded method
- **RFECV** — cross-validated recursive elimination for validation

### 4. Models Trained (10 total)
| Category | Models |
|---|---|
| Baseline | Logistic Regression, Naive Bayes |
| Tree-based | Decision Tree, Random Forest, Extra Trees |
| Boosting | AdaBoost, Gradient Boosting, **XGBoost**, **LightGBM** |
| Distance | K-Nearest Neighbours |

### 5. Evaluation
- 5-Fold Stratified Cross-Validation
- Hold-out test set (20%)
- Metrics: Accuracy, F1-Macro, F1-Weighted, per-class F1
- RandomizedSearchCV hyperparameter tuning (30 iterations)

---

## 🏆 Results

| Model | CV Accuracy | Test Accuracy | F1 Macro |
|---|:---:|:---:|:---:|
| **LightGBM (Tuned)** 🏆 | ~0.97 | **~0.97** | **~0.97** |
| LightGBM | ~0.96 | ~0.96 | ~0.96 |
| XGBoost | ~0.96 | ~0.96 | ~0.95 |
| Extra Trees | ~0.95 | ~0.95 | ~0.94 |
| Random Forest | ~0.94 | ~0.94 | ~0.93 |
| Gradient Boosting | ~0.88 | ~0.88 | ~0.87 |
| Decision Tree | ~0.85 | ~0.85 | ~0.84 |
| K-Nearest Neighbours | ~0.82 | ~0.82 | ~0.81 |
| Logistic Regression | ~0.70 | ~0.70 | ~0.68 |
| Naive Bayes | ~0.60 | ~0.60 | ~0.58 |

> **Note:** Exact scores depend on the dataset version. Run the notebook to reproduce.

---

## ⚙️ Installation

### Prerequisites
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/forest-cover-type-prediction.git
cd forest-cover-type-prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Full Notebook
```bash
jupyter notebook notebooks/forest_cover_type_prediction.ipynb
```

### Run Training Script
```bash
python src/train.py --data_path train.csv --output_dir outputs/
```

### Run Prediction on New Data
```bash
python src/predict.py --data_path new_data.csv --model_path outputs/best_model.pkl
```

### Run Tests
```bash
pytest tests/ -v
```

---

## 💡 Key Insights

1. **Elevation is the single most predictive feature** — forest cover types are strongly stratified by altitude.
2. **Distance to hydrology** (both horizontal and vertical combined as Euclidean distance) significantly boosts model performance.
3. **Hillshade range** (diurnal variation) captures sun exposure patterns that correlate with tree species.
4. **LightGBM & XGBoost dominate** — gradient boosting methods outperform all others by a wide margin.
5. **Class imbalance matters** — types 4 (Cottonwood/Willow) and 5 (Aspen) are underrepresented; F1-Macro is the most informative metric.

---

## 🔮 Future Work
- [ ] Stacking ensemble: LightGBM + XGBoost + Extra Trees
- [ ] SMOTE for minority class oversampling
- [ ] TabNet / neural network comparison
- [ ] SHAP explainability analysis
- [ ] Streamlit demo app

---

## 👤 Author

**Siddharth Sharma**  
[![GitHub](https://img.shields.io/badge/GitHub-@siddharth-black?logo=github)](https://github.com/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ using scikit-learn, XGBoost, LightGBM, pandas, and matplotlib.*
