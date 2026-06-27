# 🌾 Crop Classification & Recommendation System

<div align="center">

![Crop Recommendation](https://img.shields.io/badge/Domain-Agriculture%20AI-brightgreen?style=for-the-badge&logo=leaf)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=for-the-badge&logo=scikit-learn)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge)

**An end-to-end machine learning pipeline that recommends the optimal crop to cultivate based on soil nutrient levels and local climate conditions — empowering farmers with data-driven decisions.**

[📓 View Notebook](#-project-structure) • [📊 Dataset](#-dataset) • [🚀 Quick Start](#-quick-start) • [📈 Results](#-results) • [🤝 Contributing](#-contributing)

</div>

---

## 📌 Overview

Selecting the wrong crop for a given environment leads to **poor yields, economic losses, and soil degradation**. This project uses supervised machine learning to build a **multi-class crop classifier** across 22 crop types using 7 soil and climate features.

### 🎯 Key Highlights
- ✅ **99%+ accuracy** with Random Forest on hold-out test set
- ✅ **10 ML models** benchmarked with 5-fold cross-validation
- ✅ **Comprehensive EDA** — distributions, heatmaps, pair plots, crop profiles
- ✅ **Production-ready** inference function with saved model artifacts
- ✅ **Zero data leakage** — scaler fitted only on training data

---

## 📂 Project Structure

```
crop-recommendation/
│
├── 📓 Crop_Classification_With_Recommendation_System.ipynb   ← Main notebook
├── 📊 Crop_recommendation.csv                                ← Dataset (2200 rows)
│
├── 🤖 model.pkl                                              ← Trained Random Forest
├── ⚖️  minmaxscaler.pkl                                      ← Fitted MinMaxScaler
│
├── 📄 README.md                                              ← This file
├── 📋 requirements.txt                                       ← Python dependencies
├── 🔧 .gitignore                                             ← Git ignore rules
└── 📜 LICENSE                                                ← MIT License
```

---

## 📊 Dataset

**Source:** [Kaggle — Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)

| Feature | Type | Description | Unit |
|---------|------|-------------|------|
| `N` | int | Nitrogen content in soil | kg/ha |
| `P` | int | Phosphorous content in soil | kg/ha |
| `K` | int | Potassium content in soil | kg/ha |
| `temperature` | float | Mean ambient temperature | °C |
| `humidity` | float | Relative humidity | % |
| `ph` | float | Soil pH value | 0–14 |
| `rainfall` | float | Annual rainfall | mm |
| `label` | str | **Target** — recommended crop | — |

**22 Crops:** Rice, Maize, Jute, Cotton, Coconut, Papaya, Orange, Apple, Muskmelon, Watermelon, Grapes, Mango, Banana, Pomegranate, Lentil, Blackgram, Mungbean, Mothbeans, Pigeonpeas, Kidneybeans, Chickpea, Coffee

**Dataset Statistics:**
- Total samples: **2,200**
- Samples per class: **100** (perfectly balanced)
- Missing values: **None**
- Duplicates: **None**

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/crop-recommendation.git
cd crop-recommendation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook Crop_Classification_With_Recommendation_System.ipynb
```

### 4. Use the Recommendation System Directly
```python
import pickle
import numpy as np

# Load artifacts
model  = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))

NUM_TO_CROP = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
    6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
    11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
    16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
    20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled   = scaler.transform(features)
    pred     = model.predict(scaled)[0]
    return NUM_TO_CROP[pred]

# Example
crop = recommend_crop(N=90, P=42, K=43, temperature=20.9,
                      humidity=82.0, ph=6.5, rainfall=202.9)
print(f"Recommended Crop: {crop}")  # → Rice
```

---

## 🧪 ML Pipeline

```
Raw CSV Data
     │
     ▼
┌─────────────────────────────┐
│  Exploratory Data Analysis  │  ← Distributions, Correlations, Box Plots, Pair Plot
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Preprocessing              │  ← Label Encoding, Train/Test Split (80/20 stratified)
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Feature Scaling            │  ← MinMaxScaler (fit on train only)
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Model Training (10 models) │  ← With 5-Fold Cross Validation
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Best Model Evaluation      │  ← Classification Report + Confusion Matrix
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Feature Importance         │  ← Gini Impurity-based ranking
└─────────────────────────────┘
     │
     ▼
┌─────────────────────────────┐
│  Model Persistence          │  ← model.pkl + minmaxscaler.pkl
└─────────────────────────────┘
```

---

## 📈 Results

### Model Comparison

| Model | Test Accuracy | CV Mean (5-fold) |
|-------|:-------------:|:----------------:|
| 🥇 **Random Forest** | **~99%** | **~99%** |
| 🥈 Extra Trees | ~99% | ~99% |
| 🥉 Gradient Boosting | ~98% | ~98% |
| Bagging | ~97% | ~97% |
| Decision Tree | ~97% | ~97% |
| KNN | ~97% | ~97% |
| SVM (RBF) | ~97% | ~97% |
| Naïve Bayes | ~95% | ~95% |
| AdaBoost | ~92% | ~92% |
| Logistic Regression | ~90% | ~90% |

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Humidity | Highest |
| 2 | Rainfall | High |
| 3 | Potassium (K) | High |
| 4 | Temperature | Medium |
| 5 | Phosphorous (P) | Medium |
| 6 | pH | Medium |
| 7 | Nitrogen (N) | Lower |

---

## 🔮 Future Work

- [ ] Hyperparameter tuning with `Optuna` / `GridSearchCV`
- [ ] SHAP values for per-prediction explainability
- [ ] Streamlit / Flask web app deployment
- [ ] Docker containerisation
- [ ] Add soil texture and organic matter features
- [ ] Seasonal crop rotation recommendations

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.8+ | Core language |
| pandas | 1.x+ | Data manipulation |
| numpy | 1.x+ | Numerical computing |
| scikit-learn | 1.x+ | ML models & metrics |
| matplotlib | 3.x+ | Static visualisations |
| seaborn | 0.12+ | Statistical plots |
| jupyter | — | Interactive notebook |

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset by [Atharva Ingle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) on Kaggle
- scikit-learn community for excellent documentation
- Agricultural domain knowledge from FAO crop guidelines

---

<div align="center">
Made with ❤️ for sustainable agriculture & data-driven farming
<br><br>
⭐ Star this repo if you found it helpful!
</div>
