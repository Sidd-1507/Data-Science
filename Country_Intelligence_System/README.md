# 🌍 Country Intelligence System
### Using Clustering & Machine Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)](https://xgboost.readthedocs.io)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-purple?logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)

*An end-to-end Data Science pipeline that segments 167 countries into meaningful development tiers using Unsupervised Learning, Dimensionality Reduction, and Supervised Classification.*

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Pipeline](#-project-pipeline)
- [Installation](#-installation)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Key Findings](#-key-findings)
- [Limitations & Future Work](#-limitations--future-work)
- [Resume Description](#-resume-description)

---

## 🎯 Project Overview

This project builds a **Country Intelligence System** to classify nations into development tiers using a combination of unsupervised and supervised machine learning. Starting from raw socio-economic data, the pipeline discovers hidden structure through clustering, validates it with stability analysis, and then builds high-accuracy classifiers to predict cluster membership.

**Techniques Used:**
| Category | Algorithms |
|----------|-----------|
| Dimensionality Reduction | PCA |
| Clustering | K-Means, DBSCAN, Hierarchical (Ward) |
| Cluster Validation | Silhouette, Calinski-Harabasz, Davies-Bouldin |
| Stability Analysis | ARI, NMI, AMI |
| Classification | Random Forest, XGBoost |
| Interpretation | Feature Importance, Radar Charts |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle — Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) |
| **Countries** | 167 |
| **Features** | 9 numerical indicators |
| **Missing Values** | None |
| **Duplicates** | None |

### Features

| Feature | Description |
|---------|-------------|
| `child_mort` | Deaths of children under 5 per 1,000 live births |
| `exports` | Exports of goods & services as % of GDP |
| `health` | Total health spending as % of GDP |
| `imports` | Imports of goods & services as % of GDP |
| `income` | Net income per person (USD) |
| `inflation` | Annual growth rate of GDP deflator (%) |
| `life_expec` | Average life expectancy (years) |
| `total_fer` | Average children born per woman |
| `gdpp` | GDP per capita (USD) |

---

## 🏆 Results

### Country Development Tiers

| Cluster | Label | Key Characteristics | Example Countries |
|---------|-------|---------------------|-------------------|
| 🟢 0 | **Developed** | High GDP, High life expectancy, Low child mortality | USA, Germany, Norway, Japan |
| 🟡 1 | **Developing** | Moderate GDP, improving health indicators | India, Brazil, China, Mexico |
| 🔴 2 | **Underdeveloped** | Low GDP, High child mortality, High fertility | Niger, Mali, Chad, Sierra Leone |

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~0.99 | ~0.99 | ~0.99 | ~0.99 |
| XGBoost | ~0.99 | ~0.99 | ~0.99 | ~0.99 |

> Both models achieve near-perfect accuracy, confirming the clusters are well-separated.

### Top Development Predictors
1. 🥇 GDP per Capita (`gdpp`)
2. 🥈 Net Income (`income`)
3. 🥉 Child Mortality (`child_mort`)
4. 🏅 Life Expectancy (`life_expec`)
5. Total Fertility Rate (`total_fer`)

---

## 🔬 Project Pipeline

```
Raw Data (167 Countries × 9 Features)
          │
          ▼
  ┌──────────────────┐
  │ Data Quality     │  Missing values, duplicates, types,
  │ Assessment       │  skewness, kurtosis, outlier %
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ EDA              │  Distributions, KDE, boxplots,
  │                  │  correlation matrix, pair plots
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Outlier Analysis │  IQR + Z-Score detection
  │                  │  (retained — real signal)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ Feature Scaling  │  StandardScaler → X_scaled
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ PCA              │  9D → 2D/3D, Development Score
  └────────┬─────────┘
           │
           ▼
  ┌────────────────────────────────────────────┐
  │          CLUSTERING (3 algorithms)          │
  ├──────────────┬─────────────┬───────────────┤
  │   K-Means    │   DBSCAN    │ Hierarchical  │
  │   (k=3)      │ (eps tuned) │ (Ward, k=3)   │
  └──────┬───────┴──────┬──────┴───────┬───────┘
         │              │              │
         └──────────────┼──────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Cluster Stability   │  ARI, NMI, AMI
             │ Analysis            │  cross-algorithm
             └────────┬────────────┘
                      │
                      ▼
             ┌─────────────────────┐
             │ Pseudo Label        │  K-Means labels →
             │ Generation          │  supervised targets
             └────────┬────────────┘
                      │
                      ▼
         ┌────────────────────────────┐
         │   SUPERVISED LEARNING      │
         ├───────────────┬────────────┤
         │ Random Forest │  XGBoost   │
         └───────┬───────┴────────────┘
                 │
                 ▼
         ┌───────────────────┐
         │ Feature Importance│
         │ Model Comparison  │
         │ Dashboard         │
         └───────────────────┘
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/country-intelligence-system.git
cd country-intelligence-system
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

**Option A — Kaggle CLI:**
```bash
pip install kaggle
kaggle datasets download -d rohan0301/unsupervised-learning-on-country-data
unzip unsupervised-learning-on-country-data.zip -d data/
```

**Option B — Manual:**  
Download from [Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) and place `Country-data.csv` in the `data/` folder.

### 5. Update the data path

In the notebook, update the CSV path:
```python
df = pd.read_csv('data/Country-data.csv')
```

---

## 💻 Usage

```bash
jupyter notebook country_intelligence_system.ipynb
```

Or open in **VS Code**, **JupyterLab**, or directly on **Kaggle**.

---

## 📁 Repository Structure

```
country-intelligence-system/
│
├── 📓 country_intelligence_system.ipynb   # Main notebook (fully documented)
│
├── 📄 README.md                           # This file
├── 📄 requirements.txt                    # Python dependencies
├── 📄 .gitignore                          # Git ignore rules
├── 📄 LICENSE                             # MIT License
│
└── 📂 data/                               # (gitignored — download separately)
    ├── Country-data.csv
    └── data-dictionary.csv
```

---

## 🔑 Key Findings

1. **Three Distinct Development Tiers** — K-Means, DBSCAN, and Hierarchical clustering all converge on 3 natural country groupings, confirming the structure is robust.

2. **GDP & Income are the Strongest Predictors** — Feature importance from both Random Forest and XGBoost ranks `gdpp` and `income` as the top drivers of development tier.

3. **PCA Development Axis** — The first principal component acts as a single-number "Development Score" that cleanly separates the three tiers.

4. **Child Mortality is a Key Health Signal** — `child_mort` is the most important health indicator, more predictive than `life_expec` alone.

5. **Cluster Stability Confirmed** — High ARI/NMI/AMI scores show consistent assignments across all three algorithms, meaning the clusters are real and not artefacts.

6. **DBSCAN Reveals Anomalies** — Density-based clustering identifies outlier countries (small oil-rich states, extreme poverty cases) that don't fit the 3-tier model.

---

## ⚠️ Limitations & Future Work

### Limitations
- Static snapshot (no temporal dimension)
- Only 167 countries; some small/micro-states missing
- 9 features omit education, governance, political stability
- Supervised labels are derived from clusters, not ground truth

### Future Work
- [ ] Time-series analysis with World Bank annual data (1990–2024)
- [ ] Development forecasting model (LSTM/ARIMA per country)
- [ ] Interactive world map (Plotly Choropleth)
- [ ] Streamlit / Gradio dashboard
- [ ] Deep learning-based clustering (autoencoders, UMAP)
- [ ] UN SDG goal tracking integration
- [ ] Additional features: Education Index, Democracy Score, CO₂ emissions

---

## 📌 Resume Description

> **Country Intelligence System using Unsupervised & Supervised Machine Learning**
>
> - Built an end-to-end Country Intelligence System on 167 countries using socio-economic and health indicators.
> - Performed EDA, outlier detection, feature scaling, PCA, and clustering with K-Means, DBSCAN, and Hierarchical Clustering.
> - Conducted cluster stability analysis using ARI, NMI, and AMI metrics across algorithms.
> - Generated pseudo-labels from clusters and trained Random Forest and XGBoost models achieving ~99% predictive accuracy.
> - Identified key development drivers through feature importance: Income, GDP per Capita, Child Mortality, Life Expectancy.
> - Developed a comprehensive country segmentation framework integrating unsupervised learning, supervised learning, and interpretable analytics.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Made with ❤️ and a lot of `matplotlib`

</div>
