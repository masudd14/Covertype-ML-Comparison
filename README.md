<div align="center">

# 🌲 Forest Covertype: Comparative Machine Learning Benchmark

**A Rigorous Empirical Comparison of Classical Supervised Classification Algorithms on the 581k-Sample UCI Forest Covertype Dataset**

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![CI Status](https://img.shields.io/badge/CI-Passing-brightgreen?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/masudd14/Covertype-ML-Comparison/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Table of Contents

- [Executive Overview](#-executive-overview)
- [Dataset Architecture & Properties](#-dataset-architecture--properties)
  - [Feature Breakdown](#feature-breakdown)
  - [Target Classes & Imbalance Analysis](#target-classes--imbalance-analysis)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
  - [Pipeline Flowchart](#pipeline-flowchart)
  - [Preprocessing & Dimensionality Reduction](#preprocessing--dimensionality-reduction)
- [Algorithms Benchmarked](#-algorithms-benchmarked)
- [Experimental Results & Evaluation](#-experimental-results--evaluation)
  - [Baseline Model Comparison](#baseline-model-comparison)
  - [Final Model Performance (94% Accuracy)](#final-model-performance-94-accuracy)
  - [Class-by-Class Classification Report](#class-by-class-classification-report)
- [Repository Structure](#-repository-structure)
- [Quickstart & Execution Guide](#-quickstart--execution-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Running Unit Tests](#2-running-unit-tests)
  - [3. Sequential Notebook Walkthrough](#3-sequential-notebook-walkthrough)
- [Model Weights & Reproducibility](#-model-weights--reproducibility)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🌟 Executive Overview

Predicting forest vegetation cover from cartographic observations is a vital problem in computational ecology, forest management, and biodiversity preservation. Rather than relying on costly manual field surveys or satellite imagery prone to weather occlusion, this project explores the predictive power of **54 cartographic and geological variables** to classify forest cover types across northern Colorado.

This repository conducts an end-to-end, empirical study evaluating **8 distinct supervised learning algorithms**:
* **Ensembles:** Random Forest, Gradient Boosting, AdaBoost
* **Instance-based & Linear:** K-Nearest Neighbors (KNN), Logistic Regression
* **Support Vector & Probabilistic:** Linear Support Vector Classifier (LinearSVC), Gaussian Naive Bayes
* **Decision Trees:** DecisionTreeClassifier

### 🏆 Key Project Milestones
* **94% Accuracy & 0.94 Weighted F1-Score** achieved on an independent holdout test set of **87,152 samples**.
* Dimensionality reduction using **Principal Component Analysis (PCA)** to accelerate convergence and reduce multi-collinearity.
* Robust mitigation of extreme class imbalance (Class 4 comprises < 0.5% vs Class 2 > 48%) using weighted sample strategies and ensemble bagging.
* Modular, production-grade Python architecture (`src/` package) with automated GitHub Actions CI testing.

---

## 📊 Dataset Architecture & Properties

The research uses the **UCI Forest Covertype Dataset**, originating from the US Forest Service (USFS) Region 2 Resource Information System (RIS).

* **Observation Unit:** $30 \times 30\text{ meter}$ land cells.
* **Total Instances:** 581,012 samples.
* **Total Attributes:** 54 non-target features (10 continuous, 44 binary).

### Feature Breakdown

| Feature Category | Count | Data Type | Description |
| :--- | :---: | :--- | :--- |
| **Topographic / Physical** | 3 | Continuous (Meters / Degrees) | `Elevation` (elevation in meters), `Aspect` (aspect in degrees azimuth), `Slope` (slope in degrees) |
| **Hydrological Distances** | 2 | Continuous (Meters) | `Horizontal_Distance_To_Hydrology`, `Vertical_Distance_To_Hydrology` |
| **Infrastructure Distances**| 2 | Continuous (Meters) | `Horizontal_Distance_To_Roadways`, `Horizontal_Distance_To_Fire_Points` |
| **Hillshade Measurements** | 3 | Continuous (0 to 255 Index) | `Hillshade_9am`, `Hillshade_Noon`, `Hillshade_3pm` |
| **Wilderness Areas** | 4 | Binary (0 / 1 One-Hot) | 4 distinct areas in Roosevelt National Forest: Rawah, Neota, Comanche Peak, Cache la Poudre |
| **Soil Types** | 40 | Binary (0 / 1 One-Hot) | USFS Ecological Land Unit (ELU) soil classification codes |

### Target Classes & Imbalance Analysis

The target variable is `Cover_Type` (integers 1 to 7), presenting a severe class distribution skew:

```text
Class 1: Spruce/Fir             ████████████████████  (36.46%) - 211,840 samples
Class 2: Lodgepole Pine         ██████████████████████████  (48.76%) - 283,301 samples
Class 3: Ponderosa Pine         ███  (6.15%) - 35,754 samples
Class 4: Cottonwood/Willow      ▏  (0.47%) - 2,747 samples  <-- Extreme minority
Class 5: Aspen                  ▏  (1.63%) - 9,493 samples
Class 6: Douglas-fir            █  (2.99%) - 17,367 samples
Class 7: Krummholz              █  (3.53%) - 20,510 samples
```

---

## 🔬 Machine Learning Pipeline

### Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────┐
│                 UCI Covertype Dataset                       │
│                  581,012 Rows × 54 Features                 │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Data Understanding & Profiling (Notebook 01)             │
│    - Schema validation, 0 missing values verified           │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Exploratory Data Analysis & Feature Distributions (NB 02)│
│    - Elevation & soil type correlations, multi-collinearity │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Preprocessing & Scaling (Notebook 03)                    │
│    - Stratified Train / Validation / Test Splitting         │
│    - StandardScaler normalization (saved as scaler.pkl)     │
│    - PCA Dimensionality Reduction (saved as pca_model.pkl)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Baseline Benchmark Evaluation (Notebook 04)              │
│    - 8 Classification algorithms evaluated under default hp │
│    - Top 3 selected: Random Forest, Gradient Boosting, KNN  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. Hyperparameter Tuning (Notebook 05)                      │
│    - Systematic search over n_estimators, max_depth, etc.   │
│    - Champion Model: Tuned Random Forest (final_model.pkl)  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Final Evaluation & Error Analysis (Notebook 06)          │
│    - Holdout Test Set (87,152 samples)                      │
│    - Accuracy: 94% | Weighted F1: 0.94 | Macro F1: 0.90     │
└─────────────────────────────────────────────────────────────┘
```

### Preprocessing & Dimensionality Reduction

1. **StandardScaler:** Normalizes continuous variables with disparate scales (e.g., `Elevation` in thousands of meters vs `Slope` in degrees).
2. **Principal Component Analysis (PCA):** Compresses high-dimensional one-hot soil encodings and continuous measurements into orthogonal components, retaining maximum cumulative variance while significantly speeding up distance-based computation (such as KNN).

---

## 🧪 Algorithms Benchmarked

| Algorithm | Model Family | Key Strengths in This Experiment |
| :--- | :--- | :--- |
| **Random Forest** | Bagging Ensemble | Handles non-linear splits, soil interactions, and resistant to overfitting. Best overall performer. |
| **Gradient Boosting** | Boosting Ensemble | Sequentially corrects residual pseudo-errors, producing sharp decision boundaries. |
| **K-Nearest Neighbors** | Instance-Based Metric | Benefits strongly from standardized feature spaces and PCA compression. |
| **Decision Tree** | Single Tree | High interpretability; provides initial baseline for ensemble depth. |
| **AdaBoost** | Boosting Ensemble | Evaluated with sample-weight balancing to elevate penalty on minority forest types. |
| **Logistic Regression** | Linear ($L_2$ Regularized) | Fast convex optimization baseline; serves as lower bound for non-linear models. |
| **Linear SVC** | Support Vector Machine | Evaluates maximum margin linear hyperplanes in multi-class One-vs-Rest configuration. |
| **Gaussian Naive Bayes** | Probabilistic | Assumes feature conditional independence; fast but limited by correlation between attributes. |

---

## 📈 Experimental Results & Evaluation

### Baseline Model Comparison

During the baseline stage (Notebook 04), models were trained on decomposed training data and evaluated against the validation split:

* **Top Tier:** Random Forest and Gradient Boosting emerged as the dominant algorithms, exhibiting high recall across both majority and minority classes.
* **Mid Tier:** K-Nearest Neighbors demonstrated solid performance when operating on PCA-reduced representations.
* **Linear / Simpler Tier:** Logistic Regression and Linear SVC struggled with non-linear geological boundaries, confirming the necessity of tree ensembles.

---

### Final Model Performance (94% Accuracy)

The champion model (Tuned Random Forest) was evaluated on the **completely unseen holdout test split** of **87,152 samples**:

| Metric | Score |
| :--- | :---: |
| **Overall Accuracy** | **94.0%** |
| **Weighted Average Precision** | **0.94** |
| **Weighted Average Recall** | **0.94** |
| **Weighted Average F1-Score** | **0.94** |
| **Macro Average F1-Score** | **0.90** |

---

### Class-by-Class Classification Report

```text
              Precision    Recall    F1-Score    Test Support
--------------------------------------------------------------
  Class 1        0.95       0.93       0.94          31,776
  Class 2        0.94       0.96       0.95          42,495
  Class 3        0.92       0.94       0.93           5,363
  Class 4        0.89       0.77       0.82             412
  Class 5        0.93       0.76       0.83           1,424
  Class 6        0.88       0.85       0.87           2,605
  Class 7        0.96       0.93       0.95           3,077
--------------------------------------------------------------
  Accuracy                             0.94          87,152
  Macro Avg      0.92       0.88       0.90          87,152
  Weighted Avg   0.94       0.94       0.94          87,152
```

> **Insight:** Even on the most extreme minority class (Class 4 with only 412 test instances), the model maintains an **89% precision** and **0.82 F1-score**, proving excellent generalization without sacrificing minority detection.

---

## 📁 Repository Structure

```text
Covertype-ML-Comparison/
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI workflow (Python 3.10 & 3.11)
├── DATA/
│   └── covertype/
│       ├── README.md                  # Documentation on data splits & reproducibility
│       ├── covtype.info               # Original USFS UCI attribute codebook
│       ├── dataset.py                 # Feature mapping script
│       ├── pca_model.pkl              # Fitted PCA transformer artifact (19 KB)
│       └── scaler.pkl                 # Fitted StandardScaler transformer artifact (3 KB)
├── notebooks/                         # Sequentially numbered, fully reproducible research
│   ├── 01_dataset_understanding.ipynb # Feature definitions, schema validation, null audits
│   ├── 02_eda.ipynb                   # Distribution plots, correlations, hillshade analysis
│   ├── 03_preprocessing.ipynb         # Splitting, StandardScaler, and PCA pipeline
│   ├── 04_baseline_models.ipynb       # 8-model baseline comparison & metrics benchmark
│   ├── 05_hyperparameter_tuning.ipynb # Random Search / Grid tuning for champion selection
│   └── 06_final_evaluation.ipynb      # Final test inference, report & confusion matrices
├── src/                               # Clean, importable Python package
│   ├── __init__.py                    # Package initialization
│   ├── data_loader.py                 # Schema constants and robust CSV ingestion
│   ├── preprocessing.py               # Transformer loading and pipeline transforms
│   └── evaluate.py                    # Metric calculators & classification reports
├── tests/                             # Unit tests
│   ├── __init__.py
│   └── test_pipeline.py               # Automated schema & metric assertion tests
├── .gitignore                         # Strict exclusion of heavy binary files (>100MB)
├── CONTRIBUTING.md                    # Guidelines for opening issues and pull requests
├── LICENSE                            # MIT Open Source License
├── pyproject.toml                     # Standard Python packaging configuration
├── README.md                          # Repository documentation
└── requirements.txt                   # Locked project dependencies
```

---

## 🚀 Quickstart & Execution Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/masudd14/Covertype-ML-Comparison.git
cd Covertype-ML-Comparison

# Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Running Unit Tests

Verify the environment and pipeline components:

```bash
python -m unittest discover -s tests
```

### 3. Sequential Notebook Walkthrough

Start Jupyter Notebook or JupyterLab:

```bash
jupyter notebook notebooks/
```

Follow the numbered notebooks from `01` through `06` to reproduce every experimental step from scratch.

---

## 💾 Model Weights & Reproducibility

> [!NOTE]
> To comply with GitHub's 100 MB file limit, the large model weight file (`final_model.pkl`, ~637 MB) and large CSV data dumps (`*.csv`, `*.data`) are excluded via `.gitignore`.
> * The lightweight transformers (`scaler.pkl` at 3 KB and `pca_model.pkl` at 19 KB) are checked into the repository to allow inference pipelines.
> * To reproduce `final_model.pkl`, simply execute `notebooks/05_hyperparameter_tuning.ipynb` on your machine.

---

## 📜 License & Acknowledgments

* **Dataset Authors:** Jock A. Blackard, Denis J. Dean, and Charles W. Anderson, Colorado State University & USFS Region 2.
* **Repository License:** This project is open-source under the [MIT License](LICENSE).
