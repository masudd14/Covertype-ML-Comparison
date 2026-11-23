# Data & Model Artifacts

This directory houses the raw dataset, preprocessed data splits, and transformer artifacts.

## Dataset Origin
* **Dataset:** Forest Cover Type from UCI Machine Learning Repository
* **Instances:** 581,012
* **Features:** 54 cartographic and environmental attributes
* **Target:** 7 distinct Forest Cover Types (Integer values 1 to 7)

## Git Tracking & Storage Policies
To adhere to GitHub repository file size limits (< 100 MB per file):
* Raw dataset archives (`covtype.data`, `covtype.csv`) and large split matrices (`*_scaled.csv`, `decomposited_*.csv`) are ignored by Git.
* The pre-fitted lightweight transformers (`scaler.pkl` ~3KB and `pca_model.pkl` ~19KB) are tracked to enable inference and evaluation.
* The tuned ensemble weights (`final_model.pkl` ~637MB) exceed GitHub limits and can be regenerated locally or downloaded from external model storage.

## How to Reproduce
1. Download the raw `covtype.data.gz` from the UCI repository or run `dataset.py`.
2. Execute notebooks sequentially:
   * `01_dataset_understanding.ipynb`
   * `02_eda.ipynb`
   * `03_preprocessing.ipynb` (generates scalers and PCA transforms)
   * `04_baseline_models.ipynb`
   * `05_hyperparameter_tuning.ipynb` (trains and saves `final_model.pkl`)
   * `06_final_evaluation.ipynb`
