# Project 4: Liver Disease Multi-Class Classification & Hyperparameter Tuning Pipeline

## Overview & Problem Statement
Early detection and accurate differential diagnosis of hepatic conditions are critical for clinical intervention and liver disease management. 

This project establishes a comprehensive clinical data science workflow designed to diagnose and categorize liver pathology into 5 distinct clinical stages using patient biomarkers, lifestyle indicators, and demographic profiles. The pipeline incorporates advanced categorical encoding, multi-class class balancing via SMOTE, and an exhaustive cross-validated hyperparameter optimization grid.

---

## Dataset Description
- **Source**: Clinical Liver Disease Research Dataset (`Training_Liver_Disease_Dataset.csv`).
- **File**: `Training_Liver_Disease_Dataset.csv`
- **Volume**: 30,000 patient records, 33 columns.
- **Target Variable**: `Liver_Disease_Class` — Multi-class categorical diagnosis spanning 5 clinical cohorts:
  1. `Healthy Liver`
  2. `Fatty Liver Disease (NAFLD)`
  3. `Alcoholic Liver Disease`
  4. `General Liver Disease Severity`
  5. `Liver Cirrhosis Risk`
- **Key Predictor Attributes**:
  - **Demographics & Physical Profile**: `Age`, `Gender`, `Occupation`, `BMI`, `Obesity_Class`, `Waist_Circumference`.
  - **Lifestyle & Habits**: `Diet_Quality`, `Physical_Activity`, `Alcohol_Consumption`, `Smoking_Status`.
  - **Clinical Biomarkers & Vitals**: Blood pressure, cholesterol levels, glucose, AST, ALT, total bilirubin, albumin, and alkaline phosphatase.
  - **Medical History**: `Medication_History`, genetic risk markers, and comorbid conditions.

---

## Methodology & Machine Learning Architecture

### 1. Robust Data Preprocessing
- **Missing Value Handling**: Statistical imputation preventing data loss across continuous biochemical tests.
- **One-Hot Encoding**: Applied to nominal clinical attributes (`Smoking_Status`, `Medication_History`, `Occupation`).
- **Target Encoding**: `LabelEncoder` transforming the 5 diagnostic classes into ordinal numerical targets ($y \in [0, 4]$).
- **Feature Normalization**: Utilized `StandardScaler` to bring disparate clinical scales (e.g. enzyme units vs age in years) into zero-mean, unit-variance distributions.

### 2. Multi-Class Resampling via SMOTE
- The raw distribution across disease stages suffered from substantial class disparities.
- Employed multi-class **SMOTE** over-sampling on the normalized training partition, synthesizing samples until all 5 disease categories achieved uniform parity (10,762 instances per class, totaling 53,810 balanced training samples).

### 3. Systematic Hyperparameter Tuning Grid
Implemented an extensible evaluation framework using `GridSearchCV` with 3-fold stratified cross-validation (`cv=3`, `n_jobs=-1`):
- **Random Forest**: Exploring `n_estimators` $\in [100, 200, 300, 400]$, `max_depth` $\in [5, 10, 15, 20]$, and `min_samples_split` $\in [2, 5, 10]$.
- **Logistic Regression**: Regularization inverse penalty $C \in [0.1, 1, 10]$.
- **Gradient Boosting**: Tuned `learning_rate` $\in [0.01, 0.1, 0.2]$ and tree depth.
- **K-Nearest Neighbors (KNN)**: Neighborhood sizes $k \in [5, 9, 11, 13, 15, 17]$.

---

## Evaluation Framework & Metrics
Models are assessed across multi-class weighted metrics:
- **Weighted Precision**: Measures positive predictive value across disease stages.
- **Weighted Recall (Sensitivity)**: Critical for clinical diagnostics to minimize false negatives in advanced pathology (e.g. Cirrhosis).
- **Weighted F1-Score**: Harmonic mean evaluating balanced diagnostic precision.
- **Multi-Class Accuracy**: Overall exact stage match across holdout patient cohorts.

---

## Quickstart Guide

### 1. Environment Setup
Navigate to the project folder:
```bash
cd 04-liver-disease-multiclass-classification
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Running the Pipeline
Launch Jupyter Notebook:
```bash
jupyter notebook DP.ipynb
```
Execute the notebook cells to run exploratory data analysis, SMOTE balancing, and the grid search optimization module.
