# Credit Card Fraud Detection Using Anomaly Detection Techniques

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Testing](https://img.shields.io/badge/Tests-Pytest-green.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An assessment-ready, end-to-end Machine Learning project developed for the **HCL Technical Assessment**. This system identifies fraudulent and suspicious credit card transactions in extreme class imbalance scenarios using unsupervised and semi-supervised anomaly detection algorithms.

---

## 📌 Executive Summary & Abstract

Credit card fraud costs financial institutions and consumers billions of dollars annually. Unlike conventional classification problems, fraud detection presents two fundamental challenges:
1. **Extreme Class Imbalance**: Fraudulent transactions account for roughly **0.17%** of all transactions (less than 2 in 1,000). A standard classifier predicting "normal" on every transaction achieves **99.83% accuracy** while missing 100% of frauds (the **Accuracy Paradox**).
2. **Zero-Day Attacks**: Fraudsters continually adapt. Supervised models trained exclusively on historical fraud signatures struggle to identify novel fraud methodologies.

This project implements an **Anomaly Detection framework** that models the geometric and density distribution of legitimate cardholder behavior. By prioritizing **Precision-Recall AUC (PR-AUC)**, **Recall**, and **F1-Score** over naive accuracy, the system flags outliers with normalized continuous anomaly scores and provides actionable risk banding with feature-level explanations.

---

## 🎯 Key Objectives

- **Unsupervised Modeling**: Implement Isolation Forest as the primary algorithm, benchmarked against Local Outlier Factor (LOF) and One-Class SVM.
- **Leakage-Free Preprocessing**: Use `RobustScaler` on skewed transaction amounts and timestamps, fitting strictly on training data.
- **Dynamic Decision Engine**: Provide real-time threshold tuning to control the Precision vs. Recall trade-off.
- **Full-Stack Streamlit Dashboard**: Deliver an interactive web dashboard for single-transaction risk scoring, batch CSV processing, and exploratory data analysis.
- **Production-Ready Quality**: Complete unit test suite (`pytest`), automated fallback dataset generation, and clean object-oriented architecture.

---

## 🔬 Technologies Used

| Category | Tools / Libraries |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (Isolation Forest, Local Outlier Factor, One-Class SVM) |
| **Data Visualization** | Matplotlib, Seaborn |
| **Web Application** | Streamlit |
| **Model Persistence** | Joblib |
| **Testing** | Pytest |

---

## 📊 Dataset Details

### Benchmark Dataset
- **Source**: [Kaggle European Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Total Transactions**: 284,807 transactions recorded over 2 days in September 2013 by European cardholders.
- **Total Frauds**: 492 transactions (**0.172% fraud rate**).
- **Features**:
  - `Time`: Seconds elapsed between this transaction and the first transaction in the dataset.
  - `V1` – `V28`: Principal Component Analysis (PCA) transformed features to safeguard cardholder confidentiality.
  - `Amount`: Transaction amount in Euros/Dollars (heavily right-skewed).
  - `Class`: Ground truth label (`0` = Legitimate, `1` = Fraudulent).

### Zero-Failure Demonstration Mode
If `data/creditcard.csv` is not present, the system automatically generates a high-fidelity synthetic dataset (`data/synthetic_creditcard.csv`) matching the exact schema, statistical moments, and correlation structures of the Kaggle benchmark. This ensures the project runs immediately and deterministically for demonstration without requiring a 150MB Kaggle download.

---

## 🧮 Machine Learning Algorithms

```
                          ┌────────────────────────┐
                          │ Credit Card Transaction │
                          └───────────┬────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
             [Amount & Time]                     [V1 - V28]
                     │                                 │
            RobustScaler (IQR)                 Centered (PCA)
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │   Anomaly Detection Models   │
                       ├──────────────────────────────┤
                       │  • Isolation Forest (iTree)  │
                       │  • LOF (Local Density, k-NN) │
                       │  • One-Class SVM (RBF Kernel)│
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │ Continuous Anomaly Score     │
                       │ Normalized in [0.0, 1.0]     │
                       └──────────────┬───────────────┘
                                      │
                                      ▼
                       ┌──────────────────────────────┐
                       │ Decision Threshold Cutoff    │
                       │ (e.g., Threshold = 0.50)     │
                       └──────┬────────────────┬──────┘
                              │                │
                      Score < Threshold  Score >= Threshold
                              │                │
                              ▼                ▼
                       [APPROVED: 0]     [ALERT: 1]
                       (Legitimate)      (Fraud / 2FA)
```

### 1. Isolation Forest (Primary Model)
- **Concept**: Anomalies are rare and structurally distinct from the majority cluster. By recursively partitioning feature space using random splits, anomalies are isolated near the root of isolation trees (shorter average path lengths).
- **Complexity**: $O(n \log n)$ training and $O(n)$ inference, making it the industry standard for high-throughput payment systems.

### 2. Local Outlier Factor (LOF) (Comparison Model)
- **Concept**: Compares the local density of an instance to that of its $k$-nearest neighbors. Outliers reside in substantially sparser regions than their surrounding neighborhood.
- **Novelty Mode**: Configured with `novelty=True` to enable evaluation and scoring on out-of-sample test transactions.

### 3. One-Class SVM (OC-SVM) (Optional Comparison)
- **Concept**: Uses a non-linear Radial Basis Function (RBF) kernel to project features into a higher-dimensional Hilbert space, constructing a maximum-margin hyperplane separating normal transactions from the origin.

---

## 📁 Project Directory Structure

```
Hcl Project/
├── app.py                      # Interactive Streamlit Web Application
├── train.py                    # Standalone model training & evaluation pipeline
├── predict.py                  # CLI & programmatic inference service
├── verify_project.py           # Automated project verification & integrity audit
├── requirements.txt            # Project dependencies
├── README.md                   # Complete architectural documentation
├── INTERVIEW_PREPARATION.md    # 20+ HCL technical interview Q&A guide
├── .gitignore                  # Git ignore rules for models, data, and cache
├── run_app.bat                 # One-click launcher for Streamlit web app
├── run_tests.bat               # One-click launcher for Pytest test suite & coverage
├── train_models.bat            # One-click launcher for retraining models
├── run_predict.bat             # One-click demo for CLI predictions
├── run_verify.bat              # One-click project verification audit
├── data/
│   ├── generate_data.py        # Autonomous synthetic dataset & test batch generator
│   ├── synthetic_creditcard.csv# High-fidelity synthetic dataset (auto-generated)
│   └── batch_test_unlabelled.csv # Demo batch file for upload testing
├── src/
│   ├── __init__.py             # Package initialization
│   ├── dataset.py              # Dataset loading, schema validation, and stratified splitting
│   ├── preprocessing.py        # Leakage-free RobustScaler pipeline & input validators
│   ├── models.py               # Unified wrappers for Isolation Forest, LOF, and OC-SVM
│   └── evaluation.py           # Evaluation metrics, confusion matrix, and ROC/PR plots
├── models/
│   ├── preprocessor.joblib     # Serialized preprocessing pipeline
│   ├── isolation_forest.joblib # Serialized Isolation Forest model
│   ├── lof.joblib              # Serialized LOF model
│   ├── one_class_svm.joblib    # Serialized OC-SVM model
│   ├── evaluation_results.json # Computed empirical test metrics
│   ├── model_comparison.png    # Metric benchmark bar chart
│   └── roc_pr_curves.png       # ROC and PR curves
└── tests/
    ├── __init__.py             # Test package initialization
    ├── test_dataset.py         # Unit tests for schema validation and splitting
    ├── test_preprocessing.py   # Unit tests for scalers and input validation
    ├── test_models.py          # Unit tests for model scoring and output bounds
    ├── test_prediction.py      # Unit tests for single and batch prediction pipelines
    ├── test_evaluation.py      # Unit tests for evaluation metrics & zero-division safety
    └── test_leakage.py         # Unit tests proving zero data leakage across splits
```

---

## ⚡ Installation & Setup Instructions

### 1. Prerequisites
Ensure Python 3.10 or higher is installed.

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Instructions

### Step 1: Train Models & Generate Artifacts
Run the standalone training pipeline. It automatically detects real Kaggle data if placed in `data/creditcard.csv`, or generates synthetic benchmark data on first run:
```bash
python train.py
```
*Optional CLI Flags:*
- `--samples 15000`: Subsample dataset for quick training during live presentations.
- `--contamination 0.002`: Set expected anomaly proportion.
- `--normal-only-train`: Train strictly on legitimate records (`Class == 0`).

### Step 2: Run Unit Tests
Verify pipeline integrity, data validation, and model scoring bounds:
```bash
pytest
```

### Step 3: Launch the Streamlit Web Application
Start the interactive dashboard:
```bash
streamlit run app.py
```
Access the application in your browser at `http://localhost:8501`.

### Step 4: CLI Single & Batch Predictions
Run inference directly from the command line:
```bash
# Evaluate a preset legitimate or fraudulent transaction
python predict.py --demo-single fraud

# Process an entire batch CSV
python predict.py --batch data/batch_test_unlabelled.csv --output data/predictions_output.csv
```

---

## 📈 Real Empirical Evaluation Metrics

Below are the actual held-out test set metrics computed by `train.py` on a strict 3-way stratified partition (15,000 Train / 5,000 Validation / 5,000 Held-Out Test, with 10 actual frauds in the test set):

| Metric | Isolation Forest (Primary) | Local Outlier Factor (LOF) | One-Class SVM |
| :--- | :---: | :---: | :---: |
| **Operating Threshold** | **0.6951** *(Val Calibrated)* | 0.1000 | 0.9500 |
| **Recall (Caught Frauds)** | **50.00%** *(5 / 10)* | 0.00% *(0 / 10)* | **70.00%** *(7 / 10)* |
| **Precision** | **17.24%** *(5 TP / 24 FP)* | 0.00% *(0 TP / 3 FP)* | 4.35% *(7 TP / 154 FP)* |
| **F1-Score** | **0.2564** | 0.0000 | 0.0819 |
| **F2-Score ($\beta=2$)** | **0.3623** | 0.0000 | 0.1741 |
| **Average Precision (AP)** | **0.3718** *(185x baseline lift)* | 0.1868 | 0.0572 |
| **ROC-AUC** | **0.8878** | 0.9551 | 0.9317 |
| **Balanced Accuracy** | **74.76%** | 49.97% | 83.46% |
| **MCC (Matthews Corr)** | **+0.2913** | -0.0011 | +0.1693 |
| **Training Time** | **0.67s** | 1.87s | 0.68s |

> [!NOTE]
> - **Prevalence Baseline**: With 10 frauds in 5,000 test transactions, random guessing yields a Precision of **0.20% (0.0020)**. Isolation Forest's Average Precision of **0.3718** represents a massive **185x lift** over random baseline.
> - **Zero-Leakage Calibration**: The decision threshold was calibrated on the independent 5,000-sample validation set to optimize the $F_2$-score (favoring fraud recall), leaving the 5,000-sample held-out test set completely untouched.
> - **Test Suite & Coverage**: 35 unit/integration tests (`pytest tests/`) pass with **88% statement coverage** across all modules in `src/`.

---

## 🛡️ Key Technical Concepts for Interviewers

- **Why RobustScaler instead of StandardScaler?**
  Transaction amounts exhibit extreme positive skewness (median ~$25, max > $20,000). `StandardScaler` calculates mean and standard deviation, which are heavily distorted by outliers. `RobustScaler` scales using the median and Interquartile Range ($IQR = Q_3 - Q_1$), preserving the true distance of typical transactions.
- **Why PR-AUC over ROC-AUC?**
  In extreme class imbalance (99.83% negative class), ROC-AUC can be deceptive because the False Positive Rate ($FPR = \frac{FP}{FP + TN}$) remains tiny even when thousands of false alarms occur. Precision-Recall AUC focuses exclusively on the minority positive class, providing a faithful evaluation of fraud detection efficacy.
- **Why is an Anomaly Not Necessarily Fraud?**
  An anomaly is a mathematical deviation from normal clusters. A legitimate cardholder buying a wedding ring or traveling abroad will register as anomalous. In production, anomalies trigger **step-up friction** (SMS OTP / Biometrics) rather than immediate card termination.

---

## 🔮 Limitations & Future Scope

1. **Feature Interpretability**: Kaggle PCA components ($V_1 - V_{28}$) conceal raw business variables (Merchant Category, IP geolocation, Device Fingerprint).
2. **Dynamic Concept Drift**: Spending patterns shift during holiday periods; automated re-training pipelines with model monitoring (Evidently AI / MLflow) are recommended.
3. **Graph Neural Networks (GNNs)**: Future iterations can integrate transaction graph embeddings to capture coordinated fraud rings and mule networks.

---

## 📜 License
This project is open-source and licensed under the MIT License.
