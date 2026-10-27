"""
Credit Card Fraud Detection Dashboard using Anomaly Detection Techniques.
Assessment-ready Streamlit Application for HCL Technical Evaluation.
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import (
    load_dataset,
    split_data,
    validate_schema,
    REQUIRED_FEATURES,
    TARGET_COLUMN,
    DEFAULT_REAL_PATH,
    DEFAULT_SYNTHETIC_PATH
)
from src.preprocessing import FraudPreprocessor, validate_transaction_dict
from src.models import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
    create_model,
    BaseAnomalyModel
)
from src.evaluation import (
    evaluate_predictions,
    explain_class_imbalance,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_model_comparison
)
from predict import (
    FraudPredictor,
    generate_sample_transaction,
    get_risk_level,
    explain_single_transaction_anomaly
)
from train import run_training_pipeline

# Page Configuration
st.set_page_config(
    page_title="Fraud Sentinel | Anomaly Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #38bdf8;
        margin-top: 4px;
    }
    /* Risk Badges */
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #22c55e;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    /* Header Accent */
    .main-title {
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)


# Caching dataset loading
@st.cache_data(show_spinner="Loading dataset into memory...")
def get_cached_dataset(sample_limit: Optional[int] = None):
    return load_dataset(sample_size=sample_limit)


# Caching trained models
@st.cache_resource(show_spinner="Loading machine learning pipelines...")
def load_cached_predictor(model_name: str, threshold: float):
    try:
        return FraudPredictor(model_name=model_name, threshold=threshold)
    except FileNotFoundError:
        return None


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("Fraud Sentinel AI")
st.sidebar.caption("Anomaly Detection Technical Project | HCL Assessment")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Overview",
        "📊 Dataset & Fraud EDA",
        "🧠 Model Benchmark & Training",
        "🔍 Single Transaction Inference",
        "📁 Batch CSV Screening",
        "📘 Architecture & Technical Guide"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Global Settings")
selected_model = st.sidebar.selectbox(
    "Active Anomaly Model",
    ["isolation_forest", "lof", "one_class_svm"],
    format_func=lambda x: {
        "isolation_forest": "🌲 Isolation Forest (Primary)",
        "lof": "📍 Local Outlier Factor (LOF)",
        "one_class_svm": "⭕ One-Class SVM"
    }[x]
)

decision_threshold = st.sidebar.slider(
    "Decision Threshold (Anomaly Cutoff)",
    min_value=0.10,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Transactions with anomaly score >= threshold will be flagged as fraudulent."
)

st.sidebar.info(
    "**Tip**: Lower thresholds increase **Recall** (catch more frauds) but increase False Positives. "
    "Higher thresholds increase **Precision**."
)


# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ==============================================================================
if menu == "🏠 Executive Overview":
    st.markdown("<h1 class='main-title'>🛡️ Credit Card Fraud Detection</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='sub-title'>Unsupervised Anomaly Detection System built for extreme class imbalance in banking transactions</p>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Primary Algorithm</div>
            <div class="metric-value">Isolation Forest</div>
            <div class="metric-subtitle">Tree-based isolation</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Contamination Target</div>
            <div class="metric-value">0.17% - 0.20%</div>
            <div class="metric-subtitle">Matches Kaggle benchmark</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Feature Space</div>
            <div class="metric-value">30 Features</div>
            <div class="metric-subtitle">Time, Amount, V1-V28 (PCA)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Active Threshold</div>
            <div class="metric-value">{:.2f}</div>
            <div class="metric-subtitle">Normalized score cutoff</div>
        </div>
        """.format(decision_threshold), unsafe_allow_html=True)

    st.subheader("📌 Project Problem Statement")
    st.write(
        "Credit card fraud results in tens of billions of dollars in annual losses worldwide. "
        "Traditional supervised machine learning struggles in this domain due to two major hurdles:"
    )
    st.markdown("""
    1. **Extreme Class Imbalance**: Fraud accounts for typically less than **0.2%** of all transactions. 
       A standard classifier predicting 'legitimate' for every transaction achieves 99.8% accuracy while failing 100% of frauds (the *Accuracy Paradox*).
    2. **Evolving Attack Vectors (Zero-Day Fraud)**: Fraudsters actively adapt their tactics. Supervised models trained on past labels cannot detect novel fraud patterns.
    """)

    st.subheader("💡 Why Anomaly Detection?")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("""
        Anomaly detection fundamentally reframes the objective:
        - **Model the normal**: It learns the intrinsic geometric and density distribution of legitimate cardholder behavior.
        - **Isolate the deviations**: Any transaction that deviates substantially from normal behavior is scored with a high anomaly score.
        - **Explainability**: Deviations can be traced back to specific components (e.g., negative shifts in V14/V12 or abnormal amounts).
        - **Novelty Sensitivity**: Catches previously unseen fraud patterns without needing pre-existing labels.
        """)
    with col_b:
        st.info("""
        **System Core Capabilities:**
        - 🌲 **Multi-Algorithm Benchmark**: Isolation Forest, Local Outlier Factor (Novelty Mode), and One-Class SVM.
        - ⚖️ **Leakage-Free Preprocessing**: RobustScaler on skewed transaction amounts.
        - 🎯 **Interactive Thresholding**: Dynamic Precision/Recall trade-off control.
        - ⚡ **Real-Time & Batch Screening**: Single transaction diagnosis and CSV processing.
        """)

    st.divider()
    st.subheader("🏗️ End-to-End System Pipeline")
    st.markdown("""
    ```
    [Raw Transaction Data] 
            │
            ▼
    [Schema Validation & Leakage-Free Split (80/20)]
            │
            ▼
    [RobustScaler: Amount & Time] ──> [Fitted Preprocessor Pipeline]
            │
            ▼
    [Anomaly Detection Engines]
      ├─ Isolation Forest (iTree path length isolation)
      ├─ Local Outlier Factor (k-NN density estimation with novelty=True)
      └─ One-Class SVM (Kernel margin boundary)
            │
            ▼
    [Continuous Anomaly Scoring in [0.0, 1.0]]
            │
            ▼
    [Dynamic Decision Threshold (0.50)] ──> [Alerts, Enriched CSV & Risk Banding]
    ```
    """)


# ==============================================================================
# TAB 2: DATASET & FRAUD EDA
# ==============================================================================
elif menu == "📊 Dataset & Fraud EDA":
    st.markdown("<h1 class='main-title'>📊 Dataset Analysis & Fraud Distribution</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Exploratory data analysis of European transaction patterns and PCA component behaviors</p>", unsafe_allow_html=True)

    try:
        df, metadata = get_cached_dataset()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    # Dataset Status Banner
    if metadata["source_type"] == "synthetic_demonstration":
        st.warning(
            "ℹ️ **Dataset Status**: Currently operating with high-fidelity **Synthetic Demonstration Data** "
            "(exact Kaggle schema: Time, V1-V28, Amount, Class). To use the full real European dataset, "
            "place `creditcard.csv` inside the `data/` folder."
        )
    else:
        st.success(f"✅ **Dataset Status**: Operating with **Real Kaggle European Credit Card Dataset** ({metadata['total_records']:,} records).")

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{metadata['total_records']:,}")
    c2.metric("Legitimate Count", f"{metadata['normal_count']:,}")
    c3.metric("Fraudulent Count", f"{metadata['fraud_count']:,}")
    c4.metric("Fraud Proportion", f"{metadata['fraud_rate']:.4%}")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("1. Class Imbalance Distribution")
        fig_class, ax_class = plt.subplots(figsize=(6, 4), dpi=100)
        counts = [metadata['normal_count'], metadata['fraud_count']]
        labels = [f"Legitimate ({metadata['normal_count']:,})", f"Fraud ({metadata['fraud_count']:,})"]
        colors = ["#3b82f6", "#ef4444"]
        bars = ax_class.bar(["Legitimate (0)", "Fraud (1)"], counts, color=colors, width=0.45)
        ax_class.set_yscale("log")
        ax_class.set_ylabel("Count (Log Scale)", fontsize=10)
        ax_class.set_title("Transaction Frequency (Log Scale to reveal Fraud)", fontsize=11, weight="bold")
        for bar in bars:
            height = bar.get_height()
            ax_class.annotate(f"{height:,}", (bar.get_x() + bar.get_width() / 2, height),
                              ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")
        st.pyplot(fig_class)
        plt.close(fig_class)

    with col_right:
        st.subheader("2. Transaction Amount Distribution")
        fig_amt, ax_amt = plt.subplots(figsize=(6, 4), dpi=100)
        normal_amounts = df[df["Class"] == 0]["Amount"]
        fraud_amounts = df[df["Class"] == 1]["Amount"]

        # Clip amounts for visualization
        sns.kdeplot(normal_amounts[normal_amounts < 1500], color="#3b82f6", label="Legitimate", fill=True, alpha=0.3, ax=ax_amt)
        sns.kdeplot(fraud_amounts[fraud_amounts < 1500], color="#ef4444", label="Fraud", fill=True, alpha=0.3, ax=ax_amt)
        ax_amt.set_title("Amount Density (Transactions < $1,500)", fontsize=11, weight="bold")
        ax_amt.set_xlabel("Amount ($)", fontsize=10)
        ax_amt.legend()
        st.pyplot(fig_amt)
        plt.close(fig_amt)

    st.subheader("3. Anomaly Signals in PCA Features (V14 vs V12 vs V10)")
    st.write("In credit card fraud analysis, specific PCA components exhibit marked negative separation for fraud events:")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_v, ax_v = plt.subplots(figsize=(7, 4.5), dpi=100)
        sns.scatterplot(
            data=df.sample(min(2500, len(df)), random_state=42),
            x="V14", y="V12",
            hue="Class",
            palette={0: "#3b82f6", 1: "#ef4444"},
            alpha=0.6,
            ax=ax_v
        )
        ax_v.set_title("V14 vs V12 Scatter Separation", fontsize=11, weight="bold")
        ax_v.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig_v)
        plt.close(fig_v)

    with col_v2:
        fig_box, ax_box = plt.subplots(figsize=(7, 4.5), dpi=100)
        df_melt = pd.melt(
            df.sample(min(3000, len(df)), random_state=42),
            id_vars=["Class"],
            value_vars=["V10", "V12", "V14", "V17"],
            var_name="Component",
            value_name="Value"
        )
        sns.boxplot(data=df_melt, x="Component", y="Value", hue="Class", palette={0: "#3b82f6", 1: "#ef4444"}, ax=ax_box)
        ax_box.set_title("Key Diagnostic Features (Legit vs Fraud Boxplots)", fontsize=11, weight="bold")
        st.pyplot(fig_box)
        plt.close(fig_box)

    st.subheader("4. Raw Data Sample Preview")
    st.dataframe(df.head(15), use_container_width=True)


# ==============================================================================
# TAB 3: MODEL BENCHMARK & TRAINING
# ==============================================================================
elif menu == "🧠 Model Benchmark & Training":
    st.markdown("<h1 class='main-title'>🧠 Model Benchmark & Performance Evaluation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Empirical evaluation on held-out test data without data leakage</p>", unsafe_allow_html=True)

    results_file = os.path.join("models", "evaluation_results.json")
    has_results = os.path.exists(results_file)

    # Train / Retrain Controls
    with st.expander("🛠️ Model Training & Artifact Generation Controls", expanded=not has_results):
        st.write("Retrain the anomaly detection models and recompute all held-out test metrics:")
        c_t1, c_t2, c_t3 = st.columns(3)
        with c_t1:
            train_samples = st.selectbox(
                "Training Sample Size",
                [10000, 15000, 30000, 50000, None],
                index=1,
                help="Subsampling enables fast training during live interview presentations."
            )
        with c_t2:
            train_contamination = st.number_input(
                "Contamination Parameter",
                min_value=0.0005,
                max_value=0.05,
                value=0.002,
                step=0.0005,
                format="%.4f"
            )
        with c_t3:
            train_mode = st.radio(
                "Training Mode",
                ["Unsupervised (Full data)", "Semi-Supervised (Normal-only)"],
                index=0
            )

        if st.button("🚀 Execute Training Pipeline Now", type="primary", use_container_width=True):
            with st.spinner("Executing training pipeline (Fitting RobustScaler and Anomaly Detectors)..."):
                summary = run_training_pipeline(
                    samples=train_samples,
                    contamination=train_contamination,
                    normal_only_train=(train_mode == "Semi-Supervised (Normal-only)"),
                    models_to_train=["isolation_forest", "lof", "one_class_svm"]
                )
                st.success("✅ Models trained and evaluation metrics generated successfully!")
                time.sleep(1)
                st.rerun()

    if not os.path.exists(results_file):
        st.warning("⚠️ No pre-trained evaluation results found. Click 'Execute Training Pipeline Now' above to train models.")
        st.stop()

    with open(results_file, "r") as f:
        eval_data = json.load(f)

    # Robust extraction of split metadata
    split_cfg = eval_data.get("split_config", {})
    test_samples = split_cfg.get("test_samples") or eval_data.get("test_set_size") or 0
    test_frauds = split_cfg.get("test_frauds") or eval_data.get("test_frauds") or 0
    val_samples = split_cfg.get("val_samples") or 0
    train_samples = split_cfg.get("train_samples") or 0

    if val_samples > 0:
        caption_text = (
            f"Last Evaluated: **{eval_data.get('timestamp', 'N/A')}** | "
            f"Held-Out Test Set: **{test_samples:,} samples** ({test_frauds} frauds) | "
            f"Validation Set: **{val_samples:,} samples** | "
            f"Training Set: **{train_samples:,} samples**"
        )
    else:
        caption_text = (
            f"Last Evaluated: **{eval_data.get('timestamp', 'N/A')}** | "
            f"Held-Out Test Set: **{test_samples:,} samples** ({test_frauds} frauds)"
        )
    st.caption(caption_text)

    # Model Comparison Table
    st.subheader("1. Empirical Metric Benchmark (Held-Out Test Set)")
    df_metrics = pd.DataFrame(eval_data["results"])

    # Ensure backward/forward compatibility for PR-AUC column name
    if "pr_auc" not in df_metrics.columns:
        if "average_precision" in df_metrics.columns:
            df_metrics["pr_auc"] = df_metrics["average_precision"]
        elif "trapezoidal_pr_auc" in df_metrics.columns:
            df_metrics["pr_auc"] = df_metrics["trapezoidal_pr_auc"]
        else:
            df_metrics["pr_auc"] = 0.0

    candidate_cols = [
        "model_name", "operating_threshold", "precision", "recall",
        "f1_score", "f2_score", "balanced_accuracy", "roc_auc", "pr_auc", "train_time_sec"
    ]
    display_cols = [c for c in candidate_cols if c in df_metrics.columns]
    renamed_cols = {
        "model_name": "Algorithm",
        "operating_threshold": "Calibrated Threshold",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1-Score",
        "f2_score": "F2-Score",
        "balanced_accuracy": "Balanced Acc",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC (AP)",
        "train_time_sec": "Train Time (s)"
    }
    format_dict = {
        "Calibrated Threshold": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1-Score": "{:.4f}",
        "F2-Score": "{:.4f}",
        "Balanced Acc": "{:.4f}",
        "ROC-AUC": "{:.4f}",
        "PR-AUC (AP)": "{:.4f}",
        "Train Time (s)": "{:.2f}"
    }

    df_display = df_metrics[display_cols].rename(columns=renamed_cols)
    active_formats = {k: v for k, v in format_dict.items() if k in df_display.columns}
    highlight_cols = [c for c in ["Precision", "Recall", "F1-Score", "F2-Score", "PR-AUC (AP)"] if c in df_display.columns]

    styled_df = df_display.style.format(active_formats)
    if highlight_cols:
        styled_df = styled_df.highlight_max(subset=highlight_cols, color="#1e3a8a")

    st.dataframe(styled_df, use_container_width=True)

    # Diagnostic Plots
    st.subheader("2. Performance Visualizations")
    tab_comp, tab_curves, tab_cm = st.tabs(["📊 Metric Comparison Bar Chart", "📈 ROC & PR Curves", "🔲 Confusion Matrices"])

    with tab_comp:
        comp_img_path = os.path.join("models", "model_comparison.png")
        if os.path.exists(comp_img_path):
            st.image(comp_img_path, caption="Comparative Metric Benchmark on Held-out Test Data", use_container_width=True)
        else:
            st.info("Comparison plot will appear after running train.py.")

    with tab_curves:
        curves_img_path = os.path.join("models", "roc_pr_curves.png")
        if os.path.exists(curves_img_path):
            st.image(curves_img_path, caption="ROC and Precision-Recall Curves (Notice PR-AUC highlights imbalanced detection capability)", use_container_width=True)
        else:
            st.info("Curves will appear after running train.py.")

    with tab_cm:
        cols_cm = st.columns(len(eval_data["results"]))
        for i, res in enumerate(eval_data["results"]):
            with cols_cm[i]:
                st.write(f"**{res['model_name']}**")
                cm_img_path = os.path.join("models", f"confusion_matrix_{res['model_name'].lower().replace(' ', '_')}.png")
                if os.path.exists(cm_img_path):
                    st.image(cm_img_path, use_container_width=True)
                st.markdown(f"""
                - **True Positives (Frauds Caught)**: `{res['true_positives']}`
                - **False Positives (False Alarms)**: `{res['false_positives']}`
                - **False Negatives (Missed Frauds)**: `{res['false_negatives']}`
                - **True Negatives (Legit Cleared)**: `{res['true_negatives']}`
                """)

    # Educational Callout
    st.divider()
    st.subheader("💡 Why Accuracy is Insufficient (The Accuracy Paradox)")
    if "imbalance_explanation" in eval_data:
        st.markdown(eval_data["imbalance_explanation"])


# ==============================================================================
# TAB 4: SINGLE TRANSACTION INFERENCE
# ==============================================================================
elif menu == "🔍 Single Transaction Inference":
    st.markdown("<h1 class='main-title'>🔍 Single Transaction Real-Time Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Live anomaly scoring and risk explainability for card transactions</p>", unsafe_allow_html=True)

    predictor = load_cached_predictor(selected_model, decision_threshold)
    if predictor is None:
        st.error(f"Model artifacts for '{selected_model}' not found in `models/`. Please train models first in the 'Model Benchmark' tab.")
        st.stop()

    st.subheader("1. Preset Transaction Scenarios")
    st.write("Click a preset below to populate realistic transaction values:")
    c_p1, c_p2, c_p3 = st.columns(3)
    preset_choice = None
    if c_p1.button("💳 Legitimate Coffee / Grocery ($45.50)", use_container_width=True):
        preset_choice = "legit"
    if c_p2.button("🚨 Typical Fraud Transaction ($1,250.00)", use_container_width=True):
        preset_choice = "fraud"
    if c_p3.button("⚠️ Unusual High-Amount Outlier ($4,800.00)", use_container_width=True):
        preset_choice = "outlier"

    # Load initial values
    if "current_tx" not in st.session_state:
        st.session_state.current_tx = generate_sample_transaction(is_fraud=False)

    if preset_choice == "legit":
        st.session_state.current_tx = generate_sample_transaction(is_fraud=False)
    elif preset_choice == "fraud":
        st.session_state.current_tx = generate_sample_transaction(is_fraud=True)
    elif preset_choice == "outlier":
        tx = generate_sample_transaction(is_fraud=False)
        tx["Amount"] = 4800.0
        tx["V1"] = -3.5
        tx["V14"] = -2.8
        st.session_state.current_tx = tx

    st.subheader("2. Transaction Feature Inputs")
    with st.form("single_tx_form"):
        col_main1, col_main2 = st.columns(2)
        with col_main1:
            inp_amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                max_value=50000.0,
                value=float(st.session_state.current_tx.get("Amount", 50.0)),
                step=10.0
            )
        with col_main2:
            inp_time = st.number_input(
                "Transaction Time (Seconds elapsed, 0 - 172800)",
                min_value=0.0,
                max_value=200000.0,
                value=float(st.session_state.current_tx.get("Time", 3600.0)),
                step=60.0
            )

        with st.expander("🔧 Adjust PCA Features (V1 - V28)", expanded=False):
            st.caption("PCA features reflect anonymized cardholder, device, and behavioral embeddings. Extreme negative values in V14, V12, V10, V17 strongly indicate fraud.")
            pca_cols = st.columns(4)
            tx_features = {"Time": inp_time, "Amount": inp_amount}
            for i in range(1, 29):
                feat = f"V{i}"
                col_idx = (i - 1) % 4
                default_val = float(st.session_state.current_tx.get(feat, 0.0))
                with pca_cols[col_idx]:
                    tx_features[feat] = st.number_input(
                        feat,
                        min_value=-25.0,
                        max_value=25.0,
                        value=default_val,
                        step=0.1,
                        format="%.3f"
                    )

        submit_btn = st.form_submit_button("⚡ Evaluate Transaction Risk", type="primary", use_container_width=True)

    if submit_btn or preset_choice is not None:
        if not submit_btn:
            tx_features = st.session_state.current_tx.copy()

        # Run inference
        res = predictor.predict_single(tx_features, custom_threshold=decision_threshold)
        score = res["anomaly_score"]
        is_fraud = res["prediction"] == 1
        risk_level = res["risk_level"]

        st.divider()
        st.subheader("3. Risk Diagnosis & Decision Engine")

        col_res1, col_res2 = st.columns([2, 3])
        with col_res1:
            if is_fraud:
                st.error(f"### 🚨 ALERT: Fraudulent / Suspicious")
            else:
                st.success(f"### 🛡️ VERIFIED: Legitimate Transaction")

            # Score meter
            st.metric("Anomaly Score", f"{score:.4f}", f"Threshold: {decision_threshold:.2f}")
            st.progress(score)

            # Badge display
            badge_class = f"badge-{risk_level.lower()}"
            st.markdown(f"Risk Band: <span class='{badge_class}'>{risk_level} RISK</span>", unsafe_allow_html=True)
            st.caption(f"Evaluated via: **{res['model_used']}**")

        with col_res2:
            st.write("#### 🔍 Explainability Breakdown")
            st.write("Why did the anomaly detection engine reach this conclusion?")
            for r in res["reasons"]:
                st.markdown(f"- {r}")

            st.info(
                f"**Decision Rule**: A transaction with score **{score:.4f}** is classified as "
                f"**{'Fraud' if is_fraud else 'Legit'}** because it is "
                f"**{'>=' if is_fraud else '<'}** the operational threshold of **{decision_threshold:.2f}**."
            )


# ==============================================================================
# TAB 5: BATCH CSV SCREENING
# ==============================================================================
elif menu == "📁 Batch CSV Screening":
    st.markdown("<h1 class='main-title'>📁 Batch CSV Transaction Screening</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Upload unlabelled or historical transaction batches for automated fraud screening</p>", unsafe_allow_html=True)

    predictor = load_cached_predictor(selected_model, decision_threshold)
    if predictor is None:
        st.error(f"Model artifacts for '{selected_model}' not found in `models/`. Please train models first.")
        st.stop()

    # Pre-packaged batch selector
    st.write("Choose a demo batch file or upload your own CSV:")
    demo_batch_path = os.path.join("data", "batch_test_unlabelled.csv")
    use_demo = False
    if os.path.exists(demo_batch_path):
        if st.checkbox("Use pre-packaged test batch (`data/batch_test_unlabelled.csv` - 200 transactions)", value=True):
            use_demo = True

    uploaded_file = st.file_uploader("Upload CSV transaction file", type=["csv"], disabled=use_demo)

    batch_df = None
    if use_demo and os.path.exists(demo_batch_path):
        batch_df = pd.read_csv(demo_batch_path)
        st.info("Loaded pre-packaged test batch (200 records).")
    elif uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {e}")
            st.stop()

    if batch_df is not None:
        st.subheader("1. Batch Preview & Schema Validation")
        is_valid, msg = validate_schema(batch_df, is_inference=True)
        if not is_valid:
            st.error(f"❌ Schema validation failed: {msg}")
            st.stop()
        else:
            st.success(f"✅ Schema validated successfully. Found {len(batch_df):,} records with required features.")

        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("🚀 Run Anomaly Screening on Batch", type="primary"):
            with st.spinner(f"Screening {len(batch_df):,} transactions using {selected_model}..."):
                t_start = time.time()
                enriched_df, summary = predictor.predict_batch(batch_df, custom_threshold=decision_threshold)
                t_elapsed = time.time() - t_start

            st.success(f"Batch screening completed in {t_elapsed:.2f} seconds!")

            # KPI Summary
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Screened", f"{summary['total_transactions']:,}")
            k2.metric("Flagged Fraud Alerts", f"{summary['flagged_anomalies']:,}")
            k3.metric("Normal Approved", f"{summary['normal_transactions']:,}")
            k4.metric("Flagged Ratio", f"{summary['flagged_percentage']:.2%}")

            st.subheader("2. Screened Transactions Table")
            # Filter options
            filter_choice = st.radio("Filter Table View:", ["All Records", "Flagged Fraud Only", "Normal Only"], horizontal=True)
            if filter_choice == "Flagged Fraud Only":
                display_df = enriched_df[enriched_df["Is_Anomaly"] == 1]
            elif filter_choice == "Normal Only":
                display_df = enriched_df[enriched_df["Is_Anomaly"] == 0]
            else:
                display_df = enriched_df

            st.dataframe(display_df, use_container_width=True)

            # Anomaly Score Distribution Plot
            st.subheader("3. Batch Anomaly Score Distribution")
            fig_score, ax_score = plt.subplots(figsize=(8, 3.5), dpi=100)
            sns.histplot(enriched_df["Anomaly_Score"], bins=35, kde=True, color="#3b82f6", ax=ax_score)
            ax_score.axvline(decision_threshold, color="#ef4444", linestyle="--", lw=2, label=f"Decision Threshold ({decision_threshold:.2f})")
            ax_score.set_title("Anomaly Score Histogram (Scores >= Threshold are Flagged)", fontsize=11, weight="bold")
            ax_score.set_xlabel("Anomaly Score (0.0 to 1.0)", fontsize=10)
            ax_score.legend()
            st.pyplot(fig_score)
            plt.close(fig_score)

            # Download Enriched CSV
            st.subheader("4. Export Enriched CSV")
            csv_data = enriched_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Enriched Predictions CSV",
                data=csv_data,
                file_name="fraud_screening_results.csv",
                mime="text/csv",
                type="primary"
            )


# ==============================================================================
# TAB 6: ARCHITECTURE & TECHNICAL GUIDE
# ==============================================================================
elif menu == "📘 Architecture & Technical Guide":
    st.markdown("<h1 class='main-title'>📘 Project Architecture & Assessment Guide</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>In-depth engineering documentation and conceptual explanations for the HCL technical panel</p>", unsafe_allow_html=True)

    tech_tabs = st.tabs([
        "🏛️ System Architecture",
        "🧮 Algorithm Theoretical Comparison",
        "⚖️ Contamination & Thresholding",
        "🚧 Production Limitations & Future Scope"
    ])

    with tech_tabs[0]:
        st.subheader("End-to-End Modular Architecture")
        st.markdown("""
        The system is organized into clean, production-grade decoupled layers:

        - **`src/dataset.py`**:
          - Dataset discovery and loading (handles real Kaggle European dataset or auto-generates realistic synthetic fallback).
          - Schema verification, type validation, NaN/Inf checks.
          - Stratified train/test splitting to preserve rare fraud occurrences without data leakage.
        - **`src/preprocessing.py`**:
          - `FraudPreprocessor`: Fits `RobustScaler` on transaction Amount and Time strictly on the training partition.
          - Reusable pipeline loaded via `joblib` during real-time inference.
        - **`src/models.py`**:
          - Object-oriented wrappers around Isolation Forest, Local Outlier Factor (Novelty Mode), and One-Class SVM.
          - Standardized continuous anomaly scores normalized to `[0.0, 1.0]`.
        - **`src/evaluation.py`**:
          - Real held-out evaluation: Precision, Recall, F1, Confusion Matrix, ROC-AUC, and PR-AUC.
        - **`predict.py` & `app.py`**:
          - CLI and Streamlit interfaces for single and batch processing.
        """)

    with tech_tabs[1]:
        st.subheader("Algorithms Compared in this Project")
        st.markdown(r"""
        | Algorithm | Core Mechanism | Strengths | Weaknesses | Best Use Case |
        | :--- | :--- | :--- | :--- | :--- |
        | **Isolation Forest** | Ensembles of random binary trees isolate anomalies near tree root. | Fast $O(n \log n)$, handles high dimensions, no distance metric required. | Random splits can occasionally miss subtle boundary points. | **Large-scale transaction screening (Primary choice)** |
        | **Local Outlier Factor (LOF)** | Density-based comparison to $k$-nearest neighbors (local reachability density). | Captures local clusters with varying densities. | $O(n^2)$ inference time without indexing; requires `novelty=True`. | **Moderate volume, multi-cluster user behavior** |
        | **One-Class SVM** | Kernel margin boundary separating normal data from the origin in Hilbert space. | Strong non-linear boundary learning with RBF kernel. | Computationally intensive $O(n^3)$ training; sensitive to hyperparameter `nu`. | **Tight profile verification** |
        """)

    with tech_tabs[2]:
        st.subheader("Understanding Contamination and Decision Thresholds")
        st.markdown("""
        ### What is Contamination?
        - **Definition**: The expected proportion of outliers / anomalies in the training dataset.
        - In this dataset, the empirical fraud rate is approximately **0.17% (0.0017)**.
        - Specifying contamination allows the algorithm to calibrate its default internal decision boundary so that roughly the top 0.17% most anomalous instances are flagged.

        ### Why is an Anomaly Not Always Fraud?
        - **False Positives**: An outlier simply represents a mathematical deviation from typical patterns.
        - Examples of legitimate anomalies:
          - A cardholder purchasing an expensive flight ticket or luxury electronics.
          - Buying coffee while traveling in another country.
        - Therefore, high anomaly scores trigger **step-up authentication** (SMS OTP, biometric prompt, or 2FA) rather than instant account cancellation.
        """)

    with tech_tabs[3]:
        st.subheader("Real-World Limitations & Future Roadmap")
        st.markdown("""
        1. **Feature Anonymization**: Real PCA components (V1-V28) obscure domain intuition (e.g., Merchant Category Code, IP geolocation, Device ID).
        2. **Concept Drift**: Cardholder spending habits shift over seasons (e.g., Black Friday, holidays). Models require automated re-training pipelines.
        3. **Hybrid Ensemble**: Combining unsupervised anomaly detectors with semi-supervised Graph Neural Networks (GNNs) to capture card-merchant fraud rings.
        4. **Low Latency Deployment**: Porting the inference pipeline into ONNX or C++ microservices to meet strict **< 50ms** payment gateway SLAs.
        """)
