"""
Rigorous Leakage-Free Training, Validation, and Evaluation Pipeline.

Architecture:
1. 3-Way Partition: Train (60%) -> Validation (20%) -> Final Test (20%).
2. Leakage-Free Scaling: RobustScaler fitted strictly on Train features.
3. Model Training: Unsupervised anomaly detectors fitted strictly on Train.
4. Threshold Calibration: Operating threshold optimized on Validation set.
5. Final Evaluation: Evaluated on the held-out untouched Test set.
6. Artifact Serialization & Diagnostic Plotting.
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import load_dataset, split_data_3way, REQUIRED_FEATURES
from src.preprocessing import FraudPreprocessor
from src.models import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    OneClassSVMDetector,
    create_model
)
from src.evaluation import (
    evaluate_predictions,
    explain_class_imbalance,
    plot_confusion_matrix,
    plot_roc_pr_curves,
    plot_model_comparison
)


def run_training_pipeline(
    data_path: Optional[str] = None,
    samples: Optional[int] = None,
    contamination: float = 0.002,
    split_type: str = "stratified",
    normal_only_train: bool = False,
    models_to_train: Optional[List[str]] = None,
    output_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute full leakage-free training, calibration, and evaluation pipeline.
    """
    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    if models_to_train is None:
        models_to_train = ["isolation_forest", "lof", "one_class_svm"]

    print("=" * 78)
    print(" CREDIT CARD FRAUD DETECTION - ANOMALY DETECTION TRAINING PIPELINE ")
    print("=" * 78)
    print(f"[*] Contamination Parameter: {contamination:.4f}")
    print(f"[*] Split Methodology: {split_type.upper()} 3-Way (60% Train / 20% Val / 20% Test)")
    print(f"[*] Training Paradigm: {'Semi-supervised (Normal-only)' if normal_only_train else 'Unsupervised (Full data with contamination)'}")
    print(f"[*] Target Models: {', '.join(models_to_train)}")
    print(f"[*] Artifact Output Directory: {os.path.abspath(output_dir)}")
    print("-" * 78)

    # 1. Load Dataset
    print("\n[1/5] Loading and validating dataset...")
    df, metadata = load_dataset(path=data_path, sample_size=samples, random_state=random_state)
    print(f"      Source: {metadata['source_type']} ({'SYNTHETIC' if metadata['is_synthetic'] else 'REAL BENCHMARK'})")
    print(f"      File: {metadata['file_path']}")
    print(f"      Total Records: {metadata['total_records']:,}")
    print(f"      Legitimate: {metadata['normal_count']:,} | Fraud: {metadata['fraud_count']:,} ({metadata['fraud_rate']:.4%})")

    # 2. Leakage-Free 3-Way Partitioning
    print(f"\n[2/5] Partitioning dataset into Train (60%), Val (20%), and Test (20%) via {split_type} split...")
    splits = split_data_3way(
        df,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        split_type=split_type,
        normal_only_train=normal_only_train,
        random_state=random_state
    )

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    print(f"      Train Partition : {splits['train_samples']:,} samples ({splits['train_frauds']} frauds)")
    print(f"      Val Partition   : {splits['val_samples']:,} samples ({splits['val_frauds']} frauds) -> Threshold Calibration")
    print(f"      Test Partition  : {splits['test_samples']:,} samples ({splits['test_frauds']} frauds) -> Final Untouched Evaluation")

    # 3. Fit Preprocessing Pipeline (Strictly on Train Features)
    print("\n[3/5] Fitting RobustScaler preprocessing pipeline on training features only...")
    preprocessor = FraudPreprocessor(scale_time=True, scale_amount=True)
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    preprocessor_path = os.path.join(output_dir, "preprocessor.joblib")
    preprocessor.save(preprocessor_path)
    print(f"      Fitted RobustScaler on X_train (Median Amount=${preprocessor.amount_scaler.center_[0]:.2f}, IQR=${preprocessor.amount_scaler.scale_[0]:.2f})")
    print(f"      Saved preprocessor artifact to: {preprocessor_path}")

    # 4. Train Models, Calibrate Thresholds on Validation, and Evaluate on Test
    print("\n[4/5] Training anomaly detectors and evaluating on held-out test split...")
    results_list = []
    models_scores = {}
    trained_models = {}

    for m_type in models_to_train:
        print(f"\n      -> Initializing and training {m_type.upper()}...")
        t0 = time.time()

        # Adjust training subsampling for LOF/OCSVM if dataset is large to maintain fast runtime
        if m_type in ("lof", "one_class_svm") and len(X_train_scaled) > 15000:
            sub_idx = np.random.RandomState(random_state).choice(len(X_train_scaled), size=15000, replace=False)
            X_fit = X_train_scaled[sub_idx]
        else:
            X_fit = X_train_scaled

        detector = create_model(m_type, contamination=contamination)
        detector.fit(X_fit)
        train_duration = time.time() - t0
        print(f"         Fitted in {train_duration:.2f} seconds.")

        # Threshold Calibration on Validation Partition (Zero Test Leakage)
        t_calib_0 = time.time()
        calibrated_thresh = detector.calibrate_threshold(
            X_val_scaled,
            y_val=y_val.to_numpy(dtype=int),
            optimization_metric="f2"
        )
        calib_time = time.time() - t_calib_0
        print(f"         Calibrated threshold on Validation set: {calibrated_thresh:.4f} (in {calib_time:.3f}s)")

        # Save model artifact with calibrated threshold
        m_filename = f"{m_type}.joblib"
        m_path = os.path.join(output_dir, m_filename)
        detector.save(m_path)
        print(f"         Saved artifact: {m_path}")

        # Final Evaluation on Untouched Test Set
        test_scores = detector.predict_score(X_test_scaled)
        test_preds = detector.predict(X_test_scaled, threshold=calibrated_thresh)
        models_scores[detector.name] = test_scores
        trained_models[m_type] = detector

        eval_res = evaluate_predictions(
            y_true=y_test,
            y_pred=test_preds,
            y_scores=test_scores,
            model_name=detector.name,
            operating_threshold=calibrated_thresh
        )
        eval_res["train_time_sec"] = round(train_duration, 3)
        results_list.append(eval_res)

        # Print detailed report
        print(f"         Held-Out Test Results ({splits['test_samples']:,} samples, {splits['test_frauds']} frauds):")
        print(f"         Operating Threshold: {eval_res['operating_threshold']:.4f}")
        print(f"         Recall (Catch Rate) : {eval_res['recall']:.2%} ({eval_res['true_positives']} of {eval_res['actual_frauds']} frauds caught)")
        print(f"         Precision           : {eval_res['precision']:.2%} ({eval_res['true_positives']} TP, {eval_res['false_positives']} FP)")
        print(f"         F1-Score            : {eval_res['f1_score']:.4f} | F2-Score: {eval_res['f2_score']:.4f}")
        print(f"         Average Precision   : {eval_res['average_precision']:.4f} (Baseline: {eval_res['prevalence']:.4f})")
        print(f"         ROC-AUC             : {eval_res['roc_auc']:.4f}")
        print(f"         Balanced Accuracy   : {eval_res['balanced_accuracy']:.2%} | MCC: {eval_res['matthews_corrcoef']:.4f}")
        print(f"         Confusion Matrix    : TP={eval_res['true_positives']}, FP={eval_res['false_positives']}, FN={eval_res['false_negatives']}, TN={eval_res['true_negatives']}")

    # 5. Save Evaluation Summary and Plots
    print("\n[5/5] Generating evaluation artifacts, JSON summary, and plots...")
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_metadata": metadata,
        "split_config": {
            "split_type": split_type,
            "train_samples": splits["train_samples"],
            "train_frauds": splits["train_frauds"],
            "val_samples": splits["val_samples"],
            "val_frauds": splits["val_frauds"],
            "test_samples": splits["test_samples"],
            "test_frauds": splits["test_frauds"],
        },
        "test_set_size": splits["test_samples"],
        "test_frauds": splits["test_frauds"],
        "results": results_list,
        "imbalance_explanation": explain_class_imbalance(results_list[0]) if results_list else ""
    }

    results_json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"      Saved metrics summary to: {results_json_path}")

    # Generate and save diagnostic figures
    try:
        fig_comp = plot_model_comparison(results_list)
        comp_plot_path = os.path.join(output_dir, "model_comparison.png")
        fig_comp.savefig(comp_plot_path, bbox_inches="tight")
        plt.close(fig_comp)
        print(f"      Saved comparison chart to: {comp_plot_path}")

        if models_scores and len(np.unique(y_test)) > 1:
            fig_curves = plot_roc_pr_curves(y_test.values, models_scores)
            curves_path = os.path.join(output_dir, "roc_pr_curves.png")
            fig_curves.savefig(curves_path, bbox_inches="tight")
            plt.close(fig_curves)
            print(f"      Saved ROC & PR curves to: {curves_path}")

        for res in results_list:
            fig_cm = plot_confusion_matrix(res, title=f"{res['model_name']} Confusion Matrix")
            cm_filename = f"confusion_matrix_{res['model_name'].lower().replace(' ', '_')}.png"
            cm_path = os.path.join(output_dir, cm_filename)
            fig_cm.savefig(cm_path, bbox_inches="tight")
            plt.close(fig_cm)
            print(f"      Saved confusion matrix to: {cm_path}")

    except Exception as e:
        print(f"      [Plotting Warning] Could not save diagnostic figures: {e}")

    total_pipeline_time = time.time() - start_total_time
    print("\n" + "=" * 78)
    print(f" TRAINING & EVALUATION COMPLETED IN {total_pipeline_time:.2f} SECONDS ")
    print("=" * 78)
    return summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Anomaly Detection Models for Fraud Detection")
    parser.add_argument("--data", type=str, default=None, help="Path to credit card CSV dataset")
    parser.add_argument("--samples", type=int, default=None, help="Subsampling size for experimentation")
    parser.add_argument("--contamination", type=float, default=0.002, help="Expected contamination / anomaly rate")
    parser.add_argument("--split-type", choices=["stratified", "chronological"], default="stratified", help="Partitioning strategy")
    parser.add_argument("--normal-only-train", action="store_true", help="Train only on normal transactions (Class 0)")
    parser.add_argument("--models", nargs="+", default=["isolation_forest", "lof", "one_class_svm"], help="Models to train")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model artifacts")
    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data,
        samples=args.samples,
        contamination=args.contamination,
        split_type=args.split_type,
        normal_only_train=args.normal_only_train,
        models_to_train=args.models,
        output_dir=args.output_dir
    )
