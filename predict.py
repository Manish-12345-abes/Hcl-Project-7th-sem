"""
Prediction Service and Batch Screening Engine for Credit Card Fraud Detection.

Supports:
1. Single transaction evaluation with risk scoring, confidence grading, and explainability.
2. Batch CSV transaction screening preserving original identifiers and exporting enriched results.
3. Automatic adoption of validation-calibrated operating thresholds from saved artifacts.
4. Robust input validation rejecting malformed, non-numeric, or missing data.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Union, Optional, Tuple, List

import numpy as np
import pandas as pd

# Ensure local imports work reliably
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.dataset import REQUIRED_FEATURES, TARGET_COLUMN, validate_schema
from src.preprocessing import FraudPreprocessor, validate_transaction_dict
from src.models import BaseAnomalyModel


def get_risk_level(score: float, threshold: float = 0.5) -> str:
    """Categorize continuous anomaly score into actionable risk bands based on operating threshold."""
    if score >= max(0.85, threshold * 1.15):
        return "CRITICAL"
    elif score >= threshold:
        return "HIGH"
    elif score >= threshold * 0.70:
        return "MEDIUM"
    else:
        return "LOW"


def explain_single_transaction_anomaly(
    transaction_dict: Dict[str, Any],
    score: float,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Generate intuitive, human-understandable explanations for why a transaction
    was flagged as anomalous, avoiding black-box ambiguity.
    """
    reasons = []
    amt = float(transaction_dict.get("Amount", 0.0))

    # Check Amount anomalies
    if amt > 2500.0:
        reasons.append(f"Unusually large transaction amount: ${amt:,.2f} (above typical spending profile)")
    elif amt < 1.0:
        reasons.append(f"Suspicious micro-transaction test: ${amt:.2f} (potential card ping / testing)")

    # Check key PCA feature deviations known to correlate with credit card fraud
    for feat in ["V14", "V12", "V10", "V17"]:
        val = float(transaction_dict.get(feat, 0.0))
        if val < -3.0:
            reasons.append(f"Severe negative anomaly in component {feat} (value: {val:.2f}, typical range [-2, +2])")

    for feat in ["V4", "V11"]:
        val = float(transaction_dict.get(feat, 0.0))
        if val > 2.5:
            reasons.append(f"Elevated positive anomaly in component {feat} (value: {val:.2f}, typical range [-2, +2])")

    is_anomaly = (score >= threshold)
    if not reasons and is_anomaly:
        reasons.append("Multi-dimensional geometric outlier in PCA latent space (deviates from normal cluster).")
    elif not is_anomaly:
        reasons.append("Transaction characteristics align closely with historical legitimate spending patterns.")

    return {
        "is_fraud_suspect": is_anomaly,
        "anomaly_score": round(score, 4),
        "decision_threshold": round(threshold, 4),
        "risk_level": get_risk_level(score, threshold),
        "explanation": reasons
    }


class FraudPredictor:
    """
    Integrated prediction engine handling preprocessing and inference.
    """

    def __init__(
        self,
        model_name: str = "isolation_forest",
        models_dir: str = "models",
        threshold: Optional[float] = None
    ):
        self.models_dir = models_dir
        self.model_name = model_name

        preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        model_path = os.path.join(models_dir, f"{model_name}.joblib")

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor artifact not found at '{preprocessor_path}'. "
                "Please run 'python train.py' first to train models and generate pipelines."
            )
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Please train the model or choose an available model artifact."
            )

        self.preprocessor = FraudPreprocessor.load(preprocessor_path)
        self.model = BaseAnomalyModel.load(model_path)

        # Use validation-calibrated threshold from artifact if not explicitly overridden
        if threshold is not None:
            self.threshold = float(threshold)
            self.model.calibrated_threshold_ = float(threshold)
        else:
            self.threshold = float(getattr(self.model, "calibrated_threshold_", 0.50))

    def predict_single(
        self,
        transaction_dict: Dict[str, Any],
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single transaction dictionary.
        """
        is_valid, err = validate_transaction_dict(transaction_dict)
        if not is_valid:
            raise ValueError(f"Transaction validation error: {err}")

        thresh = float(custom_threshold) if custom_threshold is not None else self.threshold

        # Preprocess features (RobustScaler on Amount & Time)
        X_scaled = self.preprocessor.transform(transaction_dict)

        # Predict continuous anomaly score
        score = float(self.model.predict_score(X_scaled)[0])
        raw_score = float(self.model.predict_raw_score(X_scaled)[0])

        # Binary decision based on operating threshold
        is_anomaly = int(score >= thresh)

        explanation = explain_single_transaction_anomaly(transaction_dict, score, thresh)
        return {
            "prediction": is_anomaly,
            "prediction_label": "Fraudulent / Suspicious" if is_anomaly == 1 else "Legitimate",
            "anomaly_score": round(score, 4),
            "raw_anomaly_score": round(raw_score, 4),
            "threshold": round(thresh, 4),
            "risk_level": get_risk_level(score, thresh),
            "reasons": explanation["explanation"],
            "model_used": self.model.name
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        custom_threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Evaluate a batch DataFrame of transactions, preserving original columns and identifiers.
        
        Args:
            df: Input DataFrame containing required features.
            custom_threshold: Optional override for operating threshold.
            
        Returns:
            Tuple of (enriched_df, summary_metrics).
        """
        is_valid, msg = validate_schema(df, is_inference=True)
        if not is_valid:
            raise ValueError(f"Batch dataset validation error: {msg}")

        thresh = float(custom_threshold) if custom_threshold is not None else self.threshold

        # Exclude Class column if present to prevent target leakage
        feature_df = df[REQUIRED_FEATURES].copy()
        X_scaled = self.preprocessor.transform(feature_df)

        scores = self.model.predict_score(X_scaled)
        raw_scores = self.model.predict_raw_score(X_scaled)
        preds = (scores >= thresh).astype(int)

        # Build enriched DataFrame preserving all original columns/IDs
        enriched_df = df.copy()
        enriched_df["Anomaly_Score"] = np.round(scores, 4)
        enriched_df["Raw_Score"] = np.round(raw_scores, 4)
        enriched_df["Is_Anomaly"] = preds
        enriched_df["Prediction"] = np.where(preds == 1, "Fraud / Alert", "Normal")
        enriched_df["Risk_Level"] = [get_risk_level(s, thresh) for s in scores]

        # Calculate batch summary
        total = len(df)
        flagged = int(np.sum(preds == 1))
        summary = {
            "total_transactions": total,
            "flagged_anomalies": flagged,
            "normal_transactions": total - flagged,
            "flagged_percentage": (flagged / total) if total > 0 else 0.0,
            "mean_anomaly_score": float(np.mean(scores)),
            "max_anomaly_score": float(np.max(scores)),
            "model_used": self.model.name,
            "decision_threshold": thresh
        }

        return enriched_df, summary


def generate_sample_transaction(is_fraud: bool = False) -> Dict[str, float]:
    """Helper to generate a clean preset transaction dictionary."""
    rng = np.random.RandomState(42 if is_fraud else 100)
    tx = {"Time": 3600.0}
    for i in range(1, 29):
        if is_fraud:
            if i in (10, 12, 14, 17):
                tx[f"V{i}"] = round(float(rng.normal(-4.8, 1.0)), 4)
            elif i in (4, 11):
                tx[f"V{i}"] = round(float(rng.normal(3.5, 0.8)), 4)
            else:
                tx[f"V{i}"] = round(float(rng.normal(0.0, 1.2)), 4)
        else:
            tx[f"V{i}"] = round(float(rng.normal(0.0, 0.8)), 4)

    tx["Amount"] = 1250.00 if is_fraud else 45.50
    return tx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Fraud / Anomalies in Credit Card Transactions")
    parser.add_argument("--batch", type=str, default=None, help="Path to CSV file for batch inference")
    parser.add_argument("--single", type=str, default=None, help="JSON string of single transaction")
    parser.add_argument("--demo-single", choices=["legit", "fraud"], default=None, help="Run single prediction on preset example")
    parser.add_argument("--model", type=str, default="isolation_forest", help="Model to use (isolation_forest, lof, one_class_svm)")
    parser.add_argument("--threshold", type=float, default=None, help="Optional decision threshold override (defaults to calibrated threshold)")
    parser.add_argument("--output", type=str, default="data/predictions_output.csv", help="Output path for batch predictions")
    args = parser.parse_args()

    predictor = FraudPredictor(model_name=args.model, threshold=args.threshold)
    print(f"[*] Loaded model '{predictor.model.name}' with operating threshold: {predictor.threshold:.4f}")

    if args.batch:
        print(f"[*] Loading batch file: {args.batch}")
        batch_df = pd.read_csv(args.batch)
        enriched, summary = predictor.predict_batch(batch_df)
        enriched.to_csv(args.output, index=False)
        print(f"[OK] Batch processing complete!")
        print(f"     Total Transactions : {summary['total_transactions']:,}")
        print(f"     Flagged Anomalies  : {summary['flagged_anomalies']:,} ({summary['flagged_percentage']:.2%})")
        print(f"     Operating Threshold: {summary['decision_threshold']:.4f}")
        print(f"     Enriched CSV saved : {args.output}")

    elif args.single:
        tx_data = json.loads(args.single)
        res = predictor.predict_single(tx_data)
        print(json.dumps(res, indent=2))

    elif args.demo_single:
        is_fraud = (args.demo_single == "fraud")
        tx_data = generate_sample_transaction(is_fraud=is_fraud)
        print(f"[*] Running inference on preset '{args.demo_single.upper()}' transaction...")
        res = predictor.predict_single(tx_data)
        print(json.dumps(res, indent=2))

    else:
        print("Please provide --batch <csv_path>, --single <json_string>, or --demo-single [legit|fraud].")
