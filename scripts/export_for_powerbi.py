"""Export fraud detection data as Power BI-ready CSV files.

Loads the PaySim dataset, applies feature engineering, generates model
predictions, and exports three flat CSVs to the powerbi/ directory:
  - transactions.csv: Row-per-transaction fact table with predictions
  - hourly_summary.csv: Pre-aggregated stats by hour and type
  - model_performance.csv: Confusion matrix and model metrics
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from src.features import engineer_features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    PROJECT_ROOT, "data", "PS_20174392719_1491204439457_log.csv"
)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "fraud_model.joblib")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "powerbi")

THRESHOLD = 0.3


def main():
    print("=== Power BI Data Export ===\n")

    # --- Validate inputs ---
    if not os.path.exists(DATA_PATH):
        sys.exit(f"Error: Dataset not found at {DATA_PATH}")
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"Error: Model not found at {MODEL_PATH}")

    # --- Load data ---
    print("Loading dataset...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df_raw):,} transactions ({df_raw.shape[1]} columns)")

    # --- Feature engineering ---
    print("\nApplying feature engineering...")
    df = engineer_features(df_raw)
    print(f"  Filtered to {len(df):,} TRANSFER/CASH_OUT transactions")
    del df_raw  # free memory

    # --- Load model ---
    print("\nLoading model...")
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    print(f"  Model: {artifact.get('model_name', 'Unknown')}")
    stored = artifact.get("metrics", {})
    if stored:
        print(
            f"  Stored metrics: precision={stored.get('precision', '?'):.4f}, "
            f"recall={stored.get('recall', '?'):.4f}, "
            f"f1={stored.get('f1', '?'):.4f}"
        )

    # --- Generate predictions ---
    print(f"\nGenerating predictions (threshold={THRESHOLD})...")
    X = df[feature_columns]
    fraud_prob = model.predict_proba(X)[:, 1]
    predicted = (fraud_prob >= THRESHOLD).astype(int)
    is_fraud = df["isFraud"].values

    risk_level = np.select(
        [fraud_prob >= 0.7, fraud_prob >= 0.3],
        ["HIGH", "MEDIUM"],
        default="LOW",
    )

    confusion = np.select(
        [
            (is_fraud == 1) & (predicted == 1),
            (is_fraud == 0) & (predicted == 1),
            (is_fraud == 1) & (predicted == 0),
            (is_fraud == 0) & (predicted == 0),
        ],
        ["TP", "FP", "FN", "TN"],
    )

    print(f"  Predicted fraud: {predicted.sum():,} ({predicted.mean() * 100:.2f}%)")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        print(f"  {level}: {(risk_level == level).sum():,}")

    # --- Build transactions CSV ---
    transactions = pd.DataFrame(
        {
            "transaction_id": np.arange(len(df)),
            "step": df["step"].values,
            "hour": df["hour"].values,
            "type": df["type"].values,
            "amount": df["amount"].values,
            "name_orig": df["nameOrig"].values,
            "name_dest": df["nameDest"].values,
            "oldbalance_orig": df["oldbalanceOrg"].values,
            "newbalance_orig": df["newbalanceOrig"].values,
            "oldbalance_dest": df["oldbalanceDest"].values,
            "newbalance_dest": df["newbalanceDest"].values,
            "orig_balance_error": df["orig_balance_error"].values,
            "dest_balance_error": df["dest_balance_error"].values,
            "orig_emptied": df["orig_emptied"].values,
            "amount_to_balance_ratio": df["amount_to_balance_ratio"].values,
            "dest_unchanged": df["dest_unchanged"].values,
            "is_fraud": is_fraud,
            "fraud_probability": np.round(fraud_prob, 6),
            "predicted_fraud": predicted,
            "risk_level": risk_level,
            "prediction_correct": (is_fraud == predicted).astype(int),
            "confusion_category": confusion,
        }
    )

    # --- Build hourly summary CSV ---
    hourly = (
        transactions.groupby(["hour", "type"])
        .agg(
            transaction_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum"),
            predicted_fraud_count=("predicted_fraud", "sum"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
            avg_fraud_probability=("fraud_probability", "mean"),
            true_positive_count=(
                "confusion_category",
                lambda x: (x == "TP").sum(),
            ),
            false_positive_count=(
                "confusion_category",
                lambda x: (x == "FP").sum(),
            ),
        )
        .reset_index()
    )
    hourly["fraud_rate_pct"] = (
        hourly["fraud_count"] / hourly["transaction_count"] * 100
    ).round(4)

    # --- Build model performance CSV ---
    tp = int((confusion == "TP").sum())
    fp = int((confusion == "FP").sum())
    tn = int((confusion == "TN").sum())
    fn = int((confusion == "FN").sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    model_performance = pd.DataFrame(
        [
            {"metric": "true_positives", "value": tp, "category": "confusion_matrix"},
            {"metric": "false_positives", "value": fp, "category": "confusion_matrix"},
            {"metric": "true_negatives", "value": tn, "category": "confusion_matrix"},
            {"metric": "false_negatives", "value": fn, "category": "confusion_matrix"},
            {
                "metric": "precision",
                "value": round(precision, 6),
                "category": "model_metric",
            },
            {
                "metric": "recall",
                "value": round(recall, 6),
                "category": "model_metric",
            },
            {
                "metric": "f1_score",
                "value": round(f1, 6),
                "category": "model_metric",
            },
            {
                "metric": "total_transactions",
                "value": len(transactions),
                "category": "dataset_info",
            },
            {
                "metric": "total_fraud",
                "value": int(is_fraud.sum()),
                "category": "dataset_info",
            },
            {
                "metric": "fraud_rate_pct",
                "value": round(float(is_fraud.mean()) * 100, 4),
                "category": "dataset_info",
            },
            {
                "metric": "threshold",
                "value": THRESHOLD,
                "category": "model_metric",
            },
        ]
    )

    # --- Export ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = {
        "transactions.csv": transactions,
        "hourly_summary.csv": hourly,
        "model_performance.csv": model_performance,
    }

    print(f"\nExporting CSVs to {OUTPUT_DIR}/...")
    for filename, dataframe in files.items():
        path = os.path.join(OUTPUT_DIR, filename)
        dataframe.to_csv(path, index=False)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb >= 1:
            print(f"  {filename:<25} {len(dataframe):>12,} rows  ({size_mb:.1f} MB)")
        else:
            size_kb = os.path.getsize(path) / 1024
            print(f"  {filename:<25} {len(dataframe):>12,} rows  ({size_kb:.1f} KB)")

    print("\nDone. Open Power BI Desktop and import from the powerbi/ directory.")
    print("See powerbi/DASHBOARD_GUIDE.md for step-by-step instructions.")


if __name__ == "__main__":
    main()
