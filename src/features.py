"""Feature engineering functions for fraud detection pipeline."""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features for fraud detection from raw PaySim transaction data.

    Filters to TRANSFER and CASH_OUT transactions only (the only types where
    fraud occurs), then creates seven engineered features that capture balance
    anomalies and transaction characteristics.

    Args:
        df: Raw PaySim transaction DataFrame with columns: step, type, amount,
            nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest,
            newbalanceDest, isFraud, isFlaggedFraud.

    Returns:
        DataFrame filtered to TRANSFER/CASH_OUT with seven new feature columns:
            - orig_balance_error: Difference between expected and actual origin balance
            - dest_balance_error: Difference between expected and actual dest balance
            - orig_emptied: Boolean flag if origin balance went to zero
            - amount_to_balance_ratio: Transaction amount relative to origin balance
            - dest_unchanged: Boolean flag if destination balance didn't change
            - is_transfer: Boolean flag for TRANSFER transaction type
            - hour: Hour of day extracted from step (step % 24)

    Example:
        >>> from src.features import engineer_features
        >>> df_raw = pd.read_csv("data/PS_20174392719_1491204439457_log.csv")
        >>> df_features = engineer_features(df_raw)
    """
    # Filter to fraud-relevant transaction types only
    df_filtered = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

    # Feature 1: Origin balance error
    # Expected: oldbalanceOrg - amount = newbalanceOrig
    # Error = expected - actual (positive means money "disappeared")
    df_filtered["orig_balance_error"] = (
        df_filtered["oldbalanceOrg"] - df_filtered["amount"]
    ) - df_filtered["newbalanceOrig"]

    # Feature 2: Destination balance error
    # Expected: oldbalanceDest + amount = newbalanceDest
    # Error = expected - actual (positive means money didn't arrive)
    df_filtered["dest_balance_error"] = (
        df_filtered["oldbalanceDest"] + df_filtered["amount"]
    ) - df_filtered["newbalanceDest"]

    # Feature 3: Origin account emptied flag
    # Fraudsters often drain accounts completely
    df_filtered["orig_emptied"] = (df_filtered["newbalanceOrig"] == 0).astype(int)

    # Feature 4: Amount to balance ratio
    # High ratio = transaction is large relative to available funds
    # Add 1 to denominator to handle zero balances without division error
    df_filtered["amount_to_balance_ratio"] = df_filtered["amount"] / (
        df_filtered["oldbalanceOrg"] + 1
    )

    # Feature 5: Destination balance unchanged flag
    # Suspicious if destination received money but balance didn't change
    df_filtered["dest_unchanged"] = (
        df_filtered["oldbalanceDest"] == df_filtered["newbalanceDest"]
    ).astype(int)

    # Feature 6: Is transfer flag
    # TRANSFER has higher fraud rate than CASH_OUT
    df_filtered["is_transfer"] = (df_filtered["type"] == "TRANSFER").astype(int)

    # Feature 7: Hour of day
    # Extract from step (each step = 1 hour in simulation)
    df_filtered["hour"] = df_filtered["step"] % 24

    return df_filtered
