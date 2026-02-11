"""FastAPI application for fraud detection predictions."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.features import engineer_features

app = FastAPI(
    title="Fraud Detection API",
    description="Predicts whether a transaction is fraudulent based on PaySim features.",
    version="1.0.0",
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "fraud_model.joblib",
)
model_artifact = joblib.load(MODEL_PATH)
model = model_artifact["model"]
feature_columns = model_artifact["feature_columns"]


class Transaction(BaseModel):
    """Input transaction data for fraud prediction."""

    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float
    type: Literal["TRANSFER", "CASH_OUT"]
    step: int

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "amount": 181000.0,
                    "oldbalanceOrg": 181000.0,
                    "newbalanceOrig": 0.0,
                    "oldbalanceDest": 0.0,
                    "newbalanceDest": 0.0,
                    "type": "TRANSFER",
                    "step": 1,
                }
            ]
        }
    }


class Prediction(BaseModel):
    """Prediction response with fraud probability and risk level."""

    is_fraud: bool
    fraud_probability: float
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]


@app.get("/health")
def health_check():
    """Check if the API and model are healthy."""
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    """Predict whether a transaction is fraudulent."""
    input_df = pd.DataFrame(
        [
            {
                "step": transaction.step,
                "type": transaction.type,
                "amount": transaction.amount,
                "oldbalanceOrg": transaction.oldbalanceOrg,
                "newbalanceOrig": transaction.newbalanceOrig,
                "oldbalanceDest": transaction.oldbalanceDest,
                "newbalanceDest": transaction.newbalanceDest,
            }
        ]
    )

    features_df = engineer_features(input_df)

    if features_df.empty:
        raise HTTPException(
            status_code=400,
            detail="Transaction type must be TRANSFER or CASH_OUT for fraud prediction.",
        )

    feature_values = features_df[feature_columns].values

    fraud_probability = float(model.predict_proba(feature_values)[0, 1])

    is_fraud = fraud_probability >= 0.3

    if fraud_probability >= 0.7:
        risk_level = "HIGH"
    elif fraud_probability >= 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return Prediction(
        is_fraud=is_fraud,
        fraud_probability=fraud_probability,
        risk_level=risk_level,
    )
