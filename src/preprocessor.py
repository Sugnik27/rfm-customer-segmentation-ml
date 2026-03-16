"""
Preprocessing utilities for E-commerce Customer Segmentation
- Log transform RFM inputs
- Scale RFM inputs using saved scaler
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# PATHS


BASE_DIR    = Path(__file__).resolve().parents[1]
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"


# LOAD SCALER


def load_scaler():
    """
    Loads the saved StandardScaler from models/
    """
    return joblib.load(SCALER_PATH)


# PREPROCESS SINGLE INPUT


def preprocess_single(recency: float, frequency: float, monetary: float) -> np.ndarray:
    """
    Preprocesses a single customer's RFM values:
    1. Log transformation
    2. Standard scaling using saved scaler
    Returns scaled numpy array ready for prediction
    """
    # Step 1 — Log Transform
    recency_log   = np.log1p(recency)
    frequency_log = np.log1p(frequency)
    monetary_log  = np.log1p(monetary)

    # Step 2 — Create DataFrame with correct column names
    rfm_input = pd.DataFrame([{
        "Recency_log"  : recency_log,
        "Frequency_log": frequency_log,
        "Monetary_log" : monetary_log
    }])

    # Step 3 — Scale using saved scaler
    scaler = load_scaler()
    scaled = scaler.transform(rfm_input)

    return scaled


# PREPROCESS BATCH INPUT


def preprocess_batch(df: pd.DataFrame) -> np.ndarray:
    """
    Preprocesses a batch of customers from uploaded CSV
    Expects columns: Recency, Frequency, Monetary
    Returns scaled numpy array ready for prediction
    """
    df = df.copy()

    # Validate columns
    required_cols = ["Recency", "Frequency", "Monetary"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Log Transform
    df["Recency_log"]   = np.log1p(df["Recency"])
    df["Frequency_log"] = np.log1p(df["Frequency"])
    df["Monetary_log"]  = np.log1p(df["Monetary"])

    # Scale using saved scaler
    scaler = load_scaler()
    scaled = scaler.transform(df[["Recency_log", "Frequency_log", "Monetary_log"]])

    return scaled