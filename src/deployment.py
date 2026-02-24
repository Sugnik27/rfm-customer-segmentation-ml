"""
Deployment utilities for E-commerce Customer Segmentation
- Load saved models
- Predict segment for single customer
- Predict segments for batch of customers
- Return segment advice
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any

from src.preprocessor import preprocess_single, preprocess_batch


# PATHS 


BASE_DIR        = Path(__file__).resolve().parents[1]
BEST_MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
ENCODER_PATH    = BASE_DIR / "models" / "label_encoder.joblib"


# SEGMENT ADVICE


SEGMENT_ADVICE = {
    "Champions": {
        "emoji"      : "🏆",
        "color"      : "#2ECC71",
        "description": "Your best customers — recent, frequent, high spenders.",
        "tips"       : [
            "Reward with exclusive loyalty programs and early access",
            "Ask for reviews, testimonials and referrals",
            "Offer premium memberships and VIP benefits",
            "Send personalized thank you messages"
        ]
    },
    "Loyal Customers": {
        "emoji"      : "💛",
        "color"      : "#3498DB",
        "description": "Consistent buyers with moderate engagement.",
        "tips"       : [
            "Upsell and cross-sell higher value products",
            "Send personalized product recommendations",
            "Offer membership upgrade opportunities",
            "Invite to beta test new products"
        ]
    },
    "At-Risk Customers": {
        "emoji"      : "⚠️",
        "color"      : "#F39C12",
        "description": "Previously active customers showing declining engagement.",
        "tips"       : [
            "Send win-back email campaigns immediately",
            "Offer limited time personalized discounts",
            "Ask for feedback on why engagement dropped",
            "Provide incentives to return"
        ]
    },
    "Lost Customers": {
        "emoji"      : "💤",
        "color"      : "#E74C3C",
        "description": "Inactive customers who haven't purchased in a long time.",
        "tips"       : [
            "Launch aggressive re-engagement campaigns",
            "Offer significant one-time discount to return",
            "Survey to understand churn reasons",
            "Consider deprioritizing marketing spend"
        ]
    }
}


# LOAD MODELS (Lazy Loading)


_model   = None
_encoder = None

def load_models():
    """
    Lazy loads models — only loads once per session
    """
    global _model, _encoder
    if _model is None:
        _model   = joblib.load(BEST_MODEL_PATH)
        _encoder = joblib.load(ENCODER_PATH)
    return _model, _encoder


# PREDICT SINGLE CUSTOMER


def predict_single(
    recency  : float,
    frequency: float,
    monetary : float
) -> Dict[str, Any]:
    """
    Predicts segment for a single customer
    Returns segment name, confidence and advice
    """
    model, encoder = load_models()

    # Preprocess
    scaled = preprocess_single(recency, frequency, monetary)

    # Predict
    pred_encoded = model.predict(scaled)[0]
    pred_proba   = model.predict_proba(scaled)[0]
    confidence   = round(max(pred_proba) * 100, 2)

    # Decode segment label
    segment = encoder.inverse_transform([pred_encoded])[0]

    return {
        "segment"   : segment,
        "confidence": confidence,
        "advice"    : SEGMENT_ADVICE[segment],
        "rfm_input" : {
            "Recency"  : recency,
            "Frequency": frequency,
            "Monetary" : monetary
        }
    }


# PREDICT BATCH


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts segments for a batch of customers
    Expects columns: Recency, Frequency, Monetary
    Returns dataframe with Segment column added
    """
    model, encoder = load_models()

    # Preprocess
    scaled = preprocess_batch(df)

    # Predict
    pred_encoded = model.predict(scaled)
    segments     = encoder.inverse_transform(pred_encoded)

    # Add results
    result_df            = df.copy()
    result_df["Segment"] = segments

    return result_df