"""
Training module for E-commerce Customer Segmentation
- Loads cleaned data
- Computes RFM features
- Applies log transformation
- Splits data BEFORE scaling to prevent data leakage
- Fits scaler on training data only
- Fits KMeans on training data only
- Trains supervised models with GridSearchCV
- Retrains best model on full dataset
- Saves all models and scaler
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import List, Tuple, Dict

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


# PATHS 


BASE_DIR          = Path(__file__).resolve().parents[1]
CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned_data.csv"
MODEL_DIR         = BASE_DIR / "models"
BEST_MODEL_PATH   = MODEL_DIR / "best_model.joblib"
SCALER_PATH       = MODEL_DIR / "scaler.joblib"
ENCODER_PATH      = MODEL_DIR / "label_encoder.joblib"
KMEANS_PATH       = MODEL_DIR / "kmeans_model.joblib"
FEATURES_PATH     = MODEL_DIR / "feature_columns.json"


# LOAD DATA


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_DATA_PATH, dtype={"CustomerID": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="mixed")
    print(f"Data loaded — Shape: {df.shape}")
    return df


# COMPUTE RFM


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["CustomerID"])
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency = ("InvoiceNo", "nunique"),
        Monetary  = ("TotalPrice", "sum")
    ).reset_index()

    print(f"RFM computed — {rfm.shape[0]} customers")
    return rfm


# GET MODELS AND PARAMS


def get_models_and_params() -> List[Tuple[str, object, Dict[str, List]]]:
    models_and_params = []

    lr = LogisticRegression(max_iter=5000, random_state=42)
    lr_params = {
        "C"     : [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "saga"]
    }
    models_and_params.append(("Logistic Regression", lr, lr_params))

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_params = {
        "n_estimators"     : [50, 100, 200],
        "max_depth"        : [3, 4, 6, None],
        "min_samples_split": [2, 5]
    }
    models_and_params.append(("Random Forest", rf, rf_params))

    xgb = XGBClassifier(random_state=42, eval_metric="mlogloss")
    xgb_params = {
        "n_estimators" : [50, 100, 200],
        "max_depth"    : [3, 4, 6],
        "learning_rate": [0.01, 0.1, 0.2]
    }
    models_and_params.append(("XGBoost", xgb, xgb_params))

    return models_and_params


# TRAIN AND SELECT BEST MODEL


def train_and_select_model(
    X_train, y_train,
    cv=5,
    scoring="f1_weighted"
) -> Tuple[Dict, Dict, Dict]:
    models_and_params = get_models_and_params()

    tuned_models   = {}
    best_params    = {}
    best_cv_scores = {}

    for name, model, params in models_and_params:
        print(f"\nTuning {name}...")

        grid = GridSearchCV(
            model, params,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        tuned_models[name]   = grid.best_estimator_
        best_params[name]    = grid.best_params_
        best_cv_scores[name] = round(grid.best_score_, 4)

        print(f"{name} — Best CV F1: {best_cv_scores[name]}")

    return tuned_models, best_params, best_cv_scores


# MAIN TRAINING PIPELINE


def run_training():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("STARTING TRAINING PIPELINE")
    print("=" * 55)

    # Step 1 — Load
    df = load_data()

    # Step 2 — RFM
    rfm = compute_rfm(df)

    # Step 3 — Log Transform
    rfm["Recency_log"]   = np.log1p(rfm["Recency"])
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"]  = np.log1p(rfm["Monetary"])
    print("Log transformation applied")

    X = rfm[["Recency_log", "Frequency_log", "Monetary_log"]]

    # Save feature columns
    with open(FEATURES_PATH, "w") as f:
        json.dump(X.columns.tolist(), f, indent=2)

    # Step 4 — Split BEFORE scaling ← prevents data leakage
    X_train, X_test = train_test_split(
        X,
        test_size=0.2,
        random_state=42
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Step 5 — Fit scaler on TRAIN only
    # Preserve original index to prevent misalignment
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=["Recency_scaled", "Frequency_scaled", "Monetary_scaled"],
        index=X_train.index    # ← critical — preserves original index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=["Recency_scaled", "Frequency_scaled", "Monetary_scaled"],
        index=X_test.index     # ← critical — preserves original index
    )
    print("Scaler fitted on training data only — no leakage")

    # Step 6 — KMeans on TRAIN only
    kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)

    # Predict segments on full dataset using train-fitted KMeans
    X_full_scaled         = pd.concat([X_train_scaled, X_test_scaled]).sort_index()
    rfm["KMeans_Cluster"] = kmeans.predict(X_full_scaled)

    segment_map = {
        0: "At-Risk Customers",
        1: "Lost Customers",
        2: "Champions",
        3: "Loyal Customers"
    }
    rfm["Segment"] = rfm["KMeans_Cluster"].map(segment_map)
    print("KMeans fitted on training data only — no leakage")
    print(rfm["Segment"].value_counts())

    # Step 7 — Label encode
    # Use .loc with original index — no misalignment
    le     = LabelEncoder()
    le.fit(rfm["Segment"])

    y_train_encoded = le.transform(rfm.loc[X_train.index, "Segment"])
    y_test_encoded  = le.transform(rfm.loc[X_test.index,  "Segment"])
    print(f"Classes: {le.classes_}")

    # Step 8 — Train all models
    tuned_models, best_params, best_cv_scores = train_and_select_model(
        X_train_scaled, y_train_encoded
    )

    # Step 9 — Evaluate and find best
    best_name  = None
    best_score = -1

    for name, model in tuned_models.items():
        y_pred = model.predict(X_test_scaled)
        score  = f1_score(y_test_encoded, y_pred, average="weighted")
        print(f"{name} Test F1: {score:.4f}")

        if score > best_score:
            best_score = score
            best_name  = name

    print(f"\n Best Model: {best_name} (F1: {best_score:.4f})")

    # Step 10 — Retrain best model on FULL dataset
    print(f"\nRetraining {best_name} on full dataset...")

    y_full_encoded = le.transform(rfm["Segment"])

    if best_name == "Logistic Regression":
        final_model = LogisticRegression(
            **best_params[best_name],
            max_iter=5000,
            random_state=42
        )
    elif best_name == "Random Forest":
        final_model = RandomForestClassifier(
            **best_params[best_name],
            random_state=42,
            n_jobs=-1
        )
    elif best_name == "XGBoost":
        final_model = XGBClassifier(
            **best_params[best_name],
            random_state=42,
            eval_metric="mlogloss"
        )

    final_model.fit(X_full_scaled, y_full_encoded)
    print(f"{best_name} retrained on full dataset ({X_full_scaled.shape[0]} customers)")

    # Step 11 — Save everything
    joblib.dump(final_model, BEST_MODEL_PATH)
    joblib.dump(scaler,      SCALER_PATH)
    joblib.dump(le,          ENCODER_PATH)
    joblib.dump(kmeans,      KMEANS_PATH)

    
    print(f"Best Model Saved    : {best_name}")
    print(f"Scaler Saved")
    print(f"Label Encoder Saved")
    print(f"KMeans Saved")
    print(f"Features Saved")
    


if __name__ == "__main__":
    run_training()