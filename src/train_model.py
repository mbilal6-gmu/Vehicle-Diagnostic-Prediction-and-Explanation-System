"""
train_model.py
==============
Trains two XGBoost models on preprocessed Toyota sensor data:
  1. Regressor  → failure_risk_score   (continuous 0–1)
  2. Classifier → check_engine_light_likely (binary 0/1)

Outputs:
  - models/xgb_failure_risk.pkl
  - models/xgb_check_engine.pkl
  - models/shap_explainer_risk.pkl
  - models/metrics_report.json

Run after preprocess.py:
    python src/train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, classification_report,
)
from xgboost import XGBRegressor, XGBClassifier
import shap

DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "Data", "processed")
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_splits():
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    yr_train = pd.read_csv(os.path.join(DATA_DIR, "y_risk_train.csv")).squeeze()
    yr_test  = pd.read_csv(os.path.join(DATA_DIR, "y_risk_test.csv")).squeeze()
    yc_train = pd.read_csv(os.path.join(DATA_DIR, "y_cel_train.csv")).squeeze()
    yc_test  = pd.read_csv(os.path.join(DATA_DIR, "y_cel_test.csv")).squeeze()
    return X_train, X_test, yr_train, yr_test, yc_train, yc_test


def train_risk_regressor(X_train, X_test, y_train, y_test):
    print("\n--- Training Failure Risk Regressor ---")
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        eval_metric="rmse",
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    preds = model.predict(X_test).clip(0.0, 1.0)
    rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae   = float(mean_absolute_error(y_test, preds))
    r2    = float(r2_score(y_test, preds))

    print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    path = os.path.join(MODEL_DIR, "xgb_failure_risk.pkl")
    joblib.dump(model, path)
    print(f"  Saved → {path}")

    # SHAP explainer for traceability
    explainer = shap.TreeExplainer(model)
    shap_path = os.path.join(MODEL_DIR, "shap_explainer_risk.pkl")
    joblib.dump(explainer, shap_path)
    print(f"  SHAP explainer saved → {shap_path}")

    return model, {"rmse": rmse, "mae": mae, "r2": r2}


def train_cel_classifier(X_train, X_test, y_train, y_test):
    print("\n--- Training Check-Engine-Light Classifier ---")

    neg   = int((y_train == 0).sum())
    pos   = int((y_train == 1).sum())
    scale = neg / pos if pos > 0 else 1.0
    print(f"  Class balance — neg: {neg}, pos: {pos}, scale_pos_weight: {scale:.2f}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    preds      = model.predict(X_test)
    probs      = model.predict_proba(X_test)[:, 1]
    acc        = float(accuracy_score(y_test, preds))
    f1         = float(f1_score(y_test, preds, zero_division=0))
    auc        = float(roc_auc_score(y_test, probs))

    print(f"  Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    path = os.path.join(MODEL_DIR, "xgb_check_engine.pkl")
    joblib.dump(model, path)
    print(f"  Saved → {path}")

    return model, {"accuracy": acc, "f1": f1, "auc": auc}


def save_metrics(risk_metrics: dict, cel_metrics: dict):
    report = {
        "failure_risk_regressor":      risk_metrics,
        "check_engine_classifier":     cel_metrics,
        "label_note": (
            "Targets are LLM-generated from real sensor data. "
            "Metrics reflect model-to-model reproduction fidelity, "
            "not ground-truth OBD accuracy."
        ),
    }
    path = os.path.join(MODEL_DIR, "metrics_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nMetrics report saved → {path}")


if __name__ == "__main__":
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = load_splits()

    _, risk_metrics = train_risk_regressor(X_train, X_test, yr_train, yr_test)
    _, cel_metrics  = train_cel_classifier(X_train, X_test, yc_train, yc_test)

    save_metrics(risk_metrics, cel_metrics)
    print("\nTraining complete.")
