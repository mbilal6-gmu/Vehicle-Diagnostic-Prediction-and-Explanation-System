"""
preprocess.py
=============
Cleans and normalizes Toyota_Final_Current.xlsx for XGBoost training.

Outputs:
  - Data/processed/X_train.csv, X_test.csv
  - Data/processed/y_risk_train.csv, y_risk_test.csv   (failure_risk_score)
  - Data/processed/y_cel_train.csv,  y_cel_test.csv    (check_engine_light_likely)
  - Data/processed/feature_names.txt
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "Data")
OUT_DIR  = os.path.join(DATA_DIR, "processed")
os.makedirs(OUT_DIR, exist_ok=True)

EXCEL_PATH = os.path.join(DATA_DIR, "Toyota_Final_Current.xlsx")

# Columns we drop before training (identifiers, raw label strings)
DROP_COLS = [
    "vehicle_identifier",
    "predicted_toyota_dtc_code",   # LLM-generated label — not used as target
    "predicted_issue_category",    # LLM-generated label — not used as target
]

# Categorical columns to label-encode
CAT_COLS = [
    "vehicle_model",
    "trim_or_transmission_code",
    "engine_code",
]

# Regression target
TARGET_RISK  = "failure_risk_score"
# Binary classification target
TARGET_CEL   = "check_engine_light_likely"

# Sensor feature columns (used by prediction endpoint at inference time)
SENSOR_FEATURES = [
    "barometric_pressure_kpa",
    "coolant_temp_celsius",
    "engine_load_percent",
    "engine_rpm",
    "mass_air_flow_gps",
    "intake_air_temp_celsius",
    "vehicle_speed_kmh",
    "short_term_fuel_trim_bank1_percent",
    "engine_runtime_seconds",
    "throttle_position_percent",
    "timing_advance_degrees",
    "coolant_temp_change_celsius",
    "engine_rpm_change",
    "vehicle_speed_change_kmh",
    "maf_change_gps",
]


def load_and_clean(path: str) -> pd.DataFrame:
    print(f"Loading {path} …")
    df = pd.read_excel(path, engine="openpyxl")
    print(f"  Shape: {df.shape}")

    # 1. Drop useless columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # 2. Drop rows where either target is null
    before = len(df)
    df = df.dropna(subset=[TARGET_RISK, TARGET_CEL])
    print(f"  Dropped {before - len(df)} rows with null targets")

    # 3. Clip risk score to [0, 1]
    df[TARGET_RISK] = df[TARGET_RISK].clip(0.0, 1.0)

    # 4. Ensure CEL is integer 0/1
    df[TARGET_CEL] = df[TARGET_CEL].astype(int)

    # 5. Fill remaining numeric NaNs with column median
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # 6. Label-encode categoricals and save encoders
    encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    joblib.dump(encoders, os.path.join(OUT_DIR, "label_encoders.pkl"))
    print(f"  Saved label encoders for: {list(encoders.keys())}")

    return df


def split_and_save(df: pd.DataFrame):
    feature_cols = [
        c for c in df.columns
        if c not in [TARGET_RISK, TARGET_CEL]
    ]

    X = df[feature_cols]
    y_risk = df[TARGET_RISK]
    y_cel  = df[TARGET_CEL]

    # Stratified split by vehicle_model to avoid leakage
    X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
        X, y_risk, y_cel,
        test_size=0.2,
        random_state=42,
        stratify=df["vehicle_model"],
    )

    X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUT_DIR,  "X_test.csv"),  index=False)
    yr_train.to_csv(os.path.join(OUT_DIR, "y_risk_train.csv"), index=False, header=True)
    yr_test.to_csv(os.path.join(OUT_DIR,  "y_risk_test.csv"),  index=False, header=True)
    yc_train.to_csv(os.path.join(OUT_DIR, "y_cel_train.csv"),  index=False, header=True)
    yc_test.to_csv(os.path.join(OUT_DIR,  "y_cel_test.csv"),   index=False, header=True)

    with open(os.path.join(OUT_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_cols))

    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"Features: {len(feature_cols)}")
    print(f"Saved to {OUT_DIR}")


if __name__ == "__main__":
    df = load_and_clean(EXCEL_PATH)
    split_and_save(df)
    print("\nPreprocessing complete.")
