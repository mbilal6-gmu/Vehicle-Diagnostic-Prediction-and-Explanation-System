"""
generate_shap_plots.py
======================
Loads the saved SHAP TreeExplainer and the test dataset to generate
and save two explainability visualizations to evidence/.

Outputs:
  - evidence/shap_summary_plot.png  (bar chart: mean |SHAP| per feature)
  - evidence/shap_beeswarm.png      (beeswarm: SHAP value distribution)

Run:
    python src/generate_shap_plots.py
"""

import os
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import shap

DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "Data", "processed")
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)

SAMPLE_N = 500   # SHAP is slow on large datasets; sample for visualization


def main():
    print("Loading test data and SHAP explainer …")
    X_test   = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    explainer = joblib.load(os.path.join(MODEL_DIR, "shap_explainer_risk.pkl"))

    # Sample for speed
    X_sample = X_test.sample(min(SAMPLE_N, len(X_test)), random_state=42)
    print(f"  Using {len(X_sample)} rows for SHAP calculation …")

    shap_values = explainer.shap_values(X_sample)

    feature_names = list(X_sample.columns)

    # ---- Plot 1: Summary bar chart (mean absolute SHAP values) ------------ #
    print("Generating SHAP summary bar chart …")
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
    )
    plt.title("Feature Importance (Mean |SHAP Value|)\nFailure Risk Score Prediction", fontsize=13)
    plt.tight_layout()
    out1 = os.path.join(EVIDENCE_DIR, "shap_summary_plot.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out1}")

    # ---- Plot 2: Beeswarm (SHAP value distribution per feature) ----------- #
    print("Generating SHAP beeswarm plot …")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        plot_type="dot",
        max_display=15,
        show=False,
    )
    plt.title("SHAP Value Distribution by Feature\n(Red = high feature value increases risk)", fontsize=12)
    plt.tight_layout()
    out2 = os.path.join(EVIDENCE_DIR, "shap_beeswarm.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out2}")

    print("\nSHAP plots complete.")


if __name__ == "__main__":
    main()
