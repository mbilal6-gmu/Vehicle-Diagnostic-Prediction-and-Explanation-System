"""
bias_check.py
=============
Checks whether the XGBoost failure risk model produces significantly
different predictions across vehicle models (fairness check).

Method:
  - Run the trained regressor on the full test set
  - Group predictions by vehicle_model (after decoding label encoding)
  - One-way ANOVA: tests if mean risk scores differ significantly between groups
  - Bar chart: mean predicted risk ± 95% CI per vehicle model

Outputs:
  - evidence/bias_by_model.png        (bar chart)
  - evidence/bias_check_report.json   (ANOVA F-stat, p-value, group stats)

Run:
    python src/bias_check.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA_DIR     = os.path.join(os.path.dirname(__file__), "..", "Data", "processed")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
EVIDENCE_DIR = os.path.join(os.path.dirname(__file__), "..", "evidence")
os.makedirs(EVIDENCE_DIR, exist_ok=True)


def main():
    print("Loading test data and model …")
    X_test   = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "..", "Data", "processed", "label_encoders.pkl"))
    model    = joblib.load(os.path.join(MODEL_DIR, "xgb_failure_risk.pkl"))

    # Predict
    preds = np.clip(model.predict(X_test), 0.0, 1.0)

    # Decode vehicle_model column
    le_model = encoders.get("vehicle_model")
    if le_model is not None and "vehicle_model" in X_test.columns:
        model_labels = le_model.inverse_transform(X_test["vehicle_model"].astype(int))
    else:
        model_labels = X_test.get("vehicle_model", pd.Series(["unknown"] * len(X_test)))

    df = pd.DataFrame({"vehicle_model": model_labels, "predicted_risk": preds})

    # Group statistics
    groups = df.groupby("vehicle_model")["predicted_risk"]
    group_stats = groups.agg(["mean", "std", "count"]).reset_index()
    group_stats.columns = ["vehicle_model", "mean_risk", "std_risk", "n"]
    group_stats["ci95"] = 1.96 * group_stats["std_risk"] / np.sqrt(group_stats["n"])
    group_stats = group_stats.sort_values("mean_risk", ascending=False)

    print("\nGroup Statistics:")
    print(group_stats.to_string(index=False))

    # ANOVA
    group_arrays = [grp["predicted_risk"].values for _, grp in df.groupby("vehicle_model")]
    f_stat, p_value = stats.f_oneway(*group_arrays)
    significant = p_value < 0.05

    print(f"\nOne-Way ANOVA: F = {f_stat:.4f}, p = {p_value:.6f}")
    if significant:
        print("  ⚠️  Statistically significant difference in risk scores across vehicle models (p < 0.05)")
        print("     This may indicate model bias — review SHAP values per model for further investigation.")
    else:
        print("  ✅  No statistically significant bias detected across vehicle models (p ≥ 0.05)")

    # ---- Bar chart -------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#d62728" if r > 0.5 else "#1f77b4" for r in group_stats["mean_risk"]]
    bars = ax.bar(
        group_stats["vehicle_model"],
        group_stats["mean_risk"],
        yerr=group_stats["ci95"],
        color=colors,
        edgecolor="white",
        capsize=5,
        error_kw={"elinewidth": 1.5},
    )
    ax.axhline(df["predicted_risk"].mean(), color="black", linestyle="--", linewidth=1.2, label=f"Global mean ({df['predicted_risk'].mean():.3f})")
    ax.set_xlabel("Vehicle Model", fontsize=12)
    ax.set_ylabel("Mean Predicted Failure Risk Score", fontsize=12)
    ax.set_title(
        f"Predicted Failure Risk by Vehicle Model\n"
        f"ANOVA: F={f_stat:.3f}, p={p_value:.4f} — "
        + ("⚠️ Significant difference detected" if significant else "✅ No significant bias"),
        fontsize=12,
    )
    ax.set_ylim(0, min(1.0, group_stats["mean_risk"].max() + group_stats["ci95"].max() + 0.1))
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    out_png = os.path.join(EVIDENCE_DIR, "bias_by_model.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nBias chart saved → {out_png}")

    # ---- JSON report ------------------------------------------------------ #
    report = {
        "anova_f_stat":    round(float(f_stat), 6),
        "anova_p_value":   round(float(p_value), 6),
        "significant":     bool(significant),
        "interpretation":  (
            "Statistically significant difference in mean risk scores across vehicle models. "
            "Review SHAP values per model for potential bias." if significant
            else "No statistically significant bias detected across vehicle models."
        ),
        "group_stats": group_stats.to_dict(orient="records"),
    }
    out_json = os.path.join(EVIDENCE_DIR, "bias_check_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Bias report saved → {out_json}")


if __name__ == "__main__":
    main()
