"""
test_harness.py
===============
Automated evaluation over a 200-row golden dataset sampled from the test split.

Metrics reported:
  - ML Risk Regressor:  RMSE, MAE, R²
  - ML CEL Classifier:  Accuracy, F1, AUC
  - RAG Retrieval:      Precision@1 (correct DTC code in top-1 result)
  - LLM Faithfulness:   Mean faithfulness score across cases
  - End-to-end Latency: Mean seconds per case (RAG + LLM, no ML overhead)

Saves results to:
  - tests/metrics_report.json
  - tests/evaluation_results.csv  (per-row breakdown)

Usage:
    python tests/test_harness.py [--cases 50] [--skip-llm]
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_agent  import retrieve
from src.llm_judge  import generate_report, judge_faithfulness

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "Data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR   = os.path.dirname(__file__)

GOLDEN_N = 200  # rows to sample from test set


def load_test_data(n: int):
    X_test  = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    yr_test = pd.read_csv(os.path.join(DATA_DIR, "y_risk_test.csv")).squeeze()
    yc_test = pd.read_csv(os.path.join(DATA_DIR, "y_cel_test.csv")).squeeze()

    idx = np.random.RandomState(42).choice(len(X_test), size=min(n, len(X_test)), replace=False)
    return X_test.iloc[idx], yr_test.iloc[idx], yc_test.iloc[idx]


def eval_ml_models(X: pd.DataFrame, yr: pd.Series, yc: pd.Series) -> dict:
    risk_model = joblib.load(os.path.join(MODEL_DIR, "xgb_failure_risk.pkl"))
    cel_model  = joblib.load(os.path.join(MODEL_DIR, "xgb_check_engine.pkl"))

    risk_preds = np.clip(risk_model.predict(X), 0.0, 1.0)
    cel_preds  = cel_model.predict(X)
    cel_probs  = cel_model.predict_proba(X)[:, 1]

    # Save per-row evaluation results to CSV
    results_df = X.copy()
    results_df["y_risk_true"]    = yr.values
    results_df["y_risk_pred"]    = risk_preds
    results_df["risk_error"]     = np.abs(risk_preds - yr.values)
    results_df["y_cel_true"]     = yc.values
    results_df["y_cel_pred"]     = cel_preds
    results_df["y_cel_prob"]     = cel_probs
    results_df["cel_correct"]    = (cel_preds == yc.values).astype(int)
    out_csv = os.path.join(OUT_DIR, "evaluation_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"  Per-row results saved → {out_csv}")

    # AUC requires both classes present — extreme imbalance (7/9600) means
    # small samples may contain only negatives; handle gracefully.
    try:
        cel_auc = float(roc_auc_score(yc, cel_probs))
    except ValueError:
        cel_auc = float("nan")
        print("  [NOTE] CEL AUC undefined — only one class present in sample (known imbalance issue)")

    return {
        "risk_rmse":     float(np.sqrt(mean_squared_error(yr, risk_preds))),
        "risk_mae":      float(mean_absolute_error(yr, risk_preds)),
        "risk_r2":       float(r2_score(yr, risk_preds)),
        "cel_accuracy":  float(accuracy_score(yc, cel_preds)),
        "cel_f1":        float(f1_score(yc, cel_preds, zero_division=0)),
        "cel_auc":       cel_auc,
    }


def eval_rag(X: pd.DataFrame, n: int = 50) -> dict:
    """Check that DTC codes in the test data can be retrieved from the vector store."""
    # We don't have ground-truth DTC codes on X directly — use a fixed probe set
    PROBE_CODES = [
        "P0300", "P0171", "P0420", "P0401", "P0011",
        "P1130", "P0301", "P0420", "P0430", "P0172",
        "P1349", "P0335", "P0102", "P0113", "P0441",
        "P0455", "P0446", "P0138", "P0031", "P1100",
    ]
    hits = 0
    for code in PROBE_CODES:
        results = retrieve(dtc_code=code, k=1)
        if results and results[0]["dtc_code"].upper() == code.upper():
            hits += 1
    return {
        "rag_precision_at_1": hits / len(PROBE_CODES),
        "rag_probe_n":        len(PROBE_CODES),
    }


def eval_llm_faithfulness(X: pd.DataFrame, n_cases: int = 20) -> dict:
    """Run end-to-end RAG + LLM on n_cases and measure faithfulness + latency.

    Uses real DTC codes from the probe set so that RAG retrieval returns
    meaningful context chunks — empty DTC queries produce zero-context reports
    which the judge correctly scores 0.0, making the metric uninformative.
    """
    # Real DTC codes that are confirmed present in the vector store
    # (same codes used in eval_rag, Precision@1 = 100%)
    PROBE_CASES = [
        ("P0300", "camry",      2018, "2AZ-FE"),
        ("P0171", "corolla",    2017, "1ZZ-FE"),
        ("P0420", "rav4",       2019, "2GR-FE"),
        ("P0401", "highlander", 2016, "2GR-FE"),
        ("P0011", "camry",      2015, "2AR-FE"),
        ("P1130", "prius",      2018, "2ZR-FXE"),
        ("P0301", "corolla",    2016, "1ZZ-FE"),
        ("P0430", "highlander", 2017, "2GR-FE"),
        ("P0172", "camry",      2019, "2AR-FE"),
        ("P1349", "rav4",       2014, "2AZ-FE"),
        ("P0335", "tacoma",     2016, "2TR-FE"),
        ("P0102", "corolla",    2015, "1ZZ-FE"),
        ("P0113", "camry",      2017, "2AR-FE"),
        ("P0441", "prius",      2016, "2ZR-FXE"),
        ("P0455", "rav4",       2018, "2GR-FE"),
        ("P0446", "highlander", 2015, "2GR-FE"),
        ("P0138", "camry",      2014, "2AZ-FE"),
        ("P0031", "corolla",    2018, "1ZZ-FE"),
        ("P1100", "camry",      2020, "A25A-FKS"),
        ("P0300", "tacoma",     2019, "2TR-FE"),
    ]

    cases = PROBE_CASES[:min(n_cases, len(PROBE_CASES))]
    # Pad with repeated cases if more than 20 requested
    while len(cases) < n_cases:
        cases.extend(PROBE_CASES[:n_cases - len(cases)])

    scores    = []
    latencies = []

    for dtc, vehicle_model, vehicle_year, engine_code in cases:
        t0 = time.time()
        chunks = retrieve(
            dtc_code      = dtc,
            engine_code   = engine_code,
            vehicle_model = vehicle_model,
            k             = 5,
        )
        risk = 0.35   # moderate risk — realistic test value
        cel  = False

        try:
            report = generate_report(
                context_chunks = chunks,
                vehicle_model  = vehicle_model,
                vehicle_year   = vehicle_year,
                engine_code    = engine_code,
                dtc_code       = dtc,
                risk_score     = risk,
                cel_likely     = cel,
            )
            faith    = report.get("faithfulness_score", float("nan"))
            llm_used = report.get("llm_used", "unknown")
        except Exception as e:
            print(f"  [WARN] LLM call failed ({type(e).__name__}): {e}")
            faith    = float("nan")
            llm_used = "failed"

        latencies.append(time.time() - t0)
        scores.append(faith)

    valid_scores = [s for s in scores if not np.isnan(s)]
    failed       = len(scores) - len(valid_scores)

    if failed > 0:
        print(f"\n  ⚠️  {failed}/{len(scores)} LLM calls failed.")
        print("     Check that OPENAI_API_KEY is set in .env, or that Ollama is running.")
        print("     Run: ollama pull deepseek-r1:7b   (for offline fallback)")

    mean_faith = float(np.mean(valid_scores)) if valid_scores else float("nan")
    min_faith  = float(np.min(valid_scores))  if valid_scores else float("nan")

    return {
        "mean_faithfulness":  mean_faith,
        "min_faithfulness":   min_faith,
        "mean_latency_s":     float(np.mean(latencies)),
        "max_latency_s":      float(np.max(latencies)),
        "cases_evaluated":    len(scores),
        "cases_succeeded":    len(valid_scores),
        "cases_failed":       failed,
        "cases_passed_gate":  int(sum(s >= 0.7 for s in valid_scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases",    type=int, default=20,  help="LLM faithfulness cases (default 20)")
    parser.add_argument("--skip-llm", action="store_true",   help="Skip LLM evaluation (faster)")
    args = parser.parse_args()

    print("=== Toyota Diagnostic AI — Test Harness ===\n")

    # Load data
    print("Loading test split …")
    X_test, yr_test, yc_test = load_test_data(GOLDEN_N)
    print(f"  {len(X_test)} test rows loaded\n")

    # ML evaluation
    print("Evaluating ML models …")
    ml_metrics = eval_ml_models(X_test, yr_test, yc_test)
    print(f"  Risk Regressor — RMSE: {ml_metrics['risk_rmse']:.4f} | MAE: {ml_metrics['risk_mae']:.4f} | R²: {ml_metrics['risk_r2']:.4f}")
    cel_auc = ml_metrics['cel_auc']
    auc_str = f"{cel_auc:.4f}" if not np.isnan(cel_auc) else "n/a (one class)"
    print(f"  CEL Classifier — Acc: {ml_metrics['cel_accuracy']:.4f} | F1: {ml_metrics['cel_f1']:.4f} | AUC: {auc_str}\n")

    # RAG evaluation
    print("Evaluating RAG retrieval …")
    rag_metrics = eval_rag(X_test)
    print(f"  Precision@1: {rag_metrics['rag_precision_at_1']:.0%} over {rag_metrics['rag_probe_n']} probe codes\n")

    # LLM faithfulness
    llm_metrics = {}
    if not args.skip_llm:
        print(f"Evaluating LLM faithfulness over {args.cases} cases …")
        llm_metrics = eval_llm_faithfulness(X_test, args.cases)
        _mf = llm_metrics['mean_faithfulness']
        print(f"  Mean faithfulness:  {_mf:.2f}" if not np.isnan(_mf) else "  Mean faithfulness:  n/a (all LLM calls failed)")
        print(f"  Cases passed gate:  {llm_metrics['cases_passed_gate']}/{llm_metrics['cases_evaluated']}")
        print(f"  Mean latency:       {llm_metrics['mean_latency_s']:.1f}s\n")
    else:
        print("LLM evaluation skipped (--skip-llm)\n")

    # Compile report
    full_report = {
        "ml":  ml_metrics,
        "rag": rag_metrics,
        "llm": llm_metrics,
        "notes": {
            "label_origin": "LLM-generated from real sensor data (not ground-truth OBD scan)",
            "test_split":   "20% stratified by vehicle_model",
            "golden_n":     GOLDEN_N,
        },
    }

    out_path = os.path.join(OUT_DIR, "metrics_report.json")
    with open(out_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Metrics saved → {out_path}")

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'Metric':<35} {'Value':>10}")
    print("-" * 47)
    print(f"{'Risk RMSE':<35} {ml_metrics['risk_rmse']:>10.4f}")
    print(f"{'Risk R²':<35} {ml_metrics['risk_r2']:>10.4f}")
    print(f"{'CEL Accuracy':<35} {ml_metrics['cel_accuracy']:>10.4f}")
    print(f"{'CEL F1':<35} {ml_metrics['cel_f1']:>10.4f}")
    print(f"{'RAG Precision@1':<35} {rag_metrics['rag_precision_at_1']:>10.2%}")
    if llm_metrics:
        faith_val = llm_metrics['mean_faithfulness']
        faith_str = f"{faith_val:>10.2f}" if not np.isnan(faith_val) else "   n/a (LLM failed)"
        passed    = llm_metrics['cases_passed_gate']
        total     = llm_metrics['cases_evaluated']
        succeeded = llm_metrics['cases_succeeded']
        print(f"{'LLM Mean Faithfulness':<35} {faith_str}")
        print(f"{'LLM Cases Succeeded':<35} {f'{succeeded}/{total}':>10}")
        print(f"{'LLM Cases Passed Gate (≥0.70)':<35} {f'{passed}/{succeeded if succeeded else total}':>10}")
        print(f"{'LLM Mean Latency (s)':<35} {llm_metrics['mean_latency_s']:>10.1f}")
        if np.isnan(faith_val):
            print("\n  ⚠️  LLM faithfulness could not be computed.")
            print("     → Set OPENAI_API_KEY in your .env file, OR")
            print("     → Run: ollama pull deepseek-r1:7b   (offline fallback)")


if __name__ == "__main__":
    main()
