"""
fetch_nhtsa.py
==============
Pulls Toyota vehicle complaints and TSB summaries from the NHTSA public API.
Results are saved to Data/nhtsa/ as JSON files and merged into a single
flat CSV for ingestion by build_vectorstore.py.

Usage:
    python src/fetch_nhtsa.py

NHTSA API docs: https://api.nhtsa.gov/
No API key required.
"""

import os
import json
import time
import requests
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Data", "nhtsa")
os.makedirs(OUT_DIR, exist_ok=True)

BASE_URL = "https://api.nhtsa.gov"

# Models in our training dataset (2014-2020)
TOYOTA_MODELS = [
    "corolla", "camry", "rav4", "highlander",
    "prius", "tacoma", "4runner", "sienna", "yaris", "avalon"
]
YEARS = list(range(2014, 2021))


def fetch_complaints(model: str, year: int) -> list[dict]:
    """Fetch top NHTSA complaints for a Toyota model/year."""
    url = f"{BASE_URL}/complaints/complaintsByVehicle"
    params = {"make": "toyota", "model": model, "modelYear": year}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results
    except Exception as e:
        print(f"  [WARN] complaints {model} {year}: {e}")
        return []


def fetch_tsbs(model: str, year: int) -> list[dict]:
    """Fetch TSB (Technical Service Bulletin) data for a Toyota model/year."""
    url = f"{BASE_URL}/products/vehicle/modelYears/make/TOYOTA/model/{model.upper()}/modelYear/{year}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"  [WARN] tsb {model} {year}: {e}")
        return []


def complaints_to_rag_rows(model: str, year: int, complaints: list[dict]) -> list[dict]:
    """Convert NHTSA complaint records into RAG-ready rows."""
    rows = []
    for c in complaints:
        component = c.get("component", "")
        summary   = c.get("summary", "")
        if not summary:
            continue
        rows.append({
            "dtc_code":    "",               # No DTC — complaint-based
            "engine_code": "Gasoline (common)",
            "vehicle_model": model,
            "vehicle_year":  year,
            "description": f"[NHTSA Complaint] {model.title()} {year} — {component}: {summary[:300]}",
            "source":      "NHTSA complaints API",
            "rag_text": (
                f"Vehicle: Toyota {model.title()} {year} | "
                f"Component: {component} | "
                f"Issue: {summary[:300]}"
            ),
        })
    return rows


def run():
    all_rows = []
    total_fetched = 0

    for model in TOYOTA_MODELS:
        for year in YEARS:
            print(f"Fetching {model} {year} …", end=" ")
            complaints = fetch_complaints(model, year)
            rows = complaints_to_rag_rows(model, year, complaints)
            all_rows.extend(rows)
            total_fetched += len(rows)
            print(f"{len(rows)} complaints")
            time.sleep(0.3)  # polite rate-limiting

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(OUT_DIR, "nhtsa_complaints.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {total_fetched} NHTSA complaint rows → {out_csv}")


if __name__ == "__main__":
    run()
