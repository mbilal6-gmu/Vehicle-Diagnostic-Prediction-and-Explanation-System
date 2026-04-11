# Vehicle Diagnostic Prediction and Explanation System
## Project Proposal — AI Design and Deployment Risks | Spring 2026 | George Mason University

**Team:** Cain & Muhammad
**Submission Date:** April 14, 2026
**Mode:** Implementation-Heavy

---

## 1. Project Title & Mode

**Title:** Vehicle Diagnostic Prediction and Explanation System

**Mode:** Implementation-Heavy

This project builds and operationalizes a real AI pipeline — not a conceptual framework. Every component described below is implemented, validated, and running. The deliverable is a working web application backed by measurable controls, not a governance document.

---

## 2. Problem Context

Modern vehicles continuously generate rich ECU sensor data and Diagnostic Trouble Codes (DTCs), but interpreting this data reliably requires either a specialist mechanic or expensive dealership diagnostics. Standard OBD-II readers surface a fault code (e.g., P0300) with no explanation of cause, urgency, or recommended action.

**The specific problem:** There is no accessible, open system that:
1. Predicts failure risk *before* a fault code is thrown (pre-diagnostic layer)
2. Retrieves validated technical context from structured diagnostic knowledge bases
3. Generates a plain-English, traceable explanation with cited sources
4. Checks that explanation against its own sources to detect hallucinations

**Why it matters:** Incorrect or delayed diagnosis leads to avoidable safety risk, unnecessary repair costs, and inefficient technician time. An early-warning system with cited, auditable outputs reduces these risks — and the LLM hallucination problem in automotive contexts is particularly dangerous given that incorrect torque specs or repair sequences can cause physical harm.

---

## 3. Target Entity / Use Case

**Sector:** Automotive diagnostics (consumer + trade)

| Stakeholder | Role | Primary Need |
|---|---|---|
| Common drivers | End users | Understand what's wrong before visiting a shop |
| Car enthusiasts | End users | Technical detail without a dealer visit |
| Independent mechanics | Primary users | Fast pre-diagnosis to triage work orders |
| Vehicle owners | Decision-makers | Know urgency and estimated scope before authorizing repairs |
| Repair shop managers | Decision-makers | Reduce misdiagnosis liability |

**Vehicle scope:** 10 Toyota models (Corolla, Camry, RAV4, Highlander, Prius, Tacoma, 4Runner, Sienna, Yaris, Avalon), model years 2014–2020.

---

## 4. System Scope

The full AI system boundary is defined below:

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INPUTS (Two Modes)                      │
│  Mode A: Plain-text symptom description (any user)              │
│  Mode B: OBD-II sensor readings (mechanic / enthusiast)         │
└────────────────────┬────────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Pre-Diagnostic     │  XGBoost Regressor
          │  ML Layer           │  → failure_risk_score (0.0–1.0)
          │                     │  → urgency classification
          └──────────┬──────────┘
                     │
     ┌───────────────▼───────────────┐
     │  Differential Diagnosis       │  GPT-4o (Mode A only)
     │  Engine (Mode A)              │  → 3–5 candidate fault codes
     │                               │  → confidence scores (0.0–1.0)
     │                               │  → iterative follow-up narrowing
     └───────────────┬───────────────┘
                     │
          ┌──────────▼──────────┐
          │  RAG Retrieval      │  Hybrid: BM25 + ChromaDB vector
          │  Layer              │  → Top-5 chunks from 6,017 DTC records
          │                     │  (toyota-club.net + Libre Diagnostic + SAE)
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  LLM Reasoning      │  GPT-4o (primary)
          │  Layer              │  → Structured JSON report:
          │                     │    {diagnosis, causes, actions,
          │                     │     urgency, confidence, source_refs}
          │                     │  DeepSeek/Ollama (offline fallback)
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐  ◄── CONTROL POINT
          │  Faithfulness       │  GPT-4o judge scores 0.0–1.0
          │  Gate (LLM Judge)   │  → Block output if < 0.70
          │                     │  → Warning banner if 0.70–0.80
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Streamlit Web UI   │  Source citations per claim
          │  Output             │  SHAP feature attributions
          │                     │  Urgency indicator
          │                     │  Disclaimer on every report
          └─────────────────────┘

Human Review Points:
  ├── Faithfulness gate (automated) — user sees warning if < 70%
  ├── SHAP feature importance — mechanic can validate which sensors drove the prediction
  └── Mandatory disclaimer: "Verify with a certified Toyota technician"
```

**Agents:**
- RAG Retrieval Agent: hybrid BM25 + vector search (`src/rag_agent.py`)
- GPT-4o Reasoning Agent: generates and judges diagnostic reports (`src/llm_judge.py`)
- Differential Diagnosis Agent: iteratively narrows fault candidates from symptoms

**Out of scope:** Real-time OBD hardware connection, VIN lookup, parts pricing, multi-brand support.

---

## 5. Risk Framing

Top 8 risks mapped to AI lifecycle stage:

| # | Risk | Lifecycle Stage | Severity | Control Implemented |
|---|---|---|---|---|
| 1 | **Hallucinated repair specs** — LLM invents torque values, part numbers, or procedures not in source data | Inference | 🔴 High | LLM Judge faithfulness gate ≥ 70%; structured JSON forces source citations; temperature = 0 |
| 2 | **LLM-generated training labels** — `failure_risk_score` and `check_engine_light_likely` targets were generated by an LLM, not measured by an OBD scanner | Data Preparation | 🔴 High | Disclosed in `metrics_report.json`; metrics framed as "model-to-model fidelity"; regression used as proxy only |
| 3 | **CEL classifier failure** — extreme class imbalance (7 positives / 9,600 rows) causes F1 = 0 | Model Training | 🟠 Medium | Replaced with `risk_score > 0.5` threshold rule; classifier retained only for AUC reporting; limitation documented |
| 4 | **RAG retrieval miss** — DTC code not found in vector store returns irrelevant context | Retrieval | 🟠 Medium | Hybrid BM25 fallback; DTC exact-match boosting (+0.2); Precision@1 = 100% on 20-code probe set |
| 5 | **LLM offline / quota exceeded** | Deployment | 🟠 Medium | DeepSeek via Ollama fallback; auto-detected in `llm_judge._call_llm()` |
| 6 | **Biased risk scores by vehicle model** — model may systematically over/under-predict for specific models | Data / Training | 🟠 Medium | `src/bias_check.py` — ANOVA + bar chart across 10 vehicle models |
| 7 | **User over-reliance on AI diagnosis** — user skips mechanic based on AI output | Deployment | 🔴 High | Prominent disclaimer on every report; urgency labels (🔴/🟠/🟡/🟢); "verify with technician" required |
| 8 | **Vector store corruption** — ChromaDB fails to load on restart | Operations | 🟡 Low | Persistent store in `vectorstore/chroma_db/`; rebuild in ~2 minutes via `build_vectorstore.py` |

---

## 6. Data Plan

### Datasets Used

| Dataset | Source | Type | Rows | What is Real / Simulated |
|---|---|---|---|---|
| `Toyota_Final_Current.xlsx` | Synthetically generated based on real Toyota OBD-II patterns | Training data | 12,000 | **Real:** sensor readings (RPM, coolant temp, MAF, throttle, fuel trim, etc.) **Simulated:** labels (`failure_risk_score`, `check_engine_light_likely`) were generated by an LLM from sensor patterns |
| `Toyota_RAG_Data.csv` | toyota-club.net (web scrape) | RAG knowledge base | 12,978 | Real: DTC codes and descriptions sourced from Toyota technical documentation |
| `toyota.json` | Libre Automotive Diagnostic — AGPL-3.0 open-source | RAG supplement | 42 | Real: Toyota P1xxx manufacturer-specific fault codes |
| `dtc_db.json` | Libre Automotive Diagnostic — AGPL-3.0 open-source | RAG supplement | 39 | Real: SAE standard OBD-II P0001–P0038 codes |

### Schema Summary (Training Data — 25 features after preprocessing)

**Sensor features:** barometric_pressure_kpa, coolant_temp_celsius, engine_load_percent, engine_rpm, mass_air_flow_gps, intake_air_temp_celsius, vehicle_speed_kmh, short_term_fuel_trim_bank1_percent, engine_runtime_seconds, throttle_position_percent, timing_advance_degrees

**Delta features (rate-of-change):** coolant_temp_change_celsius, engine_rpm_change, vehicle_speed_change_kmh, maf_change_gps

**Categorical (label-encoded):** vehicle_model, trim_or_transmission_code, engine_code

**Temporal:** minute_of_hour, hour_of_day, day_of_week, month_of_year, record_year, vehicle_year

### Data Quality Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Missing sensor values | Filled with column median in `src/preprocess.py` |
| LLM-generated target labels | Disclosed in all reporting; regression score treated as proxy, not ground truth |
| Duplicate DTC records across sources | Deduplicated by `{dtc_code, engine_code}` key in `src/build_vectorstore.py` |
| CEL extreme class imbalance (7 / 9,600) | Documented; replaced with threshold rule in production |
| RAG descriptions are short (1–2 lines) | Supplemented with NHTSA complaint API and Libre Diagnostic data |

### Data Provenance

- `Toyota_Final_Current.xlsx`: Synthetic OBD dataset built from real Toyota vehicle sensor patterns. Labels generated by GPT-4o from sensor readings. No personally identifiable information.
- `Toyota_RAG_Data.csv`: Scraped from toyota-club.net. Used for educational/research purposes. Source URLs retained in dataset for traceability.
- Libre Diagnostic JSON: AGPL-3.0 licensed. No modifications to source data.

---

## 7. Tool Stack

### Commercial Tool: OpenAI GPT-4o (API)

| Attribute | Detail |
|---|---|
| **Why necessary** | Highest-quality structured reasoning for diagnostic report generation; required for faithfulness judging |
| **Capability provided** | JSON-mode output, temperature=0 deterministic responses, long-context understanding of technical RAG chunks |
| **Integration point** | `src/llm_judge.py` — primary LLM for report generation AND the LLM Judge |
| **Risk addressed** | Hallucination detection (judge scores its own output class), diagnostic accuracy |
| **Limitation** | API cost per query; requires internet; rate-limited at high volume |

### Open-Source Tools

| Tool | Role | Why Justified | Output Produced |
|---|---|---|---|
| **XGBoost** | Failure risk regressor + CEL classifier | Interpretable gradient boosting; handles tabular sensor data; native SHAP support | `models/xgb_failure_risk.pkl` |
| **SHAP** | Feature importance + explainability | Per-prediction attributions; satisfies "explainability reports" requirement; identifies which sensors drove the risk score | `evidence/shap_summary_plot.png` |
| **ChromaDB** | Vector store for RAG knowledge base | Persistent local store; no cloud dependency; cosine similarity; fast at ~6,000 documents | `vectorstore/chroma_db/` |
| **LangChain** | RAG orchestration | Standardized retrieval chain; compatible with ChromaDB + OpenAI | `src/rag_agent.py` |
| **rank-bm25** | Keyword search layer | DTC codes are exact alphanumeric strings — keyword matching outperforms pure vector search for codes | Integrated in `src/rag_agent.py` |
| **sentence-transformers** | Local embedding (`all-MiniLM-L6-v2`) | No API cost for embedding; runs fully offline; 384-dim vectors | Embedded in ChromaDB |
| **Streamlit** | Web UI | Python-native deployment; no frontend expertise required; rapid iteration | `app/streamlit_app.py` |

**Offline Fallback:** DeepSeek via Ollama — local LLM that activates automatically when no valid OpenAI API key is present. Satisfies the "maintenance and fallback" requirement.

---

## 8. Implementation Plan

The full system is implemented. Key components:

| Component | File | What it does |
|---|---|---|
| Data preprocessing | `src/preprocess.py` | Cleans Excel, encodes categoricals, stratified 80/20 split |
| RAG knowledge base | `src/build_vectorstore.py` | Merges 3 DTC sources, embeds into ChromaDB |
| ML model training | `src/train_model.py` | XGBoost regressor + classifier; saves SHAP explainer |
| Hybrid retrieval | `src/rag_agent.py` | BM25 + vector re-ranking; DTC exact-match boost |
| LLM reasoning | `src/llm_judge.py` | Report generation + faithfulness judge + differential diagnosis |
| Web interface | `app/streamlit_app.py` | Two-tab Streamlit app (symptom mode + sensor mode) |
| Automated evaluation | `tests/test_harness.py` | ML metrics + RAG precision + LLM faithfulness + latency |
| NHTSA supplement | `src/fetch_nhtsa.py` | Optional: fetches Toyota complaints from NHTSA public API |

**Novel feature:** The "Describe Symptoms" tab implements an iterative differential diagnosis engine. The LLM assigns confidence scores to 3–5 candidate fault codes and asks targeted follow-up questions. Each user response re-scores the candidates — eliminating unlikely ones and surfacing the probable root cause before generating a full report. This mirrors clinical differential diagnosis methodology.

---

## 9. Validation Plan

### Metrics, Targets, and Actual Results

| Metric | Target Threshold | Actual Result | Status |
|---|---|---|---|
| XGBoost Risk RMSE | ≤ 0.05 | **0.0098** | ✅ Pass |
| XGBoost Risk R² | ≥ 0.90 | **0.9935** | ✅ Pass |
| CEL Classifier Accuracy | ≥ 90% | **100%** | ⚠️ Pass (F1 = 0 due to extreme imbalance — disclosed; production replaced with risk_score > 0.5 threshold) |
| CEL Classifier AUC | ≥ 0.80 | **0.9893** | ✅ Pass |
| RAG Precision@1 | ≥ 90% | **100%** (20-code DTC probe set) | ✅ Pass |
| LLM Faithfulness (mean) | ≥ 0.70 | **0.77** (16/20 cases ≥ gate; 80% pass rate) | ✅ Pass |
| Faithfulness Gate | Block if < 0.70 | Implemented in `llm_judge.py` | ✅ |
| End-to-end latency | < 10 seconds | **2.3s** mean (20 cases) | ✅ Pass |
| Bias across vehicle models (ANOVA) | No material disparity | F = 1.95, p = 0.041 — statistically detected, practically negligible (3.2% spread vs. 11–14% within-group std) | ✅ Acceptable |

Full per-row results: `tests/evaluation_results.csv` | Full metrics JSON: `tests/metrics_report.json` | Bias report: `evidence/bias_check_report.json`

### Validation Methods

**Scenario-based testing:** The test harness (`tests/test_harness.py`) runs a 200-row golden dataset (20% stratified holdout, stratified by vehicle_model) through both ML models. A 20-code DTC probe set validates RAG retrieval. An LLM faithfulness suite runs 20 end-to-end cases with real DTC codes and scores each report.

**Deterministic checks:**
- DTC exact-match boost in hybrid retrieval (`rag_agent.py`) — verified by Precision@1 = 100%
- JSON schema enforcement — LLM must return required fields or output is flagged
- Faithfulness gate — report blocked/warned if score < 0.70

**LLM Judge evaluation:** A separate GPT-4o call audits each generated report against its retrieved context chunks, scoring faithfulness 0.0–1.0. Result displayed on every report; warning banner shown if below gate. Mean score: **0.77**; 16/20 test cases passed (80%).

**Bias check (`src/bias_check.py`):** The XGBoost risk model was evaluated across all 10 vehicle models (200 test rows each). One-way ANOVA tested whether mean predicted risk scores differ significantly by model:

| Vehicle Model | Mean Risk Score | Std Dev | n |
|---|---|---|---|
| RAV4 | 0.221 | 0.144 | 200 |
| Prius | 0.217 | 0.122 | 200 |
| Yaris | 0.209 | 0.121 | 200 |
| Camry | 0.203 | 0.118 | 400 |
| 4Runner | 0.199 | 0.114 | 200 |
| Tacoma | 0.198 | 0.108 | 200 |
| Sienna | 0.195 | 0.117 | 200 |
| Avalon | 0.191 | 0.115 | 200 |
| Highlander | 0.191 | 0.131 | 200 |
| Corolla | 0.189 | 0.111 | 400 |

**ANOVA result:** F = 1.9492, p = 0.0414. The test detected a statistically significant difference, but the practical effect is negligible: the between-group spread is only 3.2 percentage points (0.189–0.221) while within-group standard deviations are 11–14%. The model does not materially favor or penalize any vehicle model. Bar chart with 95% CI saved to `evidence/bias_by_model.png`.

**Known limitation (fully disclosed):** Training labels (`failure_risk_score`, `check_engine_light_likely`) are LLM-generated from sensor patterns, not measured by an OBD scanner. Metrics reflect model-to-model reproduction fidelity. This is explicitly stated in `models/metrics_report.json` and in the final report.

---

## 10. Deliverables & Milestones

| Date | Owner | Deliverable | Status |
|---|---|---|---|
| Apr 14 | Cain + Muhammad | Proposal submitted; in-class presentation | 🔄 |
| Apr 14 | Muhammad | Core pipeline (preprocessing → ML → RAG → LLM → UI) | ✅ Done |
| Apr 18 | Muhammad | GitHub repo live; README; .gitignore | 🔄 |
| Apr 20 | Muhammad | SHAP plots; bias check; evidence folder | ✅ Done |
| Apr 20 | Muhammad | Test harness — all metrics confirmed passing | ✅ Done |
| Apr 22 | Cain | Architecture diagram | 🔄 |
| Apr 25 | Cain | Final presentation slides (8–10 slides) | 🔄 |
| Apr 28 | Cain + Muhammad | Final submission: GitHub repo + evidence + live demo | 🔄 |

### Mandatory Deliverables Checklist (per rubric)

- [ ] Public GitHub repository with organized code and artifact folders
- [x] README explaining problem, architecture, tools, setup, metrics
- [ ] Architecture diagram with system boundary, data flow, and control points
- [x] Evidence folder: SHAP plots, bias chart, metrics JSON, evaluation CSV
- [ ] App screenshots in `evidence/screenshots/`
- [ ] Final presentation demonstrating system, tradeoffs, and limitations

---

## 11. Role Allocation

| Team Member | Responsibilities |
|---|---|
| **Muhammad** | ML pipeline (XGBoost, SHAP), LLM integration (GPT-4o, Ollama), RAG retrieval engine, vector store construction, test harness, metrics |
| **Cain** | Data sourcing and cleaning, RAG dataset construction, validation and traceability pipeline, Streamlit UI, presentation, architecture diagram |

Both members understand the full architecture and can defend any design decision independently. Work was not siloed — both reviewed each other's components during integration.

---

## 12. Risks & Fallback Plan

### What Could Fail During the Project

| Risk | Mitigation |
|---|---|
| **LLM hallucination** in diagnostic responses | LLM Judge faithfulness gate (≥ 70%); structured JSON output; temperature = 0; source citation required |
| **Poor RAG retrieval** (irrelevant context returned) | Hybrid BM25 + vector re-ranking; exact-match DTC boost; Precision@1 validated at 100% |
| **Incorrect DTC mapping** (wrong code retrieved) | Metadata filtering by engine_code; deduplication; probe set validation |
| **Inconsistent LLM Judge** | Temperature = 0; same judge model as report generator; fallback score of 0.5 if judge fails |
| **LLM API unavailable** | DeepSeek via Ollama fallback; auto-detected; no code change required |
| **Data access issues** | All datasets stored locally; no live scraping dependency at runtime |
| **Model degradation over time** | Rebuild scripts available; model retrain takes < 5 minutes on existing hardware |
| **CEL classifier failure** | Already resolved — replaced with `risk_score > 0.5` threshold; documented as known limitation |

### Scope Fallback
If the differential diagnosis engine proves unreliable for specific symptom types, the system degrades gracefully to the Sensor Data tab (direct DTC entry + RAG + LLM), which is fully functional and independently validated.
