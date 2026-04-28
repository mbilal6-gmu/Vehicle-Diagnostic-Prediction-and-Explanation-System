"""
streamlit_app.py
================
Toyota AI Diagnostic Assistant — web interface.

Two modes:
  Tab 1 — Describe Symptoms: plain-text, iterative differential diagnosis
  Tab 2 — Enter Sensor Data: OBD sensor readings → ML risk score + report

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_agent  import retrieve
from src.llm_judge  import generate_report, generate_candidates, update_candidates

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR    = os.path.join(os.path.dirname(__file__), "..", "Data", "processed")
LEMON_IMGS  = Path(os.path.dirname(__file__)) / ".." / "Data" / "lemon" / "images"

VEHICLE_MODELS = [
    "corolla", "camry", "rav4", "highlander",
    "prius", "tacoma", "4runner", "sienna", "yaris", "avalon",
]
YEARS = [2020]

# Maps each model to its compatible 2020 engine codes (first entry = default)
MODEL_ENGINE_MAP: dict[str, list[str]] = {
    "corolla":    ["2ZR-FE"],
    "yaris":      ["2ZR-FE"],
    "prius":      ["2ZR-FXE"],
    "camry":      ["A25A-FKS", "A25A-FXS", "2AR-FE", "2AR-FXE", "2GR-FE"],
    "rav4":       ["A25A-FKS", "A25A-FXS", "2GR-FE"],
    "highlander": ["2GR-FKS", "A25A-FXS"],
    "avalon":     ["2GR-FKS", "A25A-FXS", "2AR-FXE"],
    "sienna":     ["2GR-FKS"],
    "tacoma":     ["2TR-FE", "1GR-FE"],
    "4runner":    ["1GR-FE"],
}
URGENCY_COLOR = {"immediate": "🔴", "soon": "🟠", "monitor": "🟡", "none": "🟢", "unknown": "⚪"}


# --------------------------------------------------------------------------- #
# Cached resource loading
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_models():
    models = {}
    risk_path = os.path.join(MODEL_DIR, "xgb_failure_risk.pkl")
    enc_path  = os.path.join(DATA_DIR,  "label_encoders.pkl")
    if os.path.exists(risk_path):
        models["risk"]     = joblib.load(risk_path)
    if os.path.exists(enc_path):
        models["encoders"] = joblib.load(enc_path)
    return models


def predict_risk(models: dict, row: dict) -> float:
    if not models.get("risk"):
        return 0.5
    feature_path = os.path.join(DATA_DIR, "feature_names.txt")
    if not os.path.exists(feature_path):
        return 0.5
    with open(feature_path) as f:
        feature_names = [l.strip() for l in f.readlines()]
    df = pd.DataFrame([row])
    for col, le in models.get("encoders", {}).items():
        if col in df.columns:
            val = str(df[col].iloc[0])
            df[col] = le.transform([val if val in le.classes_ else le.classes_[0]])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return float(np.clip(models["risk"].predict(df[feature_names])[0], 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Shared sidebar — vehicle info only
# --------------------------------------------------------------------------- #
def vehicle_sidebar():
    with st.sidebar:
        st.header("Vehicle Information")
        model  = st.selectbox("Model", VEHICLE_MODELS, index=1)
        year   = st.selectbox("Year",  YEARS,          index=0)

        engine_options = ["Unknown"] + MODEL_ENGINE_MAP.get(model, [])
        engine = st.selectbox(
            "Engine Code",
            engine_options,
            index=0,
            help=f"Engine codes available for 2020 {model.title()}",
        )
        st.divider()
        _show_system_status()
    return model, year, engine


def _show_system_status():
    st.caption("**System Status**")
    models  = load_models()
    vs_path = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "chroma_db")
    st.caption("Risk Model: "   + ("✅" if models.get("risk") else "❌ run train_model.py"))
    st.caption("Vector Store: " + ("✅" if os.path.exists(vs_path) else "❌ run build_vectorstore.py"))

    # LLM availability
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key.startswith("sk-"):
        st.caption("LLM: ✅ OpenAI GPT-4o")
    else:
        # Check Ollama
        try:
            import requests as _req
            r = _req.get("http://localhost:11434/api/tags", timeout=2)
            ollama_ok = r.status_code == 200
        except Exception:
            ollama_ok = False

        if ollama_ok:
            st.caption("LLM: 🟡 DeepSeek (Ollama) — slower, ~90–130s")
        else:
            st.caption("LLM: ❌ No LLM configured")
            st.warning(
                "**No LLM available.** Choose one:\n\n"
                "**Option A — OpenAI (fast)**\n"
                "Add to `.env`:\n```\nOPENAI_API_KEY=sk-...\n```\n\n"
                "**Option B — DeepSeek offline (free)**\n"
                "```\n"
                "# 1. Install Ollama: https://ollama.com\n"
                "# 2. Pull model:\n"
                "ollama pull deepseek-r1:7b\n"
                "# 3. Start server:\n"
                "ollama serve\n"
                "```",
                icon="⚠️",
            )


# --------------------------------------------------------------------------- #
# Tab 1 — Symptom-based iterative diagnosis
# --------------------------------------------------------------------------- #
def _confidence_bar(confidence: float) -> str:
    filled = round(confidence * 10)
    return "█" * filled + "░" * (10 - filled)


def render_candidates(candidates: list[dict]):
    """Display candidate fault codes with confidence bars."""
    active = [c for c in candidates if not c.get("eliminated")]
    elim   = [c for c in candidates if c.get("eliminated")]

    if active:
        for c in sorted(active, key=lambda x: x["confidence"], reverse=True):
            pct = int(c["confidence"] * 100)
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**`{c['dtc_code']}`** — {c['issue']}")
                st.markdown(
                    f"<span style='font-family:monospace;color:#1f77b4'>{_confidence_bar(c['confidence'])}</span> "
                    f"**{pct}%**",
                    unsafe_allow_html=True,
                )
                st.caption(c.get("reasoning", ""))
            with col_b:
                st.metric("", f"{pct}%")
            st.divider()

    if elim:
        with st.expander(f"❌ {len(elim)} eliminated candidate(s)"):
            for c in elim:
                st.markdown(
                    f"~~`{c['dtc_code']}` — {c['issue']}~~ *({int(c['confidence']*100)}%)* — {c.get('reasoning', '')}"
                )


def symptom_tab(vehicle_model: str, vehicle_year: int, engine_code: str):
    st.subheader("Describe your car's problem")
    st.caption("Type what you're experiencing — no technical knowledge needed. The assistant will ask follow-up questions to narrow down the issue.")

    # ---- Session state init ----------------------------------------------- #
    if "diag_round"   not in st.session_state: st.session_state.diag_round   = 0
    if "candidates"   not in st.session_state: st.session_state.candidates   = []
    if "follow_up"    not in st.session_state: st.session_state.follow_up    = ""
    if "history"      not in st.session_state: st.session_state.history      = []
    if "show_report"  not in st.session_state: st.session_state.show_report  = False

    # ---- Reset button ------------------------------------------------------- #
    if st.session_state.diag_round > 0:
        if st.button("🔄 Start Over", key="reset_btn"):
            for key in ["diag_round", "candidates", "follow_up", "history", "show_report", "selected_candidate"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # ---- Round 1: initial symptom input ------------------------------------- #
    if st.session_state.diag_round == 0:
        symptom_input = st.text_area(
            "What's happening with your car?",
            placeholder="e.g. My Camry shakes badly when idling at traffic lights and sometimes stalls. It only happens when the engine is cold.",
            height=120,
        )
        if st.button("🔍 Start Diagnosis", type="primary", disabled=not symptom_input.strip()):
            with st.spinner("Analysing symptoms …"):
                result = generate_candidates(
                    symptom_text  = symptom_input.strip(),
                    vehicle_model = vehicle_model,
                    vehicle_year  = vehicle_year,
                    engine_code   = engine_code,
                )
            st.session_state.candidates  = result.get("candidates", [])
            st.session_state.follow_up   = result.get("follow_up", "")
            st.session_state.history     = [symptom_input.strip()]
            st.session_state.diag_round  = 1
            st.session_state.show_report = False
            st.rerun()
        return

    # ---- Rounds 1+: show candidates ---------------------------------------- #
    st.markdown(f"**Round {st.session_state.diag_round} — Possible Issues**")
    render_candidates(st.session_state.candidates)

    active = [c for c in st.session_state.candidates if not c.get("eliminated")]
    top_confidence = active[0]["confidence"] if active else 0.0

    st.divider()

    # ---- Follow-up + report section side-by-side --------------------------- #
    left_col, right_col = st.columns([3, 2])

    with left_col:
        if st.session_state.follow_up and not st.session_state.show_report:
            st.info(f"💬 **Follow-up:** {st.session_state.follow_up}")
            new_info = st.text_area(
                "Your answer / additional details:",
                placeholder="e.g. Yes the check engine light is on, only when cold.",
                height=80,
                key=f"new_info_{st.session_state.diag_round}",
            )
            if st.button("📝 Update Diagnosis", disabled=not new_info.strip()):
                with st.spinner("Refining diagnosis …"):
                    result = update_candidates(
                        existing_candidates = st.session_state.candidates,
                        new_info            = new_info.strip(),
                        vehicle_model       = vehicle_model,
                        vehicle_year        = vehicle_year,
                        engine_code         = engine_code,
                        history             = st.session_state.history,
                    )
                st.session_state.candidates  = result.get("candidates", st.session_state.candidates)
                st.session_state.follow_up   = result.get("follow_up", "")
                st.session_state.history.append(new_info.strip())
                st.session_state.diag_round += 1
                st.session_state.show_report = False
                st.rerun()

    with right_col:
        st.markdown("**Generate a report for:**")
        if active:
            # Let user pick any active candidate — default to highest confidence
            candidate_labels = [
                f"{c['dtc_code']} — {c['issue']} ({int(c['confidence']*100)}%)"
                for c in active
            ]
            selected_label = st.radio(
                "Select issue to report on:",
                candidate_labels,
                index=0,
                key="candidate_radio",
                label_visibility="collapsed",
            )
            selected_idx = candidate_labels.index(selected_label)
            selected_candidate = active[selected_idx]

            if top_confidence >= 0.70:
                st.success(f"✅ {int(top_confidence*100)}% confidence reached")

            if st.button("📋 Generate Full Report", type="primary", use_container_width=True):
                st.session_state.selected_candidate = selected_candidate
                st.session_state.show_report = True
                st.rerun()

    # ---- Full report display ----------------------------------------------- #
    if st.session_state.show_report:
        top = st.session_state.get("selected_candidate") or (active[0] if active else None)
        if not top:
            st.warning("No candidate selected.")
            return
        symptom_summary = " | ".join(st.session_state.history)

        with st.spinner("Generating full diagnostic report …"):
            t0     = time.time()

            # Retrieve for all active candidates so the report covers each issue.
            # Top candidate gets k=5 slots; remaining active candidates share k=2 each.
            seen_docs: set[str] = set()
            chunks: list[dict] = []

            def _add_chunks(new_chunks):
                for c in new_chunks:
                    if c["document"] not in seen_docs:
                        seen_docs.add(c["document"])
                        chunks.append(c)

            _add_chunks(retrieve(
                dtc_code      = top["dtc_code"],
                engine_code   = engine_code if engine_code != "Unknown" else "",
                vehicle_model = vehicle_model,
                vehicle_year  = vehicle_year,
                free_text     = symptom_summary,
                k             = 5,
            ))
            for other in active:
                if other["dtc_code"] == top["dtc_code"]:
                    continue
                _add_chunks(retrieve(
                    dtc_code      = other["dtc_code"],
                    engine_code   = engine_code if engine_code != "Unknown" else "",
                    vehicle_model = vehicle_model,
                    vehicle_year  = vehicle_year,
                    k             = 2,
                ))
            chunks.sort(key=lambda x: x["combined_score"], reverse=True)
            chunks = chunks[:10]
            report = generate_report(
                context_chunks = chunks,
                vehicle_model  = vehicle_model,
                vehicle_year   = vehicle_year,
                engine_code    = engine_code,
                dtc_code       = top["dtc_code"],
                risk_score     = top["confidence"],
                cel_likely     = top["confidence"] > 0.5,
            )
            latency = time.time() - t0

        with st.expander(f"📚 Retrieved {len(chunks)} knowledge chunks"):
            for i, c in enumerate(chunks, 1):
                year_badge = f" ⚠️ *from {c['model_year']} docs*" if c.get("year_mismatch") and c.get("model_year") else ""
                st.markdown(
                    f"**[{i}]** `{c['dtc_code'] or 'N/A'}` — {c['description'] or c['document'][:120]}{year_badge}  \n"
                    f"*{c['source']}* | Score: {c['combined_score']:.3f}"
                )
        _render_full_report(report, chunks, latency)


# --------------------------------------------------------------------------- #
# Tab 2 — Sensor data mode
# --------------------------------------------------------------------------- #

def _build_sensor_query(rpm: int, coolant: int, maf: float, load: int,
                        fuel_trim: float, throttle: int, speed: int,
                        intake: int, risk_score: float) -> str:
    """
    Derive a diagnostic free-text query from sensor readings so retrieve()
    returns relevant chunks even when no DTC code is supplied.
    Each condition maps to the language Toyota service docs use.
    """
    parts: list[str] = []

    # Cooling system
    if coolant > 105:
        parts.append("engine overheating high coolant temperature cooling system fan P0217")
    elif coolant < 60:
        parts.append("engine not reaching operating temperature thermostat stuck open P0128 coolant temperature below threshold")

    # Fuel trim — lean or rich running
    if fuel_trim > 10:
        parts.append("system too lean positive fuel trim P0171 vacuum leak MAF sensor air fuel ratio")
    elif fuel_trim < -10:
        parts.append("system too rich negative fuel trim P0172 fuel injector fuel pressure")

    # Idle quality
    if rpm > 1300 and speed == 0 and load < 20:
        parts.append("high idle RPM rough idle IAC throttle body P0507")
    elif rpm < 500 and speed == 0:
        parts.append("engine stalling low RPM idle P0505 P0300")

    # High load at low speed (dragging / slipping)
    if load > 80 and speed < 10:
        parts.append("high engine load low speed transmission brake drag")

    # Generic fallback when risk is elevated but nothing specific stands out
    if not parts and risk_score > 0.30:
        parts.append("engine performance diagnostic sensor fault")

    return " ".join(parts)


def sensor_tab(vehicle_model: str, vehicle_year: int, engine_code: str):
    models = load_models()

    st.subheader("OBD Sensor Readings")
    st.caption("Enter readings from your OBD scanner. Leave at default if unknown.")

    dtc_code = st.text_input(
        "DTC / TSB Code (optional)",
        placeholder="e.g. P0300  or  T-SB-0009-23",
        help="Enter a DTC fault code (P0300), a Toyota TSB reference (T-SB-0009-23), or leave blank to use sensor data only.",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        rpm      = st.number_input("Engine RPM",           0,    8000,  800)
        coolant  = st.number_input("Coolant Temp (°C)",   -40,    130,   90)
        maf      = st.number_input("MAF (g/s)",           0.0,  300.0,  3.5, step=0.1)
        load     = st.slider("Engine Load (%)",             0,    100,   25)
    with col2:
        throttle = st.slider("Throttle Position (%)",       0,    100,   10)
        speed    = st.number_input("Vehicle Speed (km/h)",  0,    250,    0)
        fuel_trim= st.number_input("Short-Term Fuel Trim (%)", -30.0, 30.0, 0.0, step=0.5)
        intake   = st.number_input("Intake Air Temp (°C)", -40,   70,   25)
    with col3:
        baro     = st.number_input("Barometric Pressure (kPa)", 60.0, 110.0, 101.3, step=0.1)
        runtime  = st.number_input("Engine Runtime (s)",    0, 86400, 300)
        timing   = st.number_input("Timing Advance (°)",  -20.0, 60.0,  10.0, step=0.5)

    run_btn = st.button("🔍 Run Diagnosis", type="primary")

    if not run_btn:
        return

    sensor_row = {
        "vehicle_model": vehicle_model, "vehicle_year": vehicle_year,
        "engine_code": engine_code if engine_code != "Unknown" else "Gasoline (common)",
        "engine_rpm": rpm, "coolant_temp_celsius": coolant, "mass_air_flow_gps": maf,
        "engine_load_percent": load, "throttle_position_percent": throttle,
        "vehicle_speed_kmh": speed, "short_term_fuel_trim_bank1_percent": fuel_trim,
        "intake_air_temp_celsius": intake, "barometric_pressure_kpa": baro,
        "engine_runtime_seconds": runtime, "timing_advance_degrees": timing,
        "coolant_temp_change_celsius": 0.0, "engine_rpm_change": 0.0,
        "vehicle_speed_change_kmh": 0.0, "maf_change_gps": 0.0,
        "trim_or_transmission_code": "le", "minute_of_hour": 0,
        "hour_of_day": 12, "day_of_week": 1, "month_of_year": 4,
        "record_year": vehicle_year,
    }

    with st.spinner("Running ML pre-diagnostic …"):
        t0         = time.time()
        risk_score = predict_risk(models, sensor_row)
        ml_lat     = time.time() - t0

    # Rule-based overrides: ML model may underestimate risk for clearly
    # out-of-range readings because training data rarely captures extremes.
    risk_boost  = 0.0
    sensor_cel  = False

    # Out-of-range sensor overrides
    if coolant > 105:
        risk_boost = max(risk_boost, 0.40)   # overheating → high risk
        sensor_cel = True                     # P0128 / P0217 likely
    if coolant < 60:
        risk_boost = max(risk_boost, 0.30)   # thermostat stuck open
        sensor_cel = True                     # P0128 likely
    if rpm > 1300 and speed == 0:
        risk_boost = max(risk_boost, 0.20)   # high idle → P0507
        sensor_cel = True
    if abs(fuel_trim) > 15:
        risk_boost = max(risk_boost, 0.25)   # lean/rich running → P0171/P0172
        sensor_cel = True

    # Normalise code input — accept loose TSB formats like TSB0008-21
    from src.rag_agent import detect_code_type, _normalize_tsb
    dtc_raw   = dtc_code.strip() if dtc_code else ""
    dtc_clean = ""
    if dtc_raw:
        dtc_clean, _code_type = detect_code_type(dtc_raw)
        if _code_type == "tsb" and dtc_clean.upper() != dtc_raw.upper():
            st.info(f"TSB format normalised: **{dtc_raw}** → **{dtc_clean}**")

    # DTC presence overrides: a stored DTC means CEL is on; risk reflects severity
    if dtc_clean and not dtc_clean.upper().startswith("T-SB"):
        first_char = dtc_clean[0] if dtc_clean else ""
        if first_char == "P":
            # P0xxx generic powertrain — higher severity
            # P1xxx/P2xxx manufacturer-specific — moderate severity
            if dtc_clean.startswith("P0"):
                risk_boost = max(risk_boost, 0.45)
            else:
                risk_boost = max(risk_boost, 0.30)
            sensor_cel = True   # any P-code triggers CEL
        elif first_char in ("B", "C", "U"):
            risk_boost = max(risk_boost, 0.15)
            # B/C/U may or may not trigger CEL; set it on conservatively
            sensor_cel = True

    risk_score = min(1.0, risk_score + risk_boost)
    cel_likely  = (risk_score > 0.5) or sensor_cel

    urgency = ("immediate" if risk_score > 0.75 else "soon" if risk_score > 0.5
               else "monitor" if risk_score > 0.25 else "none")
    c1, c2, c3 = st.columns(3)
    c1.metric("Failure Risk Score", f"{risk_score:.0%}")
    c1.progress(risk_score)
    c2.metric("Check Engine Light", "⚠️ Likely ON" if cel_likely else "✅ Not expected")
    c3.metric("Urgency", f"{URGENCY_COLOR.get(urgency)} {urgency.title()}")

    with st.spinner("Retrieving diagnostic knowledge …"):
        t1          = time.time()
        sensor_query = _build_sensor_query(
            rpm, coolant, maf, load, fuel_trim, throttle, speed, intake, risk_score
        )
        chunks = retrieve(
            dtc_code      = dtc_clean,
            engine_code   = engine_code if engine_code != "Unknown" else "",
            vehicle_model = vehicle_model,
            vehicle_year  = vehicle_year,
            free_text     = sensor_query,
            k             = 5,
        )
        rag_lat = time.time() - t1

    with st.expander(f"📚 Retrieved {len(chunks)} knowledge chunks"):
        for i, c in enumerate(chunks, 1):
            year_badge = f" ⚠️ *from {c['model_year']} docs*" if c.get("year_mismatch") and c.get("model_year") else ""
            st.markdown(
                f"**[{i}]** `{c['dtc_code'] or 'N/A'}` — {c['description'] or c['document'][:120]}{year_badge}  \n"
                f"*[{c['source']}]({c['source_url'] or '#'})* | Score: {c['combined_score']:.3f}"
            )

    with st.spinner("Generating diagnostic report …"):
        t2 = time.time()
        try:
            report  = generate_report(
                context_chunks=chunks, vehicle_model=vehicle_model,
                vehicle_year=vehicle_year, engine_code=engine_code,
                dtc_code=dtc_clean,
                risk_score=risk_score, cel_likely=risk_score > 0.5,
                sensor_readings={
                    "coolant_temp_c":    coolant,
                    "engine_rpm":        rpm,
                    "fuel_trim_pct":     fuel_trim,
                    "engine_load_pct":   load,
                    "maf_gps":           maf,
                    "vehicle_speed_kmh": speed,
                },
            )
            llm_lat = time.time() - t2
        except RuntimeError as e:
            st.error(f"LLM unavailable: {e}")
            st.stop()

    _render_full_report(report, chunks, ml_lat + rag_lat + llm_lat)


# --------------------------------------------------------------------------- #
# LEMON reference diagram display
# --------------------------------------------------------------------------- #
def _render_reference_diagrams(chunks: list[dict]):
    """
    Show wiring diagrams / component photos from the LEMON manual database.
    Only appears when retrieved chunks include LEMON pages that have images.
    Images are pre-extracted PNG files in Data/lemon/images/ by extract_lemon.py.
    """
    # Collect unique image filenames from all retrieved LEMON chunks
    image_files: list[Path] = []
    seen: set[str] = set()

    for chunk in chunks:
        if chunk.get("source") != "LEMON Vehicle Manual Database":
            continue
        raw_keys = chunk.get("image_keys", "")
        if not raw_keys:
            continue
        for fname in raw_keys.split(","):
            fname = fname.strip()
            if not fname or fname in seen:
                continue
            img_path = LEMON_IMGS / fname
            if img_path.exists():
                image_files.append(img_path)
                seen.add(fname)

    if not image_files:
        return   # no LEMON images — skip expander entirely

    with st.expander(f"📷 Reference Diagrams ({len(image_files)} image{'s' if len(image_files) != 1 else ''} from LEMON manuals)"):
        st.caption("Wiring diagrams and component photos from the vehicle service manual.")
        # Display in rows of 3
        cols_per_row = 3
        for row_start in range(0, len(image_files), cols_per_row):
            row_imgs = image_files[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_imgs))
            for col, img_path in zip(cols, row_imgs):
                with col:
                    try:
                        st.image(
                            str(img_path),
                            caption=img_path.stem,
                            use_column_width=True,
                        )
                    except Exception:
                        st.caption(f"⚠️ Could not load {img_path.name}")


# --------------------------------------------------------------------------- #
# Shared report renderer
# --------------------------------------------------------------------------- #
def _render_full_report(report: dict, chunks: list[dict], latency: float):
    st.divider()
    st.subheader("📋 Diagnostic Report")

    faith  = report.get("faithfulness_score", 0.0)
    passed = report.get("passed_gate", False)
    if not passed:
        st.warning(
            f"⚠️ Faithfulness score {faith:.0%} is below the 70% threshold. "
            "The report may contain unsupported claims — review carefully."
        )

    st.markdown(f"**Diagnosis:** {report.get('diagnosis', 'N/A')}")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Possible Causes:**")
        for cause in report.get("causes", []):
            st.markdown(f"- {cause}")
    with col_r:
        st.markdown("**Recommended Actions:**")
        for action in report.get("actions", []):
            st.markdown(f"- {action}")

    st.divider()
    urgency_val = report.get("urgency", "unknown")
    urgency_label = f"{URGENCY_COLOR.get(urgency_val, '⚪')} {urgency_val.title()}"
    ca, cb, cc, cd, ce = st.columns(5)
    ca.metric("Faithfulness",   f"{faith:.0%}")
    cb.metric("LLM Confidence", f"{report.get('confidence', 0.0):.0%}")
    cc.metric("Urgency",        urgency_label)
    cd.metric("LLM Used",       report.get("llm_used", "—").upper())
    ce.metric("Latency",        f"{latency:.1f}s")

    with st.expander("🔗 Source Traceability"):
        year_mismatches = [c for c in chunks if c.get("year_mismatch") and c.get("model_year")]
        if year_mismatches:
            mismatch_years = sorted(set(str(c["model_year"]) for c in year_mismatches))
            st.info(
                f"ℹ️ {len(year_mismatches)} source chunk(s) are from "
                f"{', '.join(mismatch_years)} documentation — no exact match found for your "
                "vehicle year. The information may still apply; verify with a Toyota technician."
            )
        refs = report.get("source_refs", [])
        sources = refs if refs else [
            f"[{c['source']}]({c['source_url']})" for c in chunks if c.get("source_url")
        ]
        for s in sources:
            st.markdown(f"- {s}")

    with st.expander("🛠️ Raw JSON Report"):
        st.json({k: v for k, v in report.items() if k != "raw_text"})

    # ── Reference Diagrams (LEMON images) ─────────────────────────────────── #
    _render_reference_diagrams(chunks)

    st.caption(
        "⚠️ **Disclaimer:** For educational and diagnostic assistance only. "
        "Always have a certified Toyota technician verify any diagnosis before performing repairs."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(
        page_title="Toyota AI Diagnostic Assistant",
        page_icon="🚗",
        layout="wide",
    )
    st.title("🚗 Toyota AI Diagnostic Assistant")

    vehicle_model, vehicle_year, engine_code = vehicle_sidebar()

    tab1, tab2 = st.tabs(["💬 Describe Symptoms", "📊 Enter Sensor Data"])

    with tab1:
        symptom_tab(vehicle_model, vehicle_year, engine_code)

    with tab2:
        sensor_tab(vehicle_model, vehicle_year, engine_code)


if __name__ == "__main__":
    main()
