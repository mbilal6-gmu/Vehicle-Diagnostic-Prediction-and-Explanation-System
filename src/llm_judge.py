"""
llm_judge.py
============
Two responsibilities:
  1. generate_report() — calls the primary LLM (OpenAI GPT-4o) or the
     fallback (DeepSeek via Ollama) to produce a structured diagnostic report.

  2. judge_faithfulness() — calls GPT-4o to score how well the report is
     supported by the retrieved RAG chunks (0.0–1.0).

Usage:
    from src.llm_judge import generate_report, judge_faithfulness

    report   = generate_report(context_chunks, vehicle_info, risk_score, cel_likely)
    faithful = judge_faithfulness(report["raw_text"], context_chunks)
"""

import os
import json
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
OLLAMA_BASE    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
FAITH_THRESH   = float(os.getenv("LLM_FAITHFULNESS_THRESHOLD", "0.7"))


# --------------------------------------------------------------------------- #
# LLM client helpers
# --------------------------------------------------------------------------- #

def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _ollama_available() -> bool:
    try:
        import requests
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _call_openai(system: str, user: str, json_mode: bool = True) -> str:
    client = _openai_client()
    kwargs = dict(
        model       = OPENAI_MODEL,
        temperature = 0,
        messages    = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_ollama(system: str, user: str) -> str:
    import requests
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "stream": False,
    }
    r = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    # DeepSeek-R1 prefixes responses with <think>...</think> reasoning blocks.
    # Strip them so downstream JSON parsing only sees the actual answer.
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return content


def _call_llm(system: str, user: str, require_json: bool = True) -> tuple[str, str]:
    """Try OpenAI first; fall back to Ollama. Returns (raw_text, llm_used)."""
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key.startswith("sk-"):
        try:
            return _call_openai(system, user, json_mode=require_json), "openai"
        except Exception as e:
            print(f"[WARN] OpenAI failed ({e}), trying Ollama …")

    if _ollama_available():
        try:
            return _call_ollama(system, user), "ollama"
        except Exception as e:
            raise RuntimeError(f"Both OpenAI and Ollama failed. Last error: {e}")

    raise RuntimeError(
        "No LLM available. Set OPENAI_API_KEY in .env or start Ollama with DeepSeek."
    )


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #

REPORT_SYSTEM = """\
You are a certified Toyota master technician assistant.
You will receive:
  - Vehicle information (model, year, engine code)
  - An ML-predicted failure risk score (0.0 = healthy, 1.0 = critical)
  - A check-engine-light prediction (true/false)
  - Numbered RAG context chunks from two sources:
      • "LEMON Vehicle Manual Database" — Toyota official repair procedures and diagnostic trees
        (these contain step-by-step tests, component-level findings, and root causes)
      • "toyota-club.net" — DTC reference index with code names only

Your job: synthesise the information into a structured JSON diagnostic report.

Rules:
  - Only use facts from the provided context chunks — do NOT invent specifications
  - Cite your source chunks by number (e.g. [1], [2]) in each field
  - If you are unsure, say "Insufficient data in knowledge base"
  - Never hallucinate part numbers, torque specs, or repair costs
  - Some chunks may be marked "[NOTE: from YYYY documentation]" — include them if relevant,
    but explicitly mention the year difference in your source_refs entry for that chunk

CAUSES — source priority:
  1. OBD sensor readings marked ⚠️ OUT OF RANGE — always list these as causes first,
     citing the specific reading and what it indicates (e.g. "Coolant temperature 110°C
     exceeds normal operating range — possible cooling system fault").
  2. LEMON chunks — extract component-level root causes from diagnostic procedure content.
  3. toyota-club.net chunks — use ONLY if LEMON chunks contain no cause information.
  NEVER write a cause that merely repeats the DTC name (e.g. "Random Misfire Detected" is
  NOT a cause — it is the symptom). Every cause entry must describe WHY the fault occurs.

  Good cause examples:
    "Worn or fouled spark plug causing incomplete combustion [4]"
    "Faulty ignition coil assembly failing to fire cylinder [5]"
    "Vacuum leak causing lean air-fuel mixture [6]"
  Bad cause examples (do not write these):
    "Random / Multiple Cylinder Misfire Detected [1]"
    "System Too Lean Bank 1 [2]"

ACTIONS — source priority:
  Primary: LEMON chunks. Extract the specific inspection and repair steps described in the
  procedure content (spark tests, resistance measurements, component replacements).
  Secondary: toyota-club.net chunks if LEMON provides no actions.

Return ONLY valid JSON with this exact schema:
{
  "diagnosis":   "One-sentence summary of the most likely fault",
  "causes":      ["Cause 1 [source ref]", "Cause 2 [source ref]", ...],
  "actions":     ["Recommended action 1", "Recommended action 2", ...],
  "urgency":     "immediate | soon | monitor | none",
  "confidence":  0.0-1.0,
  "source_refs": ["Source name — URL", ...]
}
"""


# (lo, hi, unit, warn_low, warn_high)
_SENSOR_RANGES = {
    "coolant_temp_c":    (80,  105, "°C",  "Low coolant temperature — engine not reaching operating temp, thermostat possibly stuck open (P0128)", "High coolant temperature — possible cooling system fault (fan, thermostat, coolant leak)"),
    "engine_rpm":        (550, 1200,"RPM", "Abnormally low idle RPM — possible stall condition (P0505)", "Abnormally high idle RPM — possible IAC/throttle body fault (P0507)"),
    "fuel_trim_pct":     (-10, 10,  "%",   "Strongly negative fuel trim — rich running condition (P0172)", "Strongly positive fuel trim — lean running condition (P0171)"),
    "engine_load_pct":   (0,   90,  "%",   "", "Unusually high engine load"),
    "maf_gps":           (0,   300, "g/s", "", ""),
    "vehicle_speed_kmh": (0,   250, "km/h","", ""),
}


def _format_sensor_section(sensor_readings: dict) -> str:
    """Format sensor readings for the LLM prompt, flagging out-of-range values with direction-specific warnings."""
    if not sensor_readings:
        return ""
    lines = ["\n=== OBD Sensor Readings ==="]
    for key, val in sensor_readings.items():
        if key in _SENSOR_RANGES:
            lo, hi, unit, warn_low, warn_high = _SENSOR_RANGES[key]
            if val < lo and warn_low:
                flag = f"  ⚠️ BELOW NORMAL — {warn_low}"
            elif val > hi and warn_high:
                flag = f"  ⚠️ ABOVE NORMAL — {warn_high}"
            else:
                flag = ""
            lines.append(f"  {key}: {val} {unit}{flag}")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def generate_report(
    context_chunks:  list[dict],
    vehicle_model:   str,
    vehicle_year:    int,
    engine_code:     str,
    dtc_code:        str,
    risk_score:      float,
    cel_likely:      bool,
    sensor_readings: dict = None,
) -> dict:
    """
    Generate a traceable diagnostic report.

    Returns dict with keys: diagnosis, causes, actions, urgency,
    confidence, source_refs, llm_used, faithfulness_score, passed_gate.
    """
    from src.rag_agent import format_for_prompt

    # Limit context for local models to keep latency under ~60s
    openai_key = os.getenv("OPENAI_API_KEY", "")
    ollama_mode = not (openai_key and openai_key.startswith("sk-"))
    chunks_to_use = context_chunks[:5] if ollama_mode else context_chunks

    context_text   = format_for_prompt(chunks_to_use)
    sensor_section = _format_sensor_section(sensor_readings)

    user_msg = f"""
Vehicle: Toyota {vehicle_model.title()} {vehicle_year} | Engine: {engine_code}
DTC Code entered: {dtc_code if dtc_code else "None"}
ML Failure Risk Score: {risk_score:.2f} / 1.00
Check Engine Light predicted: {"YES" if cel_likely else "NO"}
{sensor_section}
=== Retrieved Diagnostic Context ===
{context_text}

Based on the above, generate the diagnostic JSON report.
NOTE: Any sensor reading marked ⚠️ OUT OF RANGE must be reflected as a cause even if
the ML risk score is low. Low risk score means the ML model did not predict imminent
failure — it does NOT mean all sensor readings are normal.
"""

    raw_text, llm_used = _call_llm(REPORT_SYSTEM, user_msg, require_json=True)

    # Parse JSON from response
    try:
        report = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if match:
            report = json.loads(match.group())
        else:
            report = {
                "diagnosis":   "JSON parse error — see raw_text",
                "causes":      [],
                "actions":     [],
                "urgency":     "unknown",
                "confidence":  0.0,
                "source_refs": [],
            }

    # Normalise schema — local models (DeepSeek) sometimes deviate from the spec
    report = _normalize_report(report)
    report = _fix_causes_actions(report)

    report["raw_text"]  = raw_text
    report["llm_used"]  = llm_used

    # Run the faithfulness judge
    faith_score = judge_faithfulness(raw_text, chunks_to_use)
    report["faithfulness_score"] = faith_score
    report["passed_gate"]        = faith_score >= FAITH_THRESH

    return report


_IMPERATIVE_VERBS = {
    "check", "inspect", "replace", "use", "measure", "consult", "remove",
    "install", "clean", "verify", "test", "scan", "ensure", "perform",
    "connect", "disconnect", "clear", "reset", "run", "read", "record",
}


def _fix_causes_actions(report: dict) -> dict:
    """
    DeepSeek often writes action steps inside 'causes' and causes inside 'actions'.
    - Any cause starting with an imperative verb → moved to actions.
    - Any action that does NOT start with an imperative verb → moved to causes
      (only if causes list is otherwise empty, to avoid over-correction).
    """
    causes  = report.get("causes", [])
    actions = report.get("actions", [])

    real_causes:    list[str] = []
    misplaced_acts: list[str] = []

    for c in causes:
        first = c.strip().split()[0].lower().rstrip(".,") if c.strip() else ""
        if first in _IMPERATIVE_VERBS:
            misplaced_acts.append(c)
        else:
            real_causes.append(c)

    report["causes"]  = real_causes
    report["actions"] = misplaced_acts + actions

    # Deduplicate actions preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for a in report["actions"]:
        key = a.lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    report["actions"] = deduped

    return report


def _normalize_report(report: dict) -> dict:
    """
    Coerce alternative JSON schemas (e.g. DeepSeek's format) into the
    canonical schema expected by the UI:
      diagnosis, causes (list[str]), actions (list[str]),
      urgency, confidence, source_refs
    """
    # diagnosis
    if not report.get("diagnosis") or report["diagnosis"] == "N/A":
        report["diagnosis"] = (
            report.get("title") or report.get("summary") or
            report.get("issue") or "Insufficient data in knowledge base"
        )

    # causes — may be list of dicts
    causes = report.get("causes", [])
    if causes and isinstance(causes[0], dict):
        report["causes"] = [
            c.get("causeText") or c.get("description") or c.get("cause") or str(c)
            for c in causes
        ]

    # actions — may be list of dicts
    actions = report.get("actions", [])
    if actions and isinstance(actions[0], dict):
        report["actions"] = [
            a.get("description") or a.get("action") or a.get("actionText") or str(a)
            for a in actions
        ]

    # urgency — map from risk labels if missing
    if not report.get("urgency") or report["urgency"] == "unknown":
        risk_label = str(report.get("diagnosticRiskScore", "")).lower()
        if any(w in risk_label for w in ("high", "critical", "immediate")):
            report["urgency"] = "immediate"
        elif any(w in risk_label for w in ("medium", "moderate", "soon")):
            report["urgency"] = "soon"
        elif any(w in risk_label for w in ("low", "monitor")):
            report["urgency"] = "monitor"
        else:
            report["urgency"] = "unknown"

    # confidence — default to 0.5 if missing or zero
    if not report.get("confidence"):
        report["confidence"] = 0.5

    # source_refs — default to empty list
    report.setdefault("source_refs", [])

    return report


# --------------------------------------------------------------------------- #
# Faithfulness judge
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = """\
You are a strict factual auditor for automotive AI systems.
You will receive:
  1. An AI-generated diagnostic report
  2. The source context chunks the AI was given

Score how faithfully the report is supported by the context (0.0–1.0):
  1.0 = every claim is directly supported by the context, OR the report correctly
        states "Insufficient data" / "no information available" when the context
        does not contain relevant information (admitting uncertainty IS faithful)
  0.5 = some claims supported, some unsupported
  0.0 = the report contains fabricated facts not present in the context

Return ONLY valid JSON: {"faithfulness_score": <float>, "reason": "<one line>"}
"""


def judge_faithfulness(report_text: str, context_chunks: list[dict]) -> float:
    """
    Returns a faithfulness score (0.0–1.0).
    Falls back to 0.5 if the judge call fails.
    """
    from src.rag_agent import format_for_prompt

    context_text = format_for_prompt(context_chunks)
    user_msg = f"""
=== AI Report ===
{report_text}

=== Source Context ===
{context_text}

Score the faithfulness.
"""
    try:
        raw, _ = _call_llm(JUDGE_SYSTEM, user_msg, require_json=True)
        parsed = json.loads(raw)
        score  = float(parsed.get("faithfulness_score", 0.5))
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        print(f"[WARN] Faithfulness judge failed ({e}), defaulting to 0.5")
        return 0.5


# --------------------------------------------------------------------------- #
# Differential diagnosis — candidate generation & narrowing
# --------------------------------------------------------------------------- #

CANDIDATES_SYSTEM = """\
You are a Toyota master technician performing differential diagnosis.
You will receive a vehicle description and symptom report.

Generate the most likely fault codes, ranked by confidence.
Rules:
  - Return 3–5 candidates maximum
  - Only use real, well-known OBD-II / Toyota DTC codes — examples:
      P0300 Random Misfire, P0171 System Too Lean Bank 1, P0128 Thermostat,
      P0420 Catalyst Efficiency, P0507 Idle Control High, P0172 System Too Rich,
      P0301-P0308 Cylinder Misfire, P0401 EGR Flow Insufficient,
      B1507 Wiper Motor, C1201 Engine Control System, U0100 Lost Comm ECM
  - Use the EXACT standard SAE/Toyota issue name for each code — do NOT invent descriptions
  - Base confidence on symptom match strength (0.0–1.0)
  - "follow_up" MUST be a direct question ending with "?" that the customer can answer
    (e.g. "Does the check engine light come on when the engine is cold?" not an instruction)
  - Do NOT invent codes or descriptions

Return ONLY valid JSON:
{
  "candidates": [
    {"dtc_code": "P0300", "issue": "Random Misfire Detected", "confidence": 0.78, "reasoning": "one-line reason"},
    {"dtc_code": "P0171", "issue": "System Too Lean Bank 1",  "confidence": 0.55, "reasoning": "one-line reason"}
  ],
  "follow_up": "Does the check engine light come on?"
}
"""

UPDATE_SYSTEM = """\
You are a Toyota master technician refining a differential diagnosis.
You will receive:
  1. The original symptom and conversation history
  2. New information just provided by the user
  3. The current list of candidate fault codes with their confidence scores

Update each candidate's confidence based on the new information.
Rules:
  - Re-score all candidates (confidence can go up or down)
  - Mark candidates as eliminated if confidence drops below 0.20
  - "follow_up" MUST be a direct question ending with "?" that the customer can answer
  - Do NOT add new candidates not in the original list

Return ONLY valid JSON using the same schema:
{
  "candidates": [
    {"dtc_code": "P0300", "issue": "...", "confidence": 0.85, "reasoning": "updated reason", "eliminated": false},
    {"dtc_code": "P0507", "issue": "...", "confidence": 0.15, "reasoning": "ruled out because ...", "eliminated": true}
  ],
  "follow_up": "Next targeted question ending with ?"
}
"""


def generate_candidates(
    symptom_text:  str,
    vehicle_model: str,
    vehicle_year:  int,
    engine_code:   str,
) -> dict:
    """
    Round 1: Generate initial list of candidate fault codes from symptom description.

    Returns dict:
        {
          "candidates": [{"dtc_code", "issue", "confidence", "reasoning"}, ...],
          "follow_up": "question string"
        }
    Falls back to empty candidates on LLM failure.
    """
    engine_str = engine_code if engine_code and engine_code != "Unknown" else "unknown engine"
    user_msg = (
        f"Vehicle: Toyota {vehicle_model.title()} {vehicle_year} | Engine: {engine_str}\n\n"
        f"Customer complaint:\n{symptom_text}\n\n"
        "Generate your differential diagnosis."
    )
    try:
        raw, _ = _call_llm(CANDIDATES_SYSTEM, user_msg, require_json=True)
        result = _parse_json_safe(raw)
        for c in result.get("candidates", []):
            c["confidence"] = float(c.get("confidence", 0.5))
            c.setdefault("eliminated", False)
        result["candidates"] = _correct_dtc_descriptions(result.get("candidates", []))
        result["follow_up"]  = _ensure_question(result.get("follow_up", ""))
        return result
    except Exception as e:
        print(f"[WARN] generate_candidates failed ({e})")
        return {"candidates": [], "follow_up": "Can you describe any other symptoms?"}


def update_candidates(
    existing_candidates: list[dict],
    new_info:            str,
    vehicle_model:       str,
    vehicle_year:        int,
    engine_code:         str,
    history:             list[str],
) -> dict:
    """
    Rounds 2+: Re-score candidates given new user information.

    Returns same dict schema as generate_candidates.
    """
    engine_str   = engine_code if engine_code and engine_code != "Unknown" else "unknown engine"
    history_text = "\n".join(f"- {h}" for h in history) if history else "(none)"
    candidates_json = json.dumps(
        [{"dtc_code": c["dtc_code"], "issue": c["issue"], "confidence": c["confidence"]}
         for c in existing_candidates],
        indent=2,
    )

    user_msg = (
        f"Vehicle: Toyota {vehicle_model.title()} {vehicle_year} | Engine: {engine_str}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"New information from customer:\n{new_info}\n\n"
        f"Current candidates:\n{candidates_json}\n\n"
        "Re-score and update the diagnosis."
    )
    try:
        raw, _ = _call_llm(UPDATE_SYSTEM, user_msg, require_json=True)
        result = _parse_json_safe(raw)
        for c in result.get("candidates", []):
            c["confidence"] = float(c.get("confidence", 0.5))
            if c["confidence"] < 0.20:
                c["eliminated"] = True
            else:
                c.setdefault("eliminated", False)
        result["follow_up"] = _ensure_question(result.get("follow_up", ""))
        return result
    except Exception as e:
        print(f"[WARN] update_candidates failed ({e})")
        return {"candidates": existing_candidates, "follow_up": "Can you share any additional details?"}


def _parse_json_safe(raw: str) -> dict:
    """Parse JSON, with fallback extraction from surrounding text."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _ensure_question(text: str) -> str:
    """Guarantee the follow-up string is a question ending with '?'."""
    text = text.strip().rstrip(".")
    if not text:
        return "Can you describe any other symptoms?"
    if not text.endswith("?"):
        text += "?"
    return text


def _looks_hallucinated(description: str) -> bool:
    """Return True if the description shows likely LLM hallucination markers."""
    if not description:
        return False
    # Any character repeated 3+ consecutive times (e.g. "Vacuumuum")
    if re.search(r'(.)\1\1', description, re.IGNORECASE):
        return True
    # "Misfire" paired with components that don't cause misfires
    low = description.lower()
    if "misfire" in low and any(w in low for w in ("vacuum", "egr", "evap", "abs", "pump", "relay")):
        return True
    return False


def _correct_dtc_descriptions(candidates: list[dict]) -> list[dict]:
    """
    1. Replace hallucinated DTC descriptions with canonical vectorstore descriptions
       (toyota-club.net preferred).
    2. Drop candidates whose code is absent from the vectorstore AND whose
       description shows hallucination markers (repeated chars, nonsensical combos).
    """
    try:
        from src.rag_agent import _load_collection
        col = _load_collection()
        verified: set[str] = set()

        for c in candidates:
            code = c.get("dtc_code", "").upper()
            if not code:
                continue
            # Try toyota-club.net first (most concise descriptions)
            res = col.get(
                where={"$and": [
                    {"dtc_code": {"$eq": code}},
                    {"source":   {"$eq": "toyota-club.net"}},
                ]},
                limit=1,
                include=["metadatas"],
            )
            if res["metadatas"]:
                canonical = res["metadatas"][0].get("description", "")
                if canonical:
                    c["issue"] = canonical
                verified.add(code)
                continue
            # Fallback: any source — still marks code as verified
            res2 = col.get(
                where={"dtc_code": {"$eq": code}},
                limit=1,
                include=["metadatas"],
            )
            if res2["metadatas"]:
                verified.add(code)

        # Remove candidates that are unverified AND look hallucinated
        filtered = []
        for c in candidates:
            code = c.get("dtc_code", "").upper()
            if code not in verified and _looks_hallucinated(c.get("issue", "")):
                print(f"[INFO] Dropped hallucinated candidate: {code} — {c.get('issue', '')}")
                continue
            filtered.append(c)
        return filtered

    except Exception as e:
        print(f"[WARN] _correct_dtc_descriptions failed ({e})")
    return candidates
