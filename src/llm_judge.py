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
    return r.json()["message"]["content"]


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
  - Numbered RAG context chunks from Toyota diagnostic databases

Your job: synthesise the information into a structured JSON diagnostic report.
Rules:
  - Only use facts from the provided context chunks — do NOT invent specifications
  - Cite your source chunks by number (e.g. [1], [2]) in each field
  - If you are unsure, say "Insufficient data in knowledge base"
  - Never hallucinate part numbers, torque specs, or repair costs

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


def generate_report(
    context_chunks: list[dict],
    vehicle_model:  str,
    vehicle_year:   int,
    engine_code:    str,
    dtc_code:       str,
    risk_score:     float,
    cel_likely:     bool,
) -> dict:
    """
    Generate a traceable diagnostic report.

    Returns dict with keys: diagnosis, causes, actions, urgency,
    confidence, source_refs, llm_used, faithfulness_score, passed_gate.
    """
    from src.rag_agent import format_for_prompt

    context_text = format_for_prompt(context_chunks)

    user_msg = f"""
Vehicle: Toyota {vehicle_model.title()} {vehicle_year} | Engine: {engine_code}
DTC Code entered: {dtc_code if dtc_code else "None"}
ML Failure Risk Score: {risk_score:.2f} / 1.00
Check Engine Light predicted: {"YES" if cel_likely else "NO"}

=== Retrieved Diagnostic Context ===
{context_text}

Based ONLY on the above context, generate the diagnostic JSON report.
"""

    raw_text, llm_used = _call_llm(REPORT_SYSTEM, user_msg, require_json=True)

    # Parse JSON from response
    try:
        report = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON block if model added surrounding text
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

    report["raw_text"]  = raw_text
    report["llm_used"]  = llm_used

    # Run the faithfulness judge
    faith_score = judge_faithfulness(raw_text, context_chunks)
    report["faithfulness_score"] = faith_score
    report["passed_gate"]        = faith_score >= FAITH_THRESH

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
  1.0 = every claim is directly supported by the context
  0.0 = the report contains fabricated facts not in the context

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
  - Only use real Toyota/OBD-II DTC codes (P, B, C, or U codes)
  - Base confidence on symptom match strength (0.0–1.0)
  - Generate one targeted follow-up question to narrow the diagnosis further
  - Do NOT invent codes — use real documented fault codes only

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
  - Generate a new follow-up question based on remaining candidates
  - Do NOT add new candidates not in the original list

Return ONLY valid JSON using the same schema:
{
  "candidates": [
    {"dtc_code": "P0300", "issue": "...", "confidence": 0.85, "reasoning": "updated reason", "eliminated": false},
    {"dtc_code": "P0507", "issue": "...", "confidence": 0.15, "reasoning": "ruled out because ...", "eliminated": true}
  ],
  "follow_up": "Next targeted question"
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
        # Normalise confidence to float
        for c in result.get("candidates", []):
            c["confidence"] = float(c.get("confidence", 0.5))
            c.setdefault("eliminated", False)
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
