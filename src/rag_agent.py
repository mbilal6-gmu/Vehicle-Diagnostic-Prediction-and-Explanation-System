"""
rag_agent.py
============
Hybrid retrieval agent: BM25 keyword search + ChromaDB vector search.
Results are re-ranked by combined score and returned with full traceability metadata.

Usage (programmatic):
    from src.rag_agent import retrieve
    chunks = retrieve(dtc_code="P0300", engine_code="2GR-FE", vehicle_model="camry", k=5)
"""

import os
import re
from typing import Optional
import torch
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VS_DIR          = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "chroma_db")
COLLECTION_NAME = "toyota_diagnostics"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K_VECTOR    = 30   # fetch more, then re-rank
TOP_K_BM25      = 30
TOP_K_FINAL     = 8    # return top 8 to LLM

# Code pattern detection
_DTC_PATTERN = re.compile(r'^[PBCU][0-9]{4}$', re.IGNORECASE)
_TSB_PATTERN = re.compile(r'^T-SB-\d{3,4}-\d{2,4}(?:\s*REV\s*\d+)?$', re.IGNORECASE)


def detect_code_type(text: str) -> tuple[str, str]:
    """
    Returns (normalized_code, code_type) where code_type is:
      'dtc'  — OBD-II / manufacturer DTC (e.g. P0171, B1234)
      'tsb'  — Toyota TSB reference (e.g. T-SB-0009-23)
      'symptom' — free-text symptom description
    """
    t = text.strip().upper()
    if _DTC_PATTERN.match(t):
        return t, "dtc"
    if _TSB_PATTERN.match(t):
        return t, "tsb"
    return text.strip(), "symptom"


# --------------------------------------------------------------------------- #
# Singleton collection (loaded once per process)
# --------------------------------------------------------------------------- #
_collection = None
_bm25       = None
_bm25_docs  = None   # list of all document strings
_bm25_metas = None   # list of all metadata dicts


def _load_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=VS_DIR)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL,
            device=_DEVICE,
        )
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=emb_fn,
        )
    return _collection


def _load_bm25():
    """Build BM25 index over all documents in ChromaDB (done once)."""
    global _bm25, _bm25_docs, _bm25_metas
    if _bm25 is None:
        col   = _load_collection()
        total = col.count()
        # Fetch all documents in one shot (fine for ~13k docs)
        result      = col.get(limit=total, include=["documents", "metadatas"])
        _bm25_docs  = result["documents"]
        _bm25_metas = result["metadatas"]
        tokenized   = [doc.lower().split() for doc in _bm25_docs]
        _bm25       = BM25Okapi(tokenized)
    return _bm25, _bm25_docs, _bm25_metas


def _dtc_description(dtc_code: str) -> str:
    """Look up a short description for a DTC — prefer toyota-club.net over LEMON page titles."""
    try:
        col = _load_collection()
        # First try: RAG CSV source has concise descriptions ("Random Misfire Detected")
        res = col.get(
            where={"$and": [
                {"dtc_code": {"$eq": dtc_code.upper()}},
                {"source":   {"$eq": "toyota-club.net"}},
            ]},
            limit=1,
            include=["metadatas"],
        )
        if res["metadatas"]:
            return res["metadatas"][0].get("description", "")
        # Fallback: any source
        res2 = col.get(
            where={"dtc_code": {"$eq": dtc_code.upper()}},
            limit=1,
            include=["metadatas"],
        )
        if res2["metadatas"]:
            return res2["metadatas"][0].get("description", "")
    except Exception:
        pass
    return ""


def _build_query(dtc_code: str, engine_code: str, vehicle_model: str) -> str:
    """Construct a rich query string for embedding and BM25."""
    parts = []
    if dtc_code:
        parts.append(dtc_code.upper())
        desc = _dtc_description(dtc_code)
        if desc:
            parts.append(desc)
    if vehicle_model: parts.append(vehicle_model.lower())
    if engine_code:  parts.append(engine_code)
    return " ".join(parts)


def retrieve(
    dtc_code: str        = "",
    engine_code: str     = "",
    vehicle_model: str   = "",
    vehicle_year: int    = 0,
    free_text: str       = "",
    k: int               = TOP_K_FINAL,
) -> list[dict]:
    """
    Hybrid BM25 + vector retrieval with metadata-aware re-ranking.

    Handles three query types automatically:
      - DTC code (P0171)  → boosts exact dtc_code + all_dtcs metadata matches
      - TSB reference     → boosts chunks containing that TSB ref in text/tsb_refs
      - Symptom text      → pure semantic + BM25

    Returns a list of dicts:
        {document, dtc_code, all_dtcs, tsb_refs, model_year, engine_code,
         description, source, source_url, year_mismatch, combined_score}
    """
    col       = _load_collection()
    bm25, all_docs, all_metas = _load_bm25()

    # Detect whether the user entered a code or symptom text
    code_val, code_type = detect_code_type(dtc_code) if dtc_code else ("", "symptom")

    query = _build_query(dtc_code, engine_code, vehicle_model)
    if free_text:
        query = f"{query} {free_text}".strip()

    # ---- Vector search ---------------------------------------------------- #
    where_filter = None
    if engine_code and engine_code not in ("", "Unknown"):
        # Include exact match OR engine_code="" (LEMON chunks have no engine code)
        where_filter = {"$or": [
            {"engine_code": {"$eq": engine_code}},
            {"engine_code": {"$eq": ""}},
        ]}

    vec_result = col.query(
        query_texts    = [query],
        n_results      = TOP_K_VECTOR,
        where          = where_filter,
        include        = ["documents", "metadatas", "distances"],
    )
    vec_docs   = vec_result["documents"][0]
    vec_metas  = vec_result["metadatas"][0]
    vec_dists  = vec_result["distances"][0]   # cosine distance (lower = better)

    # Normalise vector scores to [0, 1] (1 = most similar)
    max_dist = max(vec_dists) if vec_dists else 1.0
    vec_scores = {
        doc: 1.0 - (dist / (max_dist + 1e-9))
        for doc, dist in zip(vec_docs, vec_dists)
    }

    # ---- BM25 search ------------------------------------------------------ #
    tokens     = query.lower().split()
    bm25_raw   = bm25.get_scores(tokens)
    top_bm25_idx = sorted(range(len(bm25_raw)), key=lambda i: bm25_raw[i], reverse=True)[:TOP_K_BM25]
    max_bm25   = bm25_raw[top_bm25_idx[0]] if top_bm25_idx else 1.0

    bm25_scores = {
        all_docs[i]: bm25_raw[i] / (max_bm25 + 1e-9)
        for i in top_bm25_idx
    }

    # ---- LEMON supplement: force repair manual chunks into candidate pool ---- #
    # For DTC queries the main vector search is biased toward CSV one-liners that
    # score high on exact-DTC match.  Two approaches:
    #   1. Direct metadata get() for LEMON chunks where dtc_code == code_val
    #      (finds procedure pages where P0300 is the primary DTC)
    #   2. Semantic query restricted to LEMON as a fallback for supporting content
    lemon_supplement: list[tuple[str, dict, float]] = []  # (doc, meta, vec_score)
    if code_type == "dtc" and code_val:
        # Approach 1: direct metadata match — highest precision
        # Fetch more than needed, then sort so actual Procedure pages come first
        # (DTC chart pages have generic table content; Procedure pages have repair steps)
        lemon_direct = col.get(
            where={"$and": [
                {"source":   {"$eq": "LEMON Vehicle Manual Database"}},
                {"dtc_code": {"$eq": code_val}},
            ]},
            limit=40,
            include=["documents", "metadatas"],
        )
        pairs = list(zip(lemon_direct["documents"], lemon_direct["metadatas"]))
        # Procedure pages have actionable repair content; push them to front
        proc_pairs = [(d, m) for d, m in pairs if m.get("description", "") == "Procedure"]
        other_pairs = [(d, m) for d, m in pairs if m.get("description", "") != "Procedure"]
        for doc, meta in (proc_pairs + other_pairs)[:8]:
            lemon_supplement.append((doc, meta, 0.80))  # fixed high score for exact match

        # Approach 2: semantic query on LEMON for any additional supporting content
        lemon_res = col.query(
            query_texts=[query],
            n_results=8,
            where={"source": {"$eq": "LEMON Vehicle Manual Database"}},
            include=["documents", "metadatas", "distances"],
        )
        l_dists = lemon_res["distances"][0]
        max_ld  = max(l_dists) if l_dists else 1.0
        direct_docs = {doc for doc, _, _ in lemon_supplement}
        for doc, meta, dist in zip(lemon_res["documents"][0], lemon_res["metadatas"][0], l_dists):
            if doc not in direct_docs:
                lemon_supplement.append((doc, meta, 1.0 - (dist / (max_ld + 1e-9))))

    # ---- Combine & re-rank ------------------------------------------------ #
    lemon_vec_scores = {doc: vs for doc, _, vs in lemon_supplement}
    candidate_docs = set(vec_docs) | set(all_docs[i] for i in top_bm25_idx) | set(lemon_vec_scores)

    # Build a lookup: doc text → metadata (from vector result first)
    meta_lookup = {d: m for d, m in zip(vec_docs, vec_metas)}
    for i in top_bm25_idx:
        if all_docs[i] not in meta_lookup:
            meta_lookup[all_docs[i]] = all_metas[i]
    for doc, meta, _ in lemon_supplement:
        if doc not in meta_lookup:
            meta_lookup[doc] = meta

    ranked = []
    for doc in candidate_docs:
        vs  = vec_scores.get(doc, lemon_vec_scores.get(doc, 0.0))
        bs  = bm25_scores.get(doc, 0.0)
        combined = 0.55 * vs + 0.45 * bs
        meta = meta_lookup.get(doc, {})

        # --- Code-type specific boosting -----------------------------------
        if code_type == "dtc" and code_val:
            chunk_dtc = meta.get("dtc_code", "").upper()
            chunk_all = [d.strip() for d in meta.get("all_dtcs", "").upper().split(",")]
            # Boost if primary dtc_code matches
            if chunk_dtc == code_val:
                combined = min(1.0, combined + 0.25)
            # Boost if code appears in all_dtcs list
            elif code_val in chunk_all:
                combined = min(1.0, combined + 0.15)
            # Penalise LEMON DTC chart pages whose primary code is a different DTC
            # (e.g. a P2135 chart page that happens to mention P0300 in passing)
            elif (chunk_dtc and chunk_dtc != code_val
                  and meta.get("source", "") == "LEMON Vehicle Manual Database"
                  and "chart" in meta.get("description", "").lower()):
                combined = max(0.0, combined - 0.20)

        elif code_type == "tsb" and code_val:
            # Boost if TSB ref appears in tsb_refs metadata or document text
            tsb_in_meta = code_val in [t.strip() for t in meta.get("tsb_refs", "").upper().split(",")]
            tsb_in_doc  = code_val.upper() in doc.upper()
            if tsb_in_meta or tsb_in_doc:
                combined = min(1.0, combined + 0.30)

        # --- Cross-year proximity boost ------------------------------------
        # Slightly prefer same-year results; don't penalise adjacent years
        chunk_year_str = meta.get("model_year", "")
        chunk_year = int(chunk_year_str) if chunk_year_str.isdigit() else 0
        if vehicle_year and chunk_year:
            year_diff = abs(vehicle_year - chunk_year)
            if year_diff == 0:
                combined = min(1.0, combined + 0.05)
            elif year_diff <= 2:
                combined = min(1.0, combined + 0.02)

        # --- Year mismatch flag --------------------------------------------
        year_mismatch = (
            bool(vehicle_year and chunk_year and vehicle_year != chunk_year
                 and meta.get("source", "") == "LEMON Vehicle Manual Database")
        )

        ranked.append({
            "document":     doc,
            "dtc_code":     meta.get("dtc_code",    ""),
            "all_dtcs":     meta.get("all_dtcs",     ""),
            "tsb_refs":     meta.get("tsb_refs",     ""),
            "model_year":   chunk_year or None,
            "engine_code":  meta.get("engine_code",  ""),
            "description":  meta.get("description",  ""),
            "source":       meta.get("source",        ""),
            "source_url":   meta.get("source_url",    ""),
            "image_keys":   meta.get("image_keys",    ""),
            "year_mismatch": year_mismatch,
            "combined_score": round(combined, 4),
        })

    ranked.sort(key=lambda x: x["combined_score"], reverse=True)

    # Diversity selection:
    #   - At most 3 chunks per source (prevents CSV one-liners flooding results)
    #   - At most 2 chunks per (source, description) pair (prevents identical
    #     "Typical Enabling Conditions" from 5 different vehicles)
    MAX_PER_SOURCE = 3
    MAX_PER_DESC   = 2
    source_counts: dict[str, int] = {}
    desc_counts:   dict[str, int] = {}
    diverse: list[dict] = []
    remaining: list[dict] = []
    for chunk in ranked:
        src  = chunk.get("source", "")
        desc = f"{src}||{chunk.get('description', '')}"
        if (source_counts.get(src, 0) < MAX_PER_SOURCE and
                desc_counts.get(desc, 0) < MAX_PER_DESC):
            diverse.append(chunk)
            source_counts[src]  = source_counts.get(src, 0) + 1
            desc_counts[desc]   = desc_counts.get(desc, 0) + 1
        else:
            remaining.append(chunk)
        if len(diverse) >= k:
            break
    if len(diverse) < k:
        for chunk in remaining:
            diverse.append(chunk)
            if len(diverse) >= k:
                break

    return diverse[:k]


def format_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks as a numbered context block for the LLM."""
    lines = []
    for i, c in enumerate(chunks, 1):
        year_note = ""
        if c.get("year_mismatch") and c.get("model_year"):
            year_note = f" [NOTE: from {c['model_year']} documentation — may still apply]"
        lines.append(
            f"[{i}] Source: {c['source']}{year_note}\n"
            f"    Code: {c['dtc_code']} | Engine: {c['engine_code']}\n"
            f"    {c['document']}\n"
        )
    return "\n".join(lines)
