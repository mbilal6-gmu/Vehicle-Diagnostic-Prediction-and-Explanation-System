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
from typing import Optional
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

VS_DIR          = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "chroma_db")
COLLECTION_NAME = "toyota_diagnostics"
EMBED_MODEL     = "all-MiniLM-L6-v2"
TOP_K_VECTOR    = 20   # fetch more, then re-rank
TOP_K_BM25      = 20
TOP_K_FINAL     = 5    # return top 5 to LLM


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
            model_name=EMBED_MODEL
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


def _build_query(dtc_code: str, engine_code: str, vehicle_model: str) -> str:
    """Construct a rich query string for embedding and BM25."""
    parts = []
    if dtc_code:     parts.append(dtc_code.upper())
    if vehicle_model: parts.append(vehicle_model.lower())
    if engine_code:  parts.append(engine_code)
    return " ".join(parts)


def retrieve(
    dtc_code: str        = "",
    engine_code: str     = "",
    vehicle_model: str   = "",
    free_text: str       = "",
    k: int               = TOP_K_FINAL,
) -> list[dict]:
    """
    Hybrid BM25 + vector retrieval with metadata-aware re-ranking.

    Returns a list of dicts:
        {document, dtc_code, engine_code, description, source, source_url,
         vector_rank, bm25_rank, combined_score}
    """
    col       = _load_collection()
    bm25, all_docs, all_metas = _load_bm25()

    query = _build_query(dtc_code, engine_code, vehicle_model)
    if free_text:
        query = f"{query} {free_text}".strip()

    # ---- Vector search ---------------------------------------------------- #
    where_filter = None
    if engine_code and engine_code not in ("", "Unknown"):
        where_filter = {"engine_code": {"$eq": engine_code}}

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

    # ---- Combine & re-rank ------------------------------------------------ #
    candidate_docs = set(vec_docs) | set(all_docs[i] for i in top_bm25_idx)

    # Build a lookup: doc text → metadata (from vector result first)
    meta_lookup = {d: m for d, m in zip(vec_docs, vec_metas)}
    for i in top_bm25_idx:
        if all_docs[i] not in meta_lookup:
            meta_lookup[all_docs[i]] = all_metas[i]

    ranked = []
    for doc in candidate_docs:
        vs  = vec_scores.get(doc, 0.0)
        bs  = bm25_scores.get(doc, 0.0)
        combined = 0.55 * vs + 0.45 * bs   # favour vector slightly
        meta = meta_lookup.get(doc, {})

        # Boost if DTC code matches exactly
        if dtc_code and meta.get("dtc_code", "").upper() == dtc_code.upper():
            combined = min(1.0, combined + 0.2)

        ranked.append({
            "document":    doc,
            "dtc_code":    meta.get("dtc_code", ""),
            "engine_code": meta.get("engine_code", ""),
            "description": meta.get("description", ""),
            "source":      meta.get("source", ""),
            "source_url":  meta.get("source_url", ""),
            "image_keys":  meta.get("image_keys", ""),
            "combined_score": round(combined, 4),
        })

    ranked.sort(key=lambda x: x["combined_score"], reverse=True)
    return ranked[:k]


def format_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks as a numbered context block for the LLM."""
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] Source: {c['source']}\n"
            f"    Code: {c['dtc_code']} | Engine: {c['engine_code']}\n"
            f"    {c['document']}\n"
        )
    return "\n".join(lines)
