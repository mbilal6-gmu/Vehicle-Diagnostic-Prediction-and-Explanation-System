"""
build_vectorstore.py
====================
Merges all DTC knowledge sources and embeds them into a persistent
ChromaDB vector store for hybrid (BM25 + vector) retrieval.

Sources merged (in priority order):
  1. Toyota_RAG_Data.csv  — 12,978 Toyota DTC records (toyota-club.net)
  2. libre toyota.json    — 42 Toyota P1xxx manufacturer-specific codes
  3. libre dtc_db.json    — 39 standard OBD-II P0001-P0038 codes
  4. Data/nhtsa/nhtsa_complaints.csv — NHTSA complaint summaries (if present)
  5. Data/lemon/toyota_pages.jsonl   — LEMON manual pages (if present)
                                       Run src/extract_lemon.py first to generate.

Run ONCE before starting the app:
    python src/build_vectorstore.py
"""

import os
import json
import csv
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

DATA_DIR      = os.path.join(os.path.dirname(__file__), "..", "Data")
LIBRE_DIR     = os.path.join(DATA_DIR, "libre_dtc")
NHTSA_CSV     = os.path.join(DATA_DIR, "nhtsa", "nhtsa_complaints.csv")
RAG_CSV       = os.path.join(DATA_DIR, "Toyota_RAG_Data.csv.xls")  # CSV despite .xls extension
LEMON_JSONL   = os.path.join(DATA_DIR, "lemon", "toyota_pages.jsonl")
VS_DIR        = os.path.join(os.path.dirname(__file__), "..", "vectorstore", "chroma_db")

COLLECTION_NAME = "toyota_diagnostics"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # fast, local, no API key needed
BATCH_SIZE      = 256


def load_rag_csv(path: str) -> list[dict]:
    """Load Toyota_RAG_Data.csv (despite .xls extension — it's CSV)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rag_text = r.get("rag_text", "").strip()
            if not rag_text:
                continue
            rows.append({
                "id":           f"rag_{r.get('dtc_code','')}__{r.get('engine_code','')}",
                "document":     rag_text,
                "dtc_code":     r.get("dtc_code", ""),
                "engine_code":  r.get("engine_code", ""),
                "description":  r.get("description", ""),
                "source":       "toyota-club.net",
                "source_url":   r.get("source_url", ""),
                "image_keys":   "",
            })
    print(f"  Loaded {len(rows)} rows from RAG CSV")
    return rows


def load_libre_toyota(path: str) -> list[dict]:
    """Load Libre Diagnostic toyota.json (P1xxx manufacturer codes)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dtcs = data.get("dtcs", {})
    rows = []
    for code, desc in dtcs.items():
        rag_text = f"Engine: Toyota (manufacturer-specific) | Code: {code} | Issue: {desc}"
        rows.append({
            "id":           f"libre_toyota_{code}",
            "document":     rag_text,
            "dtc_code":     code,
            "engine_code":  "Toyota (manufacturer-specific)",
            "description":  desc,
            "source":       "Libre Automotive Diagnostic",
            "source_url":   "https://github.com/Libre-Diagnosctic/libre-automotive-diagnostic",
            "image_keys":   "",
        })
    print(f"  Loaded {len(rows)} rows from libre toyota.json")
    return rows


def load_libre_obd2(path: str) -> list[dict]:
    """Load Libre Diagnostic dtc_db.json (standard OBD-II P0xxx codes)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for code, entry in data.items():
        desc = entry if isinstance(entry, str) else entry.get("description", str(entry))
        rag_text = f"Engine: Gasoline (common) | Code: {code} | Issue: {desc}"
        rows.append({
            "id":           f"obd2_{code}",
            "document":     rag_text,
            "dtc_code":     code,
            "engine_code":  "Gasoline (common)",
            "description":  desc,
            "source":       "SAE OBD-II Standard (Libre DB)",
            "source_url":   "https://github.com/Libre-Diagnosctic/libre-automotive-diagnostic",
            "image_keys":   "",
        })
    print(f"  Loaded {len(rows)} rows from libre dtc_db.json")
    return rows


def load_nhtsa(path: str) -> list[dict]:
    """Load NHTSA complaint summaries if the file exists."""
    if not os.path.exists(path):
        print("  NHTSA file not found — skipping (run fetch_nhtsa.py first)")
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            rag_text = r.get("rag_text", "").strip()
            if not rag_text:
                continue
            rows.append({
                "id":           f"nhtsa_{i}",
                "document":     rag_text,
                "dtc_code":     r.get("dtc_code", ""),
                "engine_code":  r.get("engine_code", "Gasoline (common)"),
                "description":  r.get("description", ""),
                "source":       "NHTSA Complaints API",
                "source_url":   "https://api.nhtsa.gov/",
                "image_keys":   "",
            })
    print(f"  Loaded {len(rows)} rows from NHTSA")
    return rows


def load_lemon(path: str) -> list[dict]:
    """Load LEMON manual pages from JSONL if the file exists."""
    if not os.path.exists(path):
        print("  LEMON JSONL not found — skipping")
        print("    (To add real repair manuals: run  python src/extract_lemon.py  in WSL first)")
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec  = json.loads(line)
                meta = rec.get("metadata", {})
                rows.append({
                    "id":           rec["id"],
                    "document":     rec["document"],
                    "dtc_code":     meta.get("dtc_code",    ""),
                    "engine_code":  meta.get("engine_code", ""),
                    "description":  meta.get("description", ""),
                    "source":       meta.get("source",      "LEMON Vehicle Manual Database"),
                    "source_url":   meta.get("source_url",  ""),
                    "image_keys":   meta.get("image_keys",  ""),
                })
            except (json.JSONDecodeError, KeyError):
                continue
    print(f"  Loaded {len(rows):,} chunks from LEMON JSONL")
    return rows


def deduplicate(rows: list[dict]) -> list[dict]:
    """Remove duplicate IDs, keeping the first occurrence."""
    seen = set()
    out  = []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            out.append(r)
    print(f"  After dedup: {len(out)} unique records")
    return out


def upsert_to_chroma(rows: list[dict], vs_dir: str):
    """Embed and upsert all rows into ChromaDB."""
    os.makedirs(vs_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=vs_dir)

    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Drop and recreate to ensure clean build
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Dropped existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches
    for i in tqdm(range(0, len(rows), BATCH_SIZE), desc="Embedding"):
        batch = rows[i : i + BATCH_SIZE]
        collection.upsert(
            ids        = [r["id"]       for r in batch],
            documents  = [r["document"] for r in batch],
            metadatas  = [{
                "dtc_code":    r["dtc_code"],
                "engine_code": r["engine_code"],
                "description": r["description"],
                "source":      r["source"],
                "source_url":  r["source_url"],
                "image_keys":  r.get("image_keys", ""),
            } for r in batch],
        )

    print(f"\nVector store built: {collection.count()} documents in '{COLLECTION_NAME}'")
    print(f"Persisted at: {vs_dir}")


def build():
    print("=== Building Toyota Diagnostic Vector Store ===\n")

    print("Loading sources …")
    rows = []
    rows += load_rag_csv(RAG_CSV)
    rows += load_libre_toyota(os.path.join(LIBRE_DIR, "toyota.json"))
    rows += load_libre_obd2(os.path.join(LIBRE_DIR,  "dtc_db.json"))
    rows += load_nhtsa(NHTSA_CSV)
    rows += load_lemon(LEMON_JSONL)

    print(f"\nTotal before dedup: {len(rows)}")
    rows = deduplicate(rows)

    print("\nEmbedding and upserting …")
    upsert_to_chroma(rows, VS_DIR)
    print("\nDone.")


if __name__ == "__main__":
    build()
