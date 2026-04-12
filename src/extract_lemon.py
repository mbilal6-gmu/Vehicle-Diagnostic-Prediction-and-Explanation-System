"""
extract_lemon.py
================
One-time extraction script: reads the LEMON automotive manual database
and produces two outputs for Toyota vehicles only:

  Data/lemon/toyota_pages.jsonl   — text chunks, one JSON per line
  Data/lemon/images/{key}.png     — referenced wiring/component diagram PNGs

Prerequisites — run in WSL (Windows Subsystem for Linux):
  sudo apt-get install -y libmtbl-dev
  pip install pymtbl beautifulsoup4 tqdm

Usage:
  # In WSL terminal:
  python src/extract_lemon.py --lemon-path "/mnt/x/lemon-manuals/lemon"

  # On Windows (if libmtbl compiled natively):
  python src/extract_lemon.py --lemon-path "X:/lemon-manuals/lemon"

After this script completes, run on Windows:
  python src/build_vectorstore.py   (picks up Data/lemon/toyota_pages.jsonl automatically)
"""

import os
import re
import sys
import json
import hashlib
import argparse
import textwrap
from pathlib import Path

# ── Config knobs ─────────────────────────────────────────────────────────────
MAX_PAGES_PER_VEHICLE = 150     # cap per vehicle to prevent runaway manuals
CHUNK_WORDS           = 350     # target words per RAG chunk
CHUNK_OVERLAP         = 50      # word overlap between consecutive chunks
MIN_CHUNK_WORDS       = 30      # discard chunks shorter than this
TARGET_MAKE           = "toyota"
# ─────────────────────────────────────────────────────────────────────────────

DTC_RE  = re.compile(r'\b([PBCU][0-9]{4})\b', re.IGNORECASE)
IMG_RE  = re.compile(r'<img[^>]+src=["\']?([^"\'>\s]+)["\']?', re.IGNORECASE)


# ── Dependency checks ─────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    try:
        import mtbl          # noqa: F401  (pymtbl)
    except ImportError:
        missing.append("pymtbl")
    try:
        from bs4 import BeautifulSoup   # noqa: F401
    except ImportError:
        missing.append("beautifulsoup4")
    try:
        from tqdm import tqdm           # noqa: F401
    except ImportError:
        missing.append("tqdm")

    if missing:
        print("=" * 60)
        print("Missing dependencies:", ", ".join(missing))
        print("\nInstall in WSL:")
        print("  sudo apt-get install -y libmtbl-dev")
        print("  pip install " + " ".join(missing))
        print("=" * 60)
        sys.exit(1)


# ── Text helpers ───────────────────────────────────────────────────────────────

def _html_to_text(html: str) -> tuple[str, list[str]]:
    """
    Returns (plain_text, image_key_list).
    Image keys are the src values from <img> tags.
    """
    from bs4 import BeautifulSoup
    img_keys = IMG_RE.findall(html)
    soup     = BeautifulSoup(html, "html.parser")
    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excess whitespace
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text, img_keys


def _chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-level chunks."""
    words  = text.split()
    step   = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for i in range(0, max(1, len(words) - MIN_CHUNK_WORDS + 1), step):
        chunk = " ".join(words[i : i + CHUNK_WORDS])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
    return chunks


def _first_sentence(text: str) -> str:
    """Return a short description (first 120 chars)."""
    line = text.split("\n")[0].strip()
    return textwrap.shorten(line, width=120, placeholder="…")


# ── MTBL key-format detection ─────────────────────────────────────────────────

def _detect_key_format(reader, sample_vehicle: dict) -> str:
    """
    Probe pages.mtbl to determine how vehicle page keys are structured.
    Returns one of: "uri_prefix" | "exact_root" | "unknown"
    """
    root_uri  = sample_vehicle["rootUriTable"].encode()
    uri_path  = sample_vehicle["uriPath"].encode()

    # Test 1: prefix = rootUriTable string
    try:
        count = sum(1 for _ in reader.get_range(root_uri, root_uri + b'\xff'))
        if count > 0:
            return "uri_prefix"
    except Exception:
        pass

    # Test 2: prefix = raw URI path (e.g. "/Toyota/2018/Camry/")
    try:
        count = sum(1 for _ in reader.get_range(uri_path, uri_path + b'\xff'))
        if count > 0:
            return "uri_path"
    except Exception:
        pass

    return "unknown"


# ── Page extraction ───────────────────────────────────────────────────────────

def _iter_vehicle_pages(reader, vehicle: dict, key_format: str):
    """
    Yield (key_bytes, value_bytes) for all pages of a given vehicle.
    Caps at MAX_PAGES_PER_VEHICLE.
    """
    count = 0

    if key_format == "uri_prefix":
        prefix = vehicle["rootUriTable"].encode()
        it = reader.get_range(prefix, prefix + b'\xff')
    elif key_format == "uri_path":
        prefix = vehicle["uriPath"].encode()
        it = reader.get_range(prefix, prefix + b'\xff')
    else:
        return   # unknown format — skip silently

    for key, value in it:
        if count >= MAX_PAGES_PER_VEHICLE:
            break
        yield key, value
        count += 1


# ── Image extraction ──────────────────────────────────────────────────────────

def _save_images(img_reader, img_keys: list[str], images_dir: Path) -> list[str]:
    """
    Look up each image key in images.mtbl and save as PNG.
    Returns list of successfully saved filenames.
    """
    saved = []
    for raw_key in img_keys:
        # Normalise key: strip leading slashes, URL-decode if needed
        key    = raw_key.strip("/").encode()
        fname  = re.sub(r'[^\w\-.]', '_', raw_key.strip("/")) + ".png"
        fpath  = images_dir / fname

        if fpath.exists():
            saved.append(fname)
            continue

        try:
            value = img_reader.get(key)
            if value:
                fpath.write_bytes(value)
                saved.append(fname)
        except Exception:
            pass   # image not found — skip

    return saved


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(lemon_path: str, output_dir: str):
    import mtbl
    from tqdm import tqdm

    lemon_path  = Path(lemon_path)
    pages_path  = lemon_path / "pages.mtbl"
    images_path = lemon_path / "images.mtbl"
    index_path  = lemon_path / "index.json"

    for p in [pages_path, index_path]:
        if not p.exists():
            print(f"ERROR: Not found: {p}")
            sys.exit(1)

    out_dir    = Path(output_dir)
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "toyota_pages.jsonl"

    # ── Load index ────────────────────────────────────────────────────────────
    print(f"Reading index.json ({index_path.stat().st_size / 1e6:.0f} MB) …")
    with open(index_path, encoding="utf-8") as f:
        idx = json.load(f)

    vehicles = idx.get("vehicles", idx) if isinstance(idx, dict) else idx
    toyota   = [v for v in vehicles if v.get("make", "").lower() == TARGET_MAKE]
    print(f"Found {len(toyota)} Toyota vehicles out of {len(vehicles)} total")

    # ── Open MTBL readers ─────────────────────────────────────────────────────
    print("Opening pages.mtbl reader …")
    pages_reader  = mtbl.reader(str(pages_path),  verify_checksums=False)
    images_reader = None
    if images_path.exists():
        print("Opening images.mtbl reader …")
        try:
            images_reader = mtbl.reader(str(images_path), verify_checksums=False)
        except Exception as e:
            print(f"  [WARN] Could not open images.mtbl: {e}  (images will be skipped)")

    # ── Detect key format ─────────────────────────────────────────────────────
    print("Detecting key format …")
    key_format = _detect_key_format(pages_reader, toyota[0])
    print(f"  Key format detected: {key_format}")
    if key_format == "unknown":
        print(
            "\n  [WARN] Could not determine key structure in pages.mtbl.\n"
            "  The LEMON database version may differ from what was expected.\n"
            "  Trying full iteration fallback (slow but safe) …\n"
        )
        # Fallback: iterate ALL keys and keep ones matching Toyota URI paths
        # This is slow (29 GB scan) but works regardless of key format.
        key_format = "full_scan"
        toyota_uris = {v["uriPath"].encode() for v in toyota}

    # ── Extract pages → JSONL ─────────────────────────────────────────────────
    total_chunks  = 0
    total_images  = 0
    skipped       = 0

    print(f"\nExtracting pages → {jsonl_path} …")

    with open(jsonl_path, "w", encoding="utf-8") as out_f:

        if key_format == "full_scan":
            # Map from URI prefix → vehicle metadata for O(1) lookup
            uri_to_vehicle = {}
            for v in toyota:
                uri_to_vehicle[v["uriPath"].encode()] = v

            for raw_key, raw_value in tqdm(pages_reader, desc="Scanning all keys"):
                # Match key against any Toyota URI prefix
                matched_vehicle = None
                for uri_prefix, veh in uri_to_vehicle.items():
                    if raw_key.startswith(uri_prefix):
                        matched_vehicle = veh
                        break
                if not matched_vehicle:
                    continue
                _process_page(
                    raw_value, matched_vehicle, raw_key.decode("utf-8", errors="replace"),
                    images_reader, images_dir, out_f,
                    total_chunks_ref=[total_chunks], total_images_ref=[total_images],
                )
                total_chunks = total_chunks  # updated inside helper via ref

        else:
            for vehicle in tqdm(toyota, desc="Vehicles"):
                pages_seen = 0
                for raw_key, raw_value in _iter_vehicle_pages(pages_reader, vehicle, key_format):
                    n_chunks, n_images = _process_page_direct(
                        raw_value, vehicle, raw_key.decode("utf-8", errors="replace"),
                        images_reader, images_dir, out_f,
                    )
                    total_chunks += n_chunks
                    total_images += n_images
                    pages_seen   += 1

                if pages_seen == 0:
                    skipped += 1

    print(f"\n=== Extraction complete ===")
    print(f"  Chunks written : {total_chunks:,}")
    print(f"  Images saved   : {total_images:,}")
    print(f"  Vehicles with no pages: {skipped}")
    print(f"  Output JSONL   : {jsonl_path}")
    print(f"  Output images  : {images_dir}/")
    print(f"\nNext step (on Windows):\n  python src/build_vectorstore.py")


def _process_page_direct(
    raw_value: bytes,
    vehicle: dict,
    page_key: str,
    images_reader,
    images_dir: Path,
    out_f,
) -> tuple[int, int]:
    """
    Process one raw page value: parse HTML, chunk, save images, write JSONL.
    Returns (chunks_written, images_saved).
    """
    try:
        html = raw_value.decode("utf-8", errors="replace")
    except Exception:
        return 0, 0

    text, img_raw_keys = _html_to_text(html)
    if not text.strip():
        return 0, 0

    # Save referenced images
    saved_images = []
    if images_reader and img_raw_keys:
        saved_images = _save_images(images_reader, img_raw_keys, images_dir)

    image_keys_str = ",".join(saved_images)
    chunks         = _chunk_text(text)
    model          = vehicle.get("model", "").lower()
    engine         = vehicle.get("engine", "")
    year           = vehicle["years"][0] if vehicle.get("years") else "unknown"
    uri_path       = vehicle.get("uriPath", page_key)

    for chunk_idx, chunk in enumerate(chunks):
        dtcs     = DTC_RE.findall(chunk)
        dtc_code = dtcs[0].upper() if dtcs else ""
        doc_id   = "lemon_" + hashlib.md5(
            f"{page_key}_{chunk_idx}".encode()
        ).hexdigest()[:16]

        record = {
            "id":       doc_id,
            "document": f"Toyota {model.title()} {year} | Engine: {engine}\n\n{chunk}",
            "metadata": {
                "dtc_code":    dtc_code,
                "engine_code": engine,
                "description": _first_sentence(chunk),
                "source":      "LEMON Vehicle Manual Database",
                "source_url":  uri_path,
                "image_keys":  image_keys_str,
            },
        }
        out_f.write(json.dumps(record) + "\n")

    return len(chunks), len(saved_images)


# Alias for full_scan mode (uses ref lists to allow mutation from inner scope)
def _process_page(raw_value, vehicle, page_key, images_reader, images_dir, out_f,
                  total_chunks_ref, total_images_ref):
    n_c, n_i = _process_page_direct(
        raw_value, vehicle, page_key, images_reader, images_dir, out_f
    )
    total_chunks_ref[0] += n_c
    total_images_ref[0] += n_i


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract Toyota pages from LEMON database into RAG-ready JSONL."
    )
    parser.add_argument(
        "--lemon-path",
        required=True,
        help='Path to lemon/ folder containing index.json + pages.mtbl + images.mtbl',
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "Data", "lemon"),
        help="Output directory for toyota_pages.jsonl and images/ (default: Data/lemon/)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=MAX_PAGES_PER_VEHICLE,
        help=f"Max pages per vehicle (default: {MAX_PAGES_PER_VEHICLE})",
    )
    args = parser.parse_args()

    global MAX_PAGES_PER_VEHICLE
    MAX_PAGES_PER_VEHICLE = args.max_pages

    _check_deps()
    extract(args.lemon_path, args.output_dir)


if __name__ == "__main__":
    main()
