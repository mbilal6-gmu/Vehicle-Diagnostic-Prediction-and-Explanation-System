"""
extract_lemon.py
================
One-time extraction script: reads the LEMON automotive manual database
and produces two outputs for Toyota vehicles only:

  Data/lemon/toyota_pages.jsonl   — text chunks, one JSON per line
  Data/lemon/images/{key}.png     — referenced wiring/component diagram PNGs
                                    (only when images.mtbl is present)

NO pymtbl REQUIRED — calls libmtbl.so directly via ctypes.
Works with libmtbl 1.7.1+ built from source.

Prerequisites (WSL):
  sudo apt-get install -y python3-pip
  pip3 install beautifulsoup4 tqdm
  # libmtbl must already be installed at /usr/local/lib/libmtbl.so (or .so.1)

Usage:
  # In WSL terminal (from project root):
  python3 src/extract_lemon.py --lemon-path "/home/mbilal6/lemon"

  # Show first 20 raw keys to debug key format:
  python3 src/extract_lemon.py --lemon-path "/home/mbilal6/lemon" --probe

After this completes, run on Windows to ingest into ChromaDB:
  python src/build_vectorstore.py
"""

import os
import re
import sys
import json
import ctypes
import hashlib
import argparse
import textwrap
from pathlib import Path

# ── Config knobs ─────────────────────────────────────────────────────────────
MAX_PAGES_PER_VEHICLE = 150
CHUNK_WORDS           = 350
CHUNK_OVERLAP         = 50
MIN_CHUNK_WORDS       = 30
TARGET_MAKE           = "toyota"
MTBL_RES_SUCCESS      = 1
# ─────────────────────────────────────────────────────────────────────────────

DTC_RE = re.compile(r'\b([PBCU][0-9]{4})\b', re.IGNORECASE)
IMG_RE = re.compile(r'<img[^>]+src=["\']?([^"\'>\s]+)["\']?', re.IGNORECASE)


# ── ctypes wrapper for libmtbl ────────────────────────────────────────────────

def _load_libmtbl() -> ctypes.CDLL:
    """Find and load libmtbl.so, returning a configured CDLL object."""
    candidates = [
        "/usr/local/lib/libmtbl.so.1",
        "/usr/local/lib/libmtbl.so",
        "/usr/lib/x86_64-linux-gnu/libmtbl.so.1",
        "/usr/lib/x86_64-linux-gnu/libmtbl.so",
        "/usr/lib/libmtbl.so.1",
        "/usr/lib/libmtbl.so",
        "libmtbl.so.1",
        "libmtbl.so",
    ]
    lib = None
    for path in candidates:
        try:
            lib = ctypes.CDLL(path)
            # Quick sanity check
            _ = lib.mtbl_reader_init
            _ = lib.mtbl_reader_source
            _ = lib.mtbl_iter_next
            print(f"  Loaded libmtbl from: {path}")
            break
        except (OSError, AttributeError):
            continue

    if lib is None:
        print("ERROR: libmtbl.so not found in standard locations.")
        print("  Make sure libmtbl 1.7.1 is installed:")
        print("    cd /tmp/mtbl-src && sudo make install && sudo ldconfig")
        sys.exit(1)

    # ── Configure function signatures ────────────────────────────────────────

    # reader
    lib.mtbl_reader_init.restype  = ctypes.c_void_p
    lib.mtbl_reader_init.argtypes = [ctypes.c_char_p, ctypes.c_void_p]

    lib.mtbl_reader_destroy.restype  = None
    lib.mtbl_reader_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    lib.mtbl_reader_source.restype  = ctypes.c_void_p
    lib.mtbl_reader_source.argtypes = [ctypes.c_void_p]

    # source queries — returns struct mtbl_iter *
    lib.mtbl_source_get_range.restype  = ctypes.c_void_p
    lib.mtbl_source_get_range.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.c_size_t,
    ]

    # mtbl_source_get_prefix may not exist in all versions — try, fall back to range
    try:
        lib.mtbl_source_get_prefix.restype  = ctypes.c_void_p
        lib.mtbl_source_get_prefix.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
        ]
        _has_get_prefix = True
    except AttributeError:
        _has_get_prefix = False

    # iter
    lib.mtbl_iter_next.restype  = ctypes.c_int
    lib.mtbl_iter_next.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.POINTER(ctypes.c_size_t),
    ]

    lib.mtbl_iter_destroy.restype  = None
    lib.mtbl_iter_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    lib._has_get_prefix = _has_get_prefix   # stash for use by MtblReader
    return lib


def _prefix_end(prefix: bytes) -> bytes:
    """
    Compute the exclusive upper bound for a prefix scan.
    e.g.  b"abc" → b"abd"   (increments last non-0xFF byte)
    """
    buf = bytearray(prefix)
    for i in range(len(buf) - 1, -1, -1):
        if buf[i] < 0xFF:
            buf[i] += 1
            return bytes(buf[: i + 1])
    # All bytes are 0xFF — no finite upper bound; use a very long sentinel
    return b"\xff" * 512


class MtblReader:
    """
    Lightweight ctypes wrapper around an open MTBL file.
    Provides get_prefix() and get_range() iterator methods.
    """

    def __init__(self, path: str, lib: ctypes.CDLL):
        self._lib  = lib
        bpath      = path.encode() if isinstance(path, str) else path
        self._ptr  = lib.mtbl_reader_init(bpath, None)
        if not self._ptr:
            raise RuntimeError(f"mtbl_reader_init failed for: {path}")
        self._src  = lib.mtbl_reader_source(self._ptr)
        if not self._src:
            raise RuntimeError(f"mtbl_reader_source returned NULL for: {path}")

    # ── public API ────────────────────────────────────────────────────────── #

    def get_prefix(self, prefix: bytes):
        """Yield (key, value) bytes pairs whose key starts with *prefix*."""
        if self._lib._has_get_prefix:
            iter_ptr = self._lib.mtbl_source_get_prefix(
                self._src, prefix, len(prefix)
            )
        else:
            iter_ptr = self._lib.mtbl_source_get_range(
                self._src,
                prefix, len(prefix),
                _prefix_end(prefix), len(_prefix_end(prefix)),
            )
        yield from self._drain(iter_ptr)

    def get_range(self, start: bytes, end: bytes):
        """Yield (key, value) bytes pairs in the half-open range [start, end)."""
        iter_ptr = self._lib.mtbl_source_get_range(
            self._src,
            start, len(start),
            end,   len(end),
        )
        yield from self._drain(iter_ptr)

    def sample_keys(self, n: int = 20):
        """Yield first *n* (key, value) pairs from the file (for probing)."""
        # Range from empty start to sentinel end
        iter_ptr = self._lib.mtbl_source_get_range(
            self._src, b"", 0, b"\xff" * 512, 512
        )
        count = 0
        for kv in self._drain(iter_ptr):
            yield kv
            count += 1
            if count >= n:
                break

    # ── internal ──────────────────────────────────────────────────────────── #

    def _drain(self, iter_ptr):
        """Generator: exhausts an mtbl_iter, then destroys it."""
        if not iter_ptr:
            return

        key_p   = ctypes.POINTER(ctypes.c_uint8)()
        len_key = ctypes.c_size_t(0)
        val_p   = ctypes.POINTER(ctypes.c_uint8)()
        len_val = ctypes.c_size_t(0)

        try:
            while True:
                res = self._lib.mtbl_iter_next(
                    iter_ptr,
                    ctypes.byref(key_p), ctypes.byref(len_key),
                    ctypes.byref(val_p), ctypes.byref(len_val),
                )
                if res != MTBL_RES_SUCCESS:
                    break
                k = bytes(bytearray(key_p[: len_key.value]))
                v = bytes(bytearray(val_p[: len_val.value]))
                yield k, v
        finally:
            holder = ctypes.c_void_p(iter_ptr)
            self._lib.mtbl_iter_destroy(ctypes.byref(holder))

    def close(self):
        if self._ptr:
            h = ctypes.c_void_p(self._ptr)
            self._lib.mtbl_reader_destroy(ctypes.byref(h))
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── Dependency checks ─────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    try:
        from bs4 import BeautifulSoup  # noqa: F401
    except ImportError:
        missing.append("beautifulsoup4")
    try:
        from tqdm import tqdm          # noqa: F401
    except ImportError:
        missing.append("tqdm")
    if missing:
        print("Missing Python packages:", ", ".join(missing))
        print("  pip3 install " + " ".join(missing))
        sys.exit(1)


# ── Text helpers ───────────────────────────────────────────────────────────────

def _html_to_text(html: str) -> tuple[str, list[str]]:
    from bs4 import BeautifulSoup
    img_keys = IMG_RE.findall(html)
    soup     = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text, img_keys


def _chunk_text(text: str) -> list[str]:
    words  = text.split()
    step   = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for i in range(0, max(1, len(words)), step):
        chunk = " ".join(words[i : i + CHUNK_WORDS])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
    return chunks


def _first_sentence(text: str) -> str:
    line = text.split("\n")[0].strip()
    return textwrap.shorten(line, width=120, placeholder="…")


# ── Key-format detection ──────────────────────────────────────────────────────

def _detect_key_format(pages: MtblReader, sample_vehicle: dict,
                       probe_print: bool = False) -> str:
    """
    Probe pages.mtbl to determine how vehicle page keys are structured.
    Returns: "uri_prefix" | "uri_path" | "unknown"
    """
    if probe_print:
        print("\n  ── Sample keys from pages.mtbl ──")
        for i, (k, v) in enumerate(pages.sample_keys(20)):
            print(f"  [{i:02d}] key={k[:120]!r}  val_len={len(v)}")
        print()
        return "unknown"

    root_uri = sample_vehicle["rootUriTable"].encode()
    uri_path = sample_vehicle["uriPath"].encode()

    # Test 1: rootUriTable as key prefix
    try:
        count = sum(1 for _ in pages.get_prefix(root_uri))
        if count > 0:
            print(f"  Key format: rootUriTable prefix ({count} pages for sample vehicle)")
            return "uri_prefix"
    except Exception as e:
        print(f"  [WARN] uri_prefix probe failed: {e}")

    # Test 2: raw URI path as key prefix
    try:
        count = sum(1 for _ in pages.get_prefix(uri_path))
        if count > 0:
            print(f"  Key format: uriPath prefix ({count} pages for sample vehicle)")
            return "uri_path"
    except Exception as e:
        print(f"  [WARN] uri_path probe failed: {e}")

    return "unknown"


# ── Image extraction ──────────────────────────────────────────────────────────

def _save_images(images: MtblReader | None, img_raw_keys: list[str],
                 images_dir: Path) -> list[str]:
    if images is None or not img_raw_keys:
        return []
    saved = []
    for raw_key in img_raw_keys:
        key   = raw_key.strip("/").encode()
        fname = re.sub(r'[^\w\-.]', '_', raw_key.strip("/")) + ".png"
        fpath = images_dir / fname
        if fpath.exists():
            saved.append(fname)
            continue
        try:
            for _, v in images.get_prefix(key):
                fpath.write_bytes(v)
                saved.append(fname)
                break
        except Exception:
            pass
    return saved


# ── Page processing ───────────────────────────────────────────────────────────

def _process_page(raw_value: bytes, vehicle: dict, page_key: str,
                  images: MtblReader | None, images_dir: Path,
                  out_f) -> tuple[int, int]:
    """Parse one page, chunk it, write JSONL records. Returns (chunks, images)."""
    try:
        html = raw_value.decode("utf-8", errors="replace")
    except Exception:
        return 0, 0

    text, img_raw_keys = _html_to_text(html)
    if not text.strip():
        return 0, 0

    saved_images   = _save_images(images, img_raw_keys, images_dir)
    image_keys_str = ",".join(saved_images)
    chunks         = _chunk_text(text)

    model    = vehicle.get("model", "").lower()
    engine   = vehicle.get("engine", "")
    year     = vehicle["years"][0] if vehicle.get("years") else "unknown"
    uri_path = vehicle.get("uriPath", page_key)

    for chunk_idx, chunk in enumerate(chunks):
        dtcs     = DTC_RE.findall(chunk)
        dtc_code = dtcs[0].upper() if dtcs else ""
        doc_id   = "lemon_" + hashlib.md5(
            f"{page_key}_{chunk_idx}".encode()
        ).hexdigest()[:16]

        out_f.write(json.dumps({
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
        }) + "\n")

    return len(chunks), len(saved_images)


# ── Main extraction ───────────────────────────────────────────────────────────

def probe(lemon_path: str):
    """Print sample keys from pages.mtbl to help identify key format."""
    lib        = _load_libmtbl()
    pages_path = Path(lemon_path) / "pages.mtbl"
    index_path = Path(lemon_path) / "index.json"

    print(f"Loading index …")
    with open(index_path, encoding="utf-8") as f:
        idx      = json.load(f)
    vehicles = idx.get("vehicles", idx) if isinstance(idx, dict) else idx
    toyota   = [v for v in vehicles if v.get("make", "").lower() == TARGET_MAKE]
    print(f"  Sample Toyota vehicle: {toyota[0]}\n")

    print(f"Opening pages.mtbl …")
    with MtblReader(str(pages_path), lib) as pages:
        _detect_key_format(pages, toyota[0], probe_print=True)


def extract(lemon_path: str, output_dir: str):
    from tqdm import tqdm

    _check_deps()
    lib = _load_libmtbl()

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
    print(f"Reading index.json …")
    with open(index_path, encoding="utf-8") as f:
        idx = json.load(f)
    vehicles = idx.get("vehicles", idx) if isinstance(idx, dict) else idx
    toyota   = [v for v in vehicles if v.get("make", "").lower() == TARGET_MAKE]
    print(f"  {len(toyota):,} Toyota vehicles of {len(vehicles):,} total")

    # ── Open readers ──────────────────────────────────────────────────────────
    print("Opening pages.mtbl …")
    pages = MtblReader(str(pages_path), lib)

    images = None
    if images_path.exists() and images_path.stat().st_size > 0:
        print("Opening images.mtbl …")
        try:
            images = MtblReader(str(images_path), lib)
        except Exception as e:
            print(f"  [WARN] images.mtbl could not be opened: {e} — skipping images")
    else:
        print("  images.mtbl not present — images will be skipped")

    # ── Detect key format ─────────────────────────────────────────────────────
    print("Detecting key format …")
    key_format = _detect_key_format(pages, toyota[0])
    if key_format == "unknown":
        print(
            "\n  ⚠️  Key format undetected.  Run with --probe to inspect raw keys:\n"
            "       python3 src/extract_lemon.py --lemon-path ... --probe\n"
            "  Then open a GitHub issue with the output so the key format can be added.\n"
        )
        pages.close()
        if images:
            images.close()
        sys.exit(1)

    # ── Extract ───────────────────────────────────────────────────────────────
    total_chunks = 0
    total_images = 0
    skipped      = 0

    print(f"\nExtracting → {jsonl_path} …")
    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for vehicle in tqdm(toyota, desc="Toyota vehicles"):

            prefix = (
                vehicle["rootUriTable"].encode()
                if key_format == "uri_prefix"
                else vehicle["uriPath"].encode()
            )

            pages_seen = 0
            for raw_key, raw_value in pages.get_prefix(prefix):
                if pages_seen >= MAX_PAGES_PER_VEHICLE:
                    break
                n_c, n_i = _process_page(
                    raw_value, vehicle,
                    raw_key.decode("utf-8", errors="replace"),
                    images, images_dir, out_f,
                )
                total_chunks += n_c
                total_images += n_i
                pages_seen   += 1

            if pages_seen == 0:
                skipped += 1

    pages.close()
    if images:
        images.close()

    print(f"\n{'='*50}")
    print(f"  Chunks written         : {total_chunks:,}")
    print(f"  Images saved           : {total_images:,}")
    print(f"  Vehicles with no pages : {skipped:,}")
    print(f"  Output JSONL           : {jsonl_path}")
    print(f"  Output images          : {images_dir}/")
    print(f"\nNext step (Windows):\n  python src/build_vectorstore.py")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global MAX_PAGES_PER_VEHICLE

    parser = argparse.ArgumentParser(
        description="Extract Toyota pages from LEMON database into RAG-ready JSONL."
    )
    parser.add_argument(
        "--lemon-path", required=True,
        help="Path to lemon/ folder containing index.json + pages.mtbl",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "Data", "lemon"),
        help="Output directory (default: Data/lemon/)",
    )
    parser.add_argument(
        "--max-pages", type=int, default=MAX_PAGES_PER_VEHICLE,
        help=f"Max pages per vehicle (default: {MAX_PAGES_PER_VEHICLE})",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Print 20 sample keys from pages.mtbl then exit (for debugging key format)",
    )
    args = parser.parse_args()

    MAX_PAGES_PER_VEHICLE = args.max_pages

    if args.probe:
        probe(args.lemon_path)
    else:
        extract(args.lemon_path, args.output_dir)


if __name__ == "__main__":
    main()
