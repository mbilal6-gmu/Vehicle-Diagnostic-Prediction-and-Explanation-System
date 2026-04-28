"""
extract_lemon.py
================
Sequential scanner: reads the LEMON pages.mtbl and emits RAG-ready JSONL
for in-scope Toyota vehicles (2014-2020, target models only).

Strategy: bypass the MTBL index entirely. Scan every data block from the
start of the file to index_block_offset, reconstruct pages via the
growing-key scheme, filter by Toyota content keywords, emit JSONL.

Prerequisites (WSL):
    pip3 install --break-system-packages zstandard beautifulsoup4 tqdm

Usage:
    python3 src/extract_lemon.py --lemon-path "/home/mbilal6/lemon"

    # Optional: inspect footer fields without extracting
    python3 src/extract_lemon.py --lemon-path "/home/mbilal6/lemon" --debug

After extraction, rebuild the vector store on Windows:
    python src/build_vectorstore.py
"""

import os, re, json, struct, hashlib, argparse
from pathlib import Path

# ── Scope ─────────────────────────────────────────────────────────────────────
TARGET_YEARS  = {str(y) for y in range(2014, 2021)}
TARGET_MODELS = {
    b"camry", b"corolla", b"rav4", b"highlander", b"prius",
    b"tacoma", b"4runner", b"sienna", b"yaris", b"avalon",
}

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_WORDS     = 350
CHUNK_OVERLAP   = 50
MIN_CHUNK_WORDS = 30

# ── MTBL constants ────────────────────────────────────────────────────────────
MTBL_MAGIC       = 0x4D54424C
MTBL_FOOTER_SIZE = 512
COMPRESSION_ZSTD = 5

# Page type suffixes used to locate the fixed page_id prefix in a key.
# Key structure: [varint][bulletin_][64-hex-sha256][_T or _S or _J or _I]
#                                                   ^^^^^^^^^^^^^^^^^^^^
#                                                   type suffix = page_id end
PAGE_TYPE_SUFFIXES = (b"_T", b"_S", b"_J", b"_I")


# ── Varint ────────────────────────────────────────────────────────────────────

def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while pos < len(data):
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


# ── Footer ────────────────────────────────────────────────────────────────────

def _read_footer(pages_path: str) -> dict:
    """Parse the 512-byte MTBL footer. Returns a dict with key fields."""
    size = os.path.getsize(pages_path)
    with open(pages_path, "rb") as f:
        f.seek(size - MTBL_FOOTER_SIZE)
        raw = f.read(MTBL_FOOTER_SIZE)

    magic = struct.unpack_from("<I", raw, 508)[0]
    if magic != MTBL_MAGIC:
        raise RuntimeError(
            f"MTBL magic not found at footer offset 508 (got {raw[508:512].hex()}). "
            f"File may be incomplete or corrupt."
        )

    # Footer: 8 × uint64 little-endian starting at byte 0.
    # Field layout (confirmed from footer inspection):
    #   [0] index_block_offset
    #   [1] data_block_size
    #   [2] compression_algorithm   ← 5 = zstd
    #   [3] key_bytes
    #   [4] val_bytes
    #   [5] (mirrors index_block_offset)
    #   [6] index_bytes
    #   [7] count (approximate)
    fields = struct.unpack_from("<8Q", raw, 0)
    return {
        "index_block_offset": fields[0],
        "compression":        int(fields[2]),
        "file_size":          size,
    }


# ── Block iteration ───────────────────────────────────────────────────────────

def _iter_data_blocks(pages_path: str, index_block_offset: int, dctx):
    """
    Yield (block_offset, decompressed_block_bytes) for every data block.

    Block wire format:
        varint(compressed_size)   1–5 bytes
        uint32 CRC32              4 bytes
        compressed_data           compressed_size bytes (zstd frame)
    """
    with open(pages_path, "rb") as f:
        offset = 0
        while offset < index_block_offset:
            f.seek(offset)
            # Read enough bytes to decode the varint (max 5 bytes) + CRC (4 bytes)
            header = f.read(9)
            if not header:
                break

            compressed_size, varint_end = _read_varint(header, 0)
            if compressed_size == 0:
                break

            block_total = varint_end + 4 + compressed_size

            # Skip only truly massive blocks (> 200 KB compressed).
            # These decompress to 10–21 MB with 20K–130K entries and contain
            # no useful page content. Normal TSB blocks are 10–28 KB.
            if compressed_size > 200_000:
                offset += block_total
                continue

            data_start = offset + varint_end + 4  # skip varint + CRC32
            f.seek(data_start)
            compressed = f.read(compressed_size)
            if len(compressed) < compressed_size:
                break

            try:
                block = dctx.decompress(compressed, max_output_size=32 * 1024 * 1024)
            except Exception:
                block = b""

            yield offset, block, block_total
            offset += block_total


# ── Block entry parser ────────────────────────────────────────────────────────

def _parse_entries(data: bytes):
    """
    Yield (key, value) from a decompressed MTBL block.
    Uses LevelDB-style prefix compression:
        varint(shared) varint(delta_len) delta varint(val_len) val
    Block trailer: [uint32 × num_restarts][uint32 num_restarts]

    Uses bytearray for in-place key mutation — avoids the O(key_len) copy
    that immutable bytes would produce on every entry for the growing-key scheme.
    """
    if len(data) < 4:
        return

    num_restarts = struct.unpack_from("<I", data, len(data) - 4)[0]
    max_r        = len(data) // 4
    entries_end  = len(data) - 4 - 4 * num_restarts if num_restarts <= max_r else len(data)

    if entries_end <= 0:
        return

    # _KEY_CAP: page_id fits in ~80 bytes; we never need more of the key.
    # Capping at 100 bytes avoids copying 40–640 KB restart keys for deep pages.
    _KEY_CAP = 100

    pos     = 0
    cur_key = bytearray()
    while pos < entries_end:
        try:
            shared,      pos = _read_varint(data, pos)
            dlen,        pos = _read_varint(data, pos)
            if pos + dlen > entries_end:
                break
            delta_start = pos; pos += dlen      # no slice — use offsets
            vlen,        pos = _read_varint(data, pos)
            if pos + vlen > entries_end:
                break
            val = data[pos:pos + vlen]; pos += vlen

            # Keep only the first _KEY_CAP bytes of the logical key.
            keep = min(shared, _KEY_CAP)        # bytes to retain from prev key
            del cur_key[keep:]                  # O(len-keep) ≈ O(0) for end-trim
            take = min(dlen, _KEY_CAP - keep)   # bytes to copy from delta
            if take > 0:
                cur_key.extend(data[delta_start:delta_start + take])  # O(take)≤O(100)

            yield cur_key, val  # yield bytearray ref — caller must not store it
        except Exception:
            break


# ── Page ID extraction ────────────────────────────────────────────────────────

def _get_page_id(key) -> bytes | None:
    """
    Return the fixed page_id prefix of a LEMON key (up to and including the
    type suffix _T/_S/_J/_I).  Returns None if no suffix found.

    Accepts bytes or bytearray. Returns a fresh small bytes object (~77 bytes)
    so the caller doesn't hold a reference to the mutable cur_key bytearray.

    LEMON uses a growing-key scheme: all chunks of a page share the same
    page_id prefix. Entry 0's key IS the page_id; entry N's key is
    page_id + concat(values 0..N-1).
    """
    for suffix in PAGE_TYPE_SUFFIXES:
        pos = key.find(suffix)
        if pos > 0:
            return bytes(key[:pos + 2])  # copy only the small prefix (~77 bytes)
    return None


# ── Scope filter ──────────────────────────────────────────────────────────────

def _is_in_scope(content: bytes) -> bool:
    """Pass any page that mentions Toyota — model/year filtering happens at query time."""
    return b"toyota" in content.lower()


# ── Text helpers ──────────────────────────────────────────────────────────────

_DTC_RE   = re.compile(r'\b([PBCU][0-9]{4})\b', re.IGNORECASE)
_MODEL_RE = re.compile(
    r'\b(camry|corolla|rav4|highlander|prius|tacoma|4runner|sienna|yaris|avalon)\b',
    re.IGNORECASE,
)
_YEAR_RE  = re.compile(r'\b(201[4-9]|2020)\b')


def _html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _chunk_text(text: str) -> list[str]:
    words  = text.split()
    step   = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for i in range(0, max(1, len(words)), step):
        chunk = " ".join(words[i:i + CHUNK_WORDS])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
    return chunks


# ── Page flush ────────────────────────────────────────────────────────────────

def _flush_page(page_id: bytes, value_chunks: list[bytes], out_f) -> int:
    """
    Assemble a page from its value chunks, check scope, convert to text,
    chunk, and write JSONL records. Returns number of chunks written.
    """
    content = b"".join(value_chunks)

    if not _is_in_scope(content):
        return 0

    try:
        html = content.decode("utf-8", errors="replace")
    except Exception:
        return 0

    text = _html_to_text(html)
    if not text.strip():
        return 0

    model_m = _MODEL_RE.search(text)
    year_m  = _YEAR_RE.search(text)
    model   = model_m.group(1).lower() if model_m else "unknown"
    year    = year_m.group(1)          if year_m  else "unknown"
    page_key = page_id.hex()

    written = 0
    for i, chunk in enumerate(_chunk_text(text)):
        dtcs   = _DTC_RE.findall(chunk)
        doc_id = "lemon_" + hashlib.md5(f"{page_key}_{i}".encode()).hexdigest()[:16]
        out_f.write(json.dumps({
            "id":       doc_id,
            "document": f"Toyota {model.title()} {year}\n\n{chunk}",
            "metadata": {
                "dtc_code":    dtcs[0].upper() if dtcs else "",
                "engine_code": "",
                "description": chunk.split("\n")[0][:120],
                "source":      "LEMON Vehicle Manual Database",
                "source_url":  "",
                "image_keys":  "",
            },
        }) + "\n")
        written += 1

    return written


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(lemon_path: str, output_dir: str):
    import zstandard
    from tqdm import tqdm

    try:
        from bs4 import BeautifulSoup  # noqa — check early
    except ImportError:
        print("Missing: beautifulsoup4  →  pip3 install --break-system-packages beautifulsoup4")
        raise SystemExit(1)

    pages_path = Path(lemon_path) / "pages.mtbl"
    if not pages_path.exists():
        print(f"ERROR: not found: {pages_path}")
        raise SystemExit(1)

    out_dir    = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "toyota_pages.jsonl"
    # Note: images.mtbl is empty in this LEMON copy — no images to extract.

    print("Reading footer …")
    footer      = _read_footer(str(pages_path))
    idx_offset  = footer["index_block_offset"]
    compression = footer["compression"]
    file_gb     = footer["file_size"] / 1e9
    print(f"  file size          : {file_gb:.2f} GB")
    print(f"  index_block_offset : {idx_offset:,}")
    print(f"  compression        : {compression} (5=zstd)")

    if compression != COMPRESSION_ZSTD:
        raise RuntimeError(f"Unexpected compression algorithm: {compression} (expected 5=zstd)")

    dctx = zstandard.ZstdDecompressor()

    # page_id → [chunks: list[bytes], has_toyota: bool]
    # has_toyota is set True the moment any chunk contains b"toyota".
    # Non-Toyota pages are discarded at flush time without ever joining chunks.
    page_buffers: dict[bytes, list] = {}

    total_chunks = 0
    total_pages  = 0

    print(f"\nScanning data blocks → {jsonl_path}\n")

    with open(jsonl_path, "w", encoding="utf-8") as out_f:
        for _blk_offset, block, _blk_size in tqdm(
            _iter_data_blocks(str(pages_path), idx_offset, dctx),
            desc="Blocks", unit="blk",
        ):
            if not block:
                continue

            block_pids: set[bytes] = set()

            for key, val in _parse_entries(block):
                pid = _get_page_id(key)
                if pid is None:
                    continue
                block_pids.add(pid)
                entry = page_buffers.get(pid)
                if entry is None:
                    page_buffers[pid] = [[val], b"toyota" in val.lower()]
                else:
                    entry[0].append(val)
                    if not entry[1] and b"toyota" in val.lower():
                        entry[1] = True

            # Pages absent from this block are complete — flush them.
            if total_pages % 5_000 == 0 and total_pages > 0:
                out_f.flush()

            for pid in [p for p in page_buffers if p not in block_pids]:
                entry = page_buffers.pop(pid)
                total_pages += 1
                if entry[1]:  # has_toyota flag — skip join for non-Toyota pages
                    total_chunks += _flush_page(pid, entry[0], out_f)

            # Memory guard: if too many open pages accumulate, flush oldest.
            if len(page_buffers) > 2_000:
                for pid in list(page_buffers.keys())[:500]:
                    entry = page_buffers.pop(pid)
                    total_pages += 1
                    if entry[1]:
                        total_chunks += _flush_page(pid, entry[0], out_f)

        # Flush any pages still open at end of file.
        for pid, entry in page_buffers.items():
            total_pages += 1
            if entry[1]:
                total_chunks += _flush_page(pid, entry[0], out_f)

    print(f"\n{'='*50}")
    print(f"  Pages scanned  : {total_pages:,}")
    print(f"  Chunks written : {total_chunks:,}")
    print(f"  Output JSONL   : {jsonl_path}")
    print(f"\nNext step (Windows):")
    print(f"  python src/build_vectorstore.py")


# ── Debug footer ──────────────────────────────────────────────────────────────

def debug_footer(lemon_path: str):
    pages_path = Path(lemon_path) / "pages.mtbl"
    size       = pages_path.stat().st_size
    print(f"File : {pages_path}")
    print(f"Size : {size:,} bytes ({size/1e9:.2f} GB)\n")

    with open(pages_path, "rb") as f:
        f.seek(size - MTBL_FOOTER_SIZE)
        raw = f.read(MTBL_FOOTER_SIZE)

    fields = struct.unpack_from("<8Q", raw, 0)
    names  = [
        "index_block_offset", "data_block_size", "compression_algorithm",
        "key_bytes", "val_bytes", "field_5", "index_bytes", "count",
    ]
    for name, val in zip(names, fields):
        print(f"  {name:<26}: {val:,}")

    magic = struct.unpack_from("<I", raw, 508)[0]
    print(f"\n  magic @ offset 508: {magic:#010x}  "
          f"{'✓ VALID' if magic == MTBL_MAGIC else '✗ BAD'}")

    # First 20 raw bytes of block 0 for sanity check
    import zstandard
    with open(pages_path, "rb") as f:
        header = f.read(9)
    cs, he = _read_varint(header, 0)
    print(f"\n  Block 0: varint={cs} ({he} bytes), data starts at offset {he+4}")
    with open(pages_path, "rb") as f:
        f.seek(he + 4)
        sample = f.read(8)
    zstd_magic = 0xFD2FB528
    got_magic  = struct.unpack_from("<I", sample, 0)[0] if len(sample) >= 4 else 0
    print(f"  Block 0 data magic: {got_magic:#010x}  "
          f"{'✓ zstd' if got_magic == zstd_magic else '✗ NOT zstd'}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Extract in-scope Toyota pages from the LEMON MTBL database."
    )
    ap.add_argument("--lemon-path", required=True,
                    help="Path to lemon/ folder (contains pages.mtbl)")
    ap.add_argument("--output-dir",
                    default=os.path.join(os.path.dirname(__file__), "..", "Data", "lemon"),
                    help="Output directory (default: Data/lemon/)")
    ap.add_argument("--debug", action="store_true",
                    help="Print footer fields and block-0 sanity check, then exit")
    args = ap.parse_args()

    if args.debug:
        debug_footer(args.lemon_path)
    else:
        extract(args.lemon_path, args.output_dir)


if __name__ == "__main__":
    main()
