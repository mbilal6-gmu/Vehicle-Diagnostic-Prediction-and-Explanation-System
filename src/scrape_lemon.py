"""
Scrape TSB and DTC pages from the running LEMON server (http://127.0.0.1:8080).
Writes Data/lemon/toyota_pages.jsonl with text chunks for vector-store ingestion.

Run after starting lemon-website.exe with lemon/index.json.
"""
import json
import re
import hashlib
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, unquote

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://127.0.0.1:8080"

TARGET_MODELS = {
    "camry", "corolla", "rav4", "highlander", "prius",
    "tacoma", "4runner", "sienna", "yaris", "avalon",
    "tundra", "sequoia", "land cruiser", "venza",
    "prius c", "prius v", "prius prime", "c-hr",
}
TARGET_YEARS = {2020}

# Sections within each vehicle's Repair and Diagnosis tree we want
# Sections that have a real index page — BFS starts from that URL
SECTION_PATHS = [
    "Repair%20and%20Diagnosis/Quick%20Lookups/Technical%20Bulletins/Technical%20Service%20Bulletins/",
    "Repair%20and%20Diagnosis/Quick%20Lookups/Technical%20Bulletins/Safety%20Recalls/",
    "Repair%20and%20Diagnosis/Quick%20Lookups/DTC%20Index/",
]

# Full repair categories — no index page; seeds are harvested from the R&D parent page
# Values are the URL-decoded category names as they appear as href prefixes
REPAIR_CATEGORIES = [
    "Accessories%20%26%20Equipment",
    "Engine%20Performance",
    "Electrical",
    "Brakes",
    "Engine%20Mechanical",
    "Transmission",
    "Heating%2C%20Ventilation%20%26%20Air%20Conditioning",
    "Restraints",
]

CHUNK_WORDS  = 300
CHUNK_OVERLAP = 50
MIN_CHUNK_WORDS = 40

_DTC_RE  = re.compile(r'\b([PBCU][0-9]{4})\b', re.IGNORECASE)
_TSB_RE  = re.compile(r'\bT-SB-\d{3,4}-\d{2,4}(?:\s*REV\s*\d+)?\b', re.IGNORECASE)


def get(url: str, retries: int = 3):
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.ok:
                return r.text
            return None
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return None


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("div", class_="main") or soup.find("body")
    if not main:
        return ""
    for tag in main.find_all(["script", "style", "nav"]):
        tag.decompose()
    text = main.get_text(separator="\n")
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def child_links(html: str, base_url: str) -> list[str]:
    """Return absolute URLs of links that go deeper (relative, non-parent)."""
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("..") or href.startswith("/") or ":" in href:
            continue
        full = base_url.rstrip("/") + "/" + href.lstrip("./")
        links.append(full)
    return links


def chunk_text(text: str) -> list[str]:
    words = text.split()
    step = CHUNK_WORDS - CHUNK_OVERLAP
    chunks = []
    for i in range(0, max(1, len(words)), step):
        chunk = " ".join(words[i : i + CHUNK_WORDS])
        if len(chunk.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk)
    return chunks


MAX_PAGES_PER_SECTION = 250  # raised — full repair sections can have 177+ pages


def _harvest_category_seeds(vehicle_url: str, category_prefix: str) -> list[str]:
    """
    Fetch the Repair & Diagnosis parent page and return all href URLs
    whose relative path starts with `category_prefix`.
    Used for repair categories that have no index page of their own.
    """
    rd_url = vehicle_url.rstrip("/") + "/Repair%20and%20Diagnosis/"
    html = get(rd_url)
    if not html:
        return []
    soup  = BeautifulSoup(html, "html.parser")
    seeds = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(category_prefix + "/"):
            seeds.append(rd_url.rstrip("/") + "/" + href)
    return seeds


def scrape_section(vehicle_url: str, section_path: str,
                   model: str, year: int,
                   seen: set, out,
                   seed_urls: list = None) -> int:
    """BFS crawl a section, up to MAX_PAGES_PER_SECTION content pages.

    If seed_urls is provided the BFS starts from those URLs directly
    (used for repair categories with no index page).
    Returns number of chunks written.
    """
    if seed_urls is not None:
        if not seed_urls:
            return 0
        queue = list(seed_urls)
        html  = None   # fetched per-URL inside the loop
    else:
        section_url = vehicle_url.rstrip("/") + "/" + section_path
        html = get(section_url)
        if not html or "Page Not Found" in html:
            return 0
        queue = [section_url]

    visited: set[str] = set()
    content_pages = 0
    written = 0

    # For the index-page case, pre-load html for the first URL
    _first_url = queue[0] if queue else None
    _first_html = html  # may be None (seed-list mode) or pre-fetched

    while queue and content_pages < MAX_PAGES_PER_SECTION:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        if url == _first_url and _first_html is not None:
            html = _first_html
        else:
            html = get(url)
            if not html:
                continue

        if "Page Not Found" in html:
            continue

        text = extract_text(html)

        if len(text.split()) < MIN_CHUNK_WORDS:
            # Navigation page — enqueue children
            for link in child_links(html, url):
                if link not in visited:
                    queue.append(link)
            continue

        content_pages += 1
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in seen:
            # Still enqueue children (same page shared across vehicles)
            for link in child_links(html, url):
                if link not in visited:
                    queue.append(link)
            continue
        seen.add(content_hash)

        # Extract page title from breadcrumb (last part = page name)
        soup = BeautifulSoup(html, "html.parser")
        bc = soup.find_all("a", class_="breadcrumb-part")
        title = bc[-1].text.strip() if bc else ""

        # Build a meaningful document prefix: vehicle + page title
        prefix = f"Toyota {model.title()} {year}"
        if title:
            prefix += f" — {title}"

        # Extract all unique DTCs and TSB refs from full page text
        all_dtcs = list(dict.fromkeys(c.upper() for c in _DTC_RE.findall(text)))[:8]
        all_tsbs = list(dict.fromkeys(t.upper() for t in _TSB_RE.findall(text)))[:5]

        for i, chunk in enumerate(chunk_text(text)):
            dtcs = _DTC_RE.findall(chunk)
            doc_id = "lemon_" + hashlib.md5(f"{content_hash}_{i}".encode()).hexdigest()[:16]
            out.write(json.dumps({
                "id": doc_id,
                "document": f"{prefix}\n\n{chunk}",
                "metadata": {
                    "dtc_code":    dtcs[0].upper() if dtcs else "",
                    "all_dtcs":    ",".join(all_dtcs),
                    "tsb_refs":    ",".join(all_tsbs),
                    "model_year":  str(year),
                    "engine_code": "",
                    "description": (title or text.split("\n")[0])[:120],
                    "source":      "LEMON Vehicle Manual Database",
                    "source_url":  url,
                    "image_keys":  "",
                },
            }) + "\n")
            written += 1

        # Enqueue child pages
        for link in child_links(html, url):
            if link not in visited:
                queue.append(link)

    return written


def main():
    index_path = Path(r"C:\Users\Admin\Documents\Claude\Projects\AIDesign\Data\lemon\full_lemon_app\lemon\index.json")
    out_dir    = Path(r"C:\Users\Admin\Documents\Claude\Projects\AIDesign\Data\lemon")
    out_path   = out_dir / "toyota_pages.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading index.json …")
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    # Pick one representative vehicle per (year, base_model) combination.
    # Sort by model name so we get consistent "first" picks.
    # Sort longer model names first so "prius c" matches before "prius"
    sorted_models = sorted(TARGET_MODELS, key=len, reverse=True)
    by_key = {}
    for v in sorted(index["vehicles"], key=lambda x: x["model"]):
        if v["make"] != "Toyota":
            continue
        model_lower = v["model"].lower()
        base = next((m for m in sorted_models if m in model_lower), None)
        if base is None:
            continue
        for yr_str in v["years"]:
            try:
                yr = int(yr_str)
            except ValueError:
                continue
            if yr not in TARGET_YEARS:
                continue
            key = (yr, base)
            if key not in by_key:
                by_key[key] = v

    vehicles = sorted(by_key.items(), key=lambda x: x[0])
    print(f"Vehicles selected: {len(vehicles)}  (one variant per model/year)")

    seen: set[str] = set()
    total = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for (year, base_model), vehicle in vehicles:
            uri_path = vehicle["uriPath"]   # e.g. "/Toyota/2018/Camry%20LE%202.5L/"
            vehicle_url = BASE_URL + uri_path
            v_chunks = 0

            for section_path in SECTION_PATHS:
                n = scrape_section(vehicle_url, section_path,
                                   base_model, year, seen, out_f)
                v_chunks += n

            for category in REPAIR_CATEGORIES:
                seeds = _harvest_category_seeds(vehicle_url, category)
                n = scrape_section(vehicle_url, "", base_model, year,
                                   seen, out_f, seed_urls=seeds)
                v_chunks += n

            total += v_chunks
            print(f"  {year} {base_model:12s}  variant={unquote(uri_path.split('/')[3][:30]):30s}  +{v_chunks} chunks  total={total}",
                  flush=True)
            out_f.flush()

    print(f"\nDone — {total} chunks written to {out_path}")
    print("\nNext step:")
    print("  python src/build_vectorstore.py")


if __name__ == "__main__":
    main()
