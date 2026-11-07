"""
Photograph candidate harvester — Openverse route (images served from Flickr's CDN).

Date guarantee without an upload-date field: Flickr assigns photo IDs
monotonically in upload order, so a photo ID below FLICKR_ID_CUTOFF
(50,000,000,000 ~ uploads before ~Oct 2020) predates mainstream AI image
generation. We accept only Flickr-sourced results passing that cutoff, or any
result Openverse indexed before 2021 (indexing postdates upload).

Licenses: CC0 / CC-BY / CC-BY-SA (Openverse `license` filter).
Files: ov_{class}_{NNN}.jpg under candidates/; provenance in
candidates_sources_openverse.csv. The review sheet is regenerated from BOTH
provenance CSVs (Commons v2 + this one) so curation sees everything.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in _Path(__file__).resolve().parents:
    if (_p / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from paths import DATASET, CODE_ROOT, OUTPUT, under
for _d in (
    CODE_ROOT / "results" / "laion_profile",
    CODE_ROOT / "results" / "separability",
    CODE_ROOT / "results" / "matched_text",
    CODE_ROOT / "dataset",
    CODE_ROOT / "scoring",
):
    if _d.is_dir() and str(_d) not in _sys.path:
        _sys.path.insert(0, str(_d))


import csv
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image

HERE = Path(__file__).resolve().parent
TARGET_PER_CLASS = 150
FLICKR_ID_CUTOFF = 50_000_000_000     # ~Oct 2020
INDEXED_CUTOFF = "2021-01-01"
HEADERS = {"User-Agent": "SubCulture-2.6K-audit/1.0 (academic; https://github.com/design-ai-lab/subculture-2.6k)"}
API = "https://api.openverse.org/v1/images/"

QUERIES = {
    # short strict-AND queries; the Flickr-ID date filter does the sieving
    "cyberpunk": [
        "neon sign night",
        "neon street",
        "neon tokyo",
        "shinjuku night",
        "neon hong kong",
        "kowloon night",
        "night market hong kong",
        "dotonbori",
        "las vegas neon",
        "neon sign",
        "night street rain",
        "akihabara night",
        "neon lights",
        "chinatown night",
        "taipei night",
        "seoul night",
        "bangkok night",
        "osaka night",
        "shibuya crossing night",
        "times square night",
        "night alley",
        "arcade neon",
    ],
    "y2k": [
        "windows xp",
        "gameboy",
        "walkman",
        "nokia 3310",
        "nokia phone",
        "imac g3",
        "flip phone",
        "playstation 2",
        "crt monitor",
        "ipod nano",
        "motorola razr",
        "internet cafe",
        "sony ericsson",
        "discman",
        "mp3 player",
    ],
}


def flickr_id(rec: dict) -> int | None:
    m = re.search(r"/photos/[^/]+/(\d+)", rec.get("foreign_landing_url") or "")
    return int(m.group(1)) if m else None


def usable(rec: dict) -> tuple[bool, str]:
    if not rec.get("url") or not rec.get("license"):
        return False, ""
    idx = (rec.get("indexed_on") or "")[:10]
    if idx and idx < INDEXED_CUTOFF:
        return True, f"openverse indexed {idx}"
    if rec.get("source") == "flickr":
        fid = flickr_id(rec)
        if fid and fid < FLICKR_ID_CUTOFF:
            return True, f"flickr id {fid} < 50e9 (~pre-2020-10)"
    return False, ""


def search(query: str, page: int) -> list[dict]:
    for attempt in range(4):
        r = requests.get(API, params={
            "q": query, "license": "by,by-sa,cc0", "category": "photograph",
            "page_size": 20, "page": page}, headers=HEADERS, timeout=25)
        if r.status_code == 429:
            wait = max(int(r.headers.get("Retry-After", 0)), 30)
            print(f"  [429-api] backing off {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code != 200:
            print(f"  [warn] {query!r} p{page}: HTTP {r.status_code}")
            return []
        return r.json().get("results", [])
    return []


def download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=40)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        if min(img.size) < 256:
            return False
        img.thumbnail((1024, 1024))
        img.save(dest, "JPEG", quality=92)
        return True
    except Exception as e:
        print(f"  [skip] {url[:60]}: {type(e).__name__}")
        return False


def rebuild_review_sheet() -> None:
    rows = []
    for csv_name in ("candidates_sources.csv", "candidates_sources_openverse.csv"):
        p = HERE / csv_name
        if p.exists():
            with open(p, newline="") as f:
                rows += [r for r in csv.DictReader(f)
                         if (HERE / "candidates" / r["class"] / r["filename"]).exists()]
    cells = "".join(
        f'<figure><img src="candidates/{r["class"]}/{r["filename"]}" loading="lazy">'
        f'<figcaption>{r["filename"]}<br>{r.get("license", "")}</figcaption></figure>'
        for r in rows)
    (HERE / "review_sheet.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>Photograph candidate review</title>"
        "<style>body{font-family:sans-serif;display:grid;"
        "grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;padding:10px}"
        "figure{margin:0}img{width:100%;height:200px;object-fit:cover}"
        "figcaption{font-size:11px;text-align:center}</style>"
        f"<h1 style='grid-column:1/-1'>Photograph candidates (n={len(rows)}) — DELETE bad "
        "files from candidates/, survivors move to images/</h1>" + cells)
    print(f"[done] review_sheet.html ({len(rows)} candidates, both sources)")


def main() -> None:
    csv_path = HERE / "candidates_sources_openverse.csv"
    rows = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if (HERE / "candidates" / r["class"] / r["filename"]).exists()]
        print(f"[resume] {len(rows)} existing openverse candidates kept")

    for cls, queries in QUERIES.items():
        out_dir = HERE / "candidates" / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = [r for r in rows if r["class"] == cls]
        n = len(existing)
        seen = {r["source_url"] for r in existing}
        for q in queries:
            if n >= TARGET_PER_CLASS:
                break
            for page in range(1, 13):   # deep pagination: old photos live in later pages
                if n >= TARGET_PER_CLASS:
                    break
                recs = search(q, page)
                time.sleep(1.0)
                if not recs:
                    break
                for rec in recs:
                    if n >= TARGET_PER_CLASS:
                        break
                    ok, evidence = usable(rec)
                    landing = rec.get("foreign_landing_url", "")
                    if not ok or landing in seen:
                        continue
                    seen.add(landing)
                    fname = f"ov_{cls}_{n:03d}.jpg"
                    if download(rec["url"], out_dir / fname):
                        rows.append({
                            "filename": fname, "class": cls, "source_url": landing,
                            "license": f'{rec.get("license", "")} {rec.get("license_version", "")}'.strip(),
                            "provider": rec.get("source", ""), "date_evidence": evidence,
                            "query": q})
                        n += 1
                        time.sleep(0.5)
            print(f"[{cls}] after {q!r}: {n} candidates", flush=True)
        print(f"[{cls}] TOTAL: {n}", flush=True)

    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[done] {len(rows)} candidates + {csv_path.name}")
    rebuild_review_sheet()


if __name__ == "__main__":
    main()
