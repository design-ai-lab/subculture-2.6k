"""
Photograph control — apply the author's curation decisions from photo_decisions.csv
(downloaded from review_sheet.html; default location ~/Downloads).

  keep     -> move candidates/<class>/<file> to images/<class>/<file>,
              provenance row appended to sources.csv
  discard  -> delete from candidates/

Usage:
    python apply_decisions.py [path/to/photo_decisions.csv]
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
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

dec_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "photo_decisions.csv"
if not dec_path.exists():
    sys.exit(f"[FAIL] decisions file not found: {dec_path}")

prov = {}
for csv_name in ("candidates_sources.csv", "candidates_sources_openverse.csv"):
    p = HERE / csv_name
    if p.exists():
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                prov[r["filename"]] = r

kept_rows, n_keep, n_discard, n_missing = [], 0, 0, 0
with open(dec_path, newline="") as f:
    for r in csv.DictReader(f):
        src = HERE / "candidates" / r["class"] / r["filename"]
        if not src.exists():
            n_missing += 1
            continue
        if r["decision"] == "discard":
            src.unlink()
            n_discard += 1
        else:
            dst_dir = HERE / "images" / r["class"]
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dst_dir / r["filename"])
            n_keep += 1
            pr = prov.get(r["filename"], {"filename": r["filename"], "class": r["class"]})
            kept_rows.append({
                "filename": pr.get("filename", r["filename"]),
                "class": pr.get("class", r["class"]),
                "source_url": pr.get("source_url", ""),
                "license": pr.get("license", ""),
                "date_evidence": pr.get("date_evidence", pr.get("uploaded", "")),
                "query": pr.get("query", ""),
            })

with open(HERE / "sources.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["filename", "class", "source_url",
                                      "license", "date_evidence", "query"])
    w.writeheader()
    w.writerows(kept_rows)

per_class = {}
for r in kept_rows:
    per_class[r["class"]] = per_class.get(r["class"], 0) + 1
print(f"[done] kept {n_keep} -> images/  ({per_class})   discarded {n_discard}"
      + (f"   missing {n_missing}" if n_missing else ""))
print(f"[done] sources.csv rewritten with {len(kept_rows)} curated rows")
