"""
Build output/prompts.csv, the filename -> prompt table that the three
text-conditioned scorers (PickScore, ImageReward, HPS-v2) read at scoring time.
Its inputs, the per-subculture prompt text files from the original collection,
are not released; data/prompts/prompts_with_flags.csv carries the result.

Why this exists
---------------
The Lexica collection script (not released) saved per-image sidecars
`image_<N>_prompt.txt` next to each `image_<N>.jpg`, but those sidecars are no
longer on disk. What survived is `data/prompts/combined_prompts_<sub>_*.txt`,
one per subculture, listing UUID + prompt + byte-size + timestamp per record.
The image files are `image_1.jpg ... image_N.jpg` with no preserved UUID
mapping, so we have to reconstruct filename<->prompt.

Matching strategy
-----------------
For each subculture independently:

1. Parse the combined-prompts TXT into records (uuid, prompt, bytes).
2. For each `image_<N>.jpg` on disk, read its actual byte size.
3. Match by byte size. Where a size is unique on both sides we get a clean
   1:1 mapping. Where a size collides (multiple images with the same byte
   count, or multiple TXT records with the same Dimensions value), fall back
   to ordinal alignment within the colliding group: sort the colliding
   files by numeric N, sort the colliding TXT records by their TXT order,
   and zip them.
4. Report any unmatched filenames or unmatched TXT records.

This is conservative: if any subculture has uncovered images, the script exits
non-zero and refuses to write a partial CSV. The peer-scorer runs require full
coverage to be valid.

Output
------
data/prompts/prompts.csv with columns:
    filename, subculture, prompt, image_uuid, dimensions_bytes, source_txt

Run
---
    python3 dataset/parse_prompts.py
    python3 dataset/parse_prompts.py --strict-size  (no ordinal fallback; fail on any size collision)
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


import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO_ROOT = DATASET
META_PATH = REPO_ROOT / "data" / "clip_metadata.csv"
PROMPTS_DIR = REPO_ROOT / "data" / "prompts"
IMAGES_DIR = REPO_ROOT / "data" / "images"
OUT_CSV = OUTPUT / "prompts.csv"

# Lowercase canonical-folder name -> uppercase-or-mixed-case TXT token used by the scraper.
TXT_FILE_GLOBS = {
    "cottagecore":   "combined_prompts_cottagecore_*.txt",
    "cyberpunk":     "combined_prompts_cyberpunk_*.txt",
    "dark_academia": "combined_prompts_dark_academia_*.txt",
    "goblincore":    "combined_prompts_goblincore_*.txt",
    "synthwave":     "combined_prompts_synthwave_*.txt",
    "y2k":           "combined_prompts_Y2K_*.txt",
}


# --------------------------------------------------------------------------- #

def parse_combined_txt(path: Path) -> list[dict]:
    """Parse a combined_prompts_*.txt into ordered records.

    Each record is delimited by a long '=' line. Within a record we read:
      Image ID: <uuid>            (header)
      Prompt: <text>              (sometimes spans multiple lines until next 'Field: ' line)
      Dimensions: <int> bytes
      URL: ...
      Timestamp: ...
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    # Split on the equals-line separator. The first chunk is the file header.
    chunks = re.split(r"\n=+\n", text)
    records = []
    for chunk in chunks[1:]:                      # skip the file-level header
        chunk = chunk.strip()
        if not chunk:
            continue
        m_uuid = re.search(r"^Image ID:\s*([0-9a-f-]{36})\s*$", chunk, re.MULTILINE)
        m_prompt = re.search(r"^Prompt:\s*(.+?)(?=\nDimensions:|\nURL:|\nTimestamp:|\Z)",
                             chunk, re.DOTALL | re.MULTILINE)
        m_dim = re.search(r"^Dimensions:\s*(\d+)\s*bytes", chunk, re.MULTILINE)
        if not (m_uuid and m_prompt and m_dim):
            # Skip malformed entries silently — we'll catch the gap in coverage check.
            continue
        records.append({
            "uuid":   m_uuid.group(1),
            "prompt": m_prompt.group(1).strip().replace("\n", " "),
            "bytes":  int(m_dim.group(1)),
        })
    return records


def numeric_n(filename: str) -> int:
    """Extract N from 'image_N.jpg' for stable numeric sorting."""
    m = re.match(r"image_(\d+)\.jpg$", filename)
    return int(m.group(1)) if m else -1


def match_subculture(
    folder: Path,
    txt_records: list[dict],
    strict_size: bool,
) -> tuple[list[dict], list[str], list[dict]]:
    """Return (matched_rows, unmatched_filenames, unmatched_txt_records)."""
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() == ".jpg"],
        key=lambda p: numeric_n(p.name),
    )
    file_size = {p.name: p.stat().st_size for p in files}

    # Group both sides by byte size.
    by_size_files: dict[int, list[str]] = defaultdict(list)
    for fname, sz in file_size.items():
        by_size_files[sz].append(fname)
    by_size_txt: dict[int, list[dict]] = defaultdict(list)
    for rec in txt_records:
        by_size_txt[rec["bytes"]].append(rec)

    matched: dict[str, dict] = {}        # filename -> txt record
    used_txt: set[int] = set()           # indices into txt_records that have been claimed

    sizes = set(by_size_files) | set(by_size_txt)
    for sz in sizes:
        fs = sorted(by_size_files.get(sz, []), key=numeric_n)
        ts = list(by_size_txt.get(sz, []))   # preserves TXT order within a size group
        if not fs or not ts:
            continue
        if len(fs) == 1 and len(ts) == 1:
            matched[fs[0]] = ts[0]
            used_txt.add(id(ts[0]))
            continue
        # Size collision (typically 2-5 entries; usually small images that hit a default
        # CDN-resize cap and end up exactly equal in bytes).
        if strict_size:
            continue                      # leave unmatched; will surface in the report
        # Ordinal fallback: zip in deterministic per-side order.
        for f, t in zip(fs, ts):
            matched[f] = t
            used_txt.add(id(t))

    unmatched_files = [p.name for p in files if p.name not in matched]
    unmatched_txt   = [r for r in txt_records if id(r) not in used_txt]

    rows = []
    for fname in [p.name for p in files]:
        rec = matched.get(fname)
        if rec is None:
            continue
        rows.append({
            "filename":         fname,
            "prompt":           rec["prompt"],
            "image_uuid":       rec["uuid"],
            "dimensions_bytes": rec["bytes"],
        })
    return rows, unmatched_files, unmatched_txt


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruct filename->prompt mapping for SubCulture-2.6K.")
    ap.add_argument("--strict-size", action="store_true",
                    help="Refuse the ordinal fallback when byte-sizes collide. Use to inspect collisions.")
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    metadata = pd.read_csv(META_PATH)
    print(f"[info] canonical image list: {len(metadata)} rows from {META_PATH}")

    # Map subculture -> (folder, txt_path)
    subcultures = sorted(metadata["subculture"].unique())
    expected_counts = metadata["subculture"].value_counts().to_dict()

    all_rows = []
    summary_lines = []
    any_gap = False

    for sub in subcultures:
        folder = IMAGES_DIR / sub
        glob = TXT_FILE_GLOBS.get(sub)
        if glob is None:
            print(f"[FAIL] no TXT glob registered for subculture '{sub}'", file=sys.stderr)
            sys.exit(2)
        candidates = sorted(PROMPTS_DIR.glob(glob))
        if not candidates:
            print(f"[FAIL] no TXT file matched {glob}", file=sys.stderr)
            sys.exit(2)
        if len(candidates) > 1:
            print(f"[warn] multiple TXT files for {sub}; using newest: {candidates[-1].name}", file=sys.stderr)
        txt_path = candidates[-1]

        records = parse_combined_txt(txt_path)
        rows, unmatched_files, unmatched_txt = match_subculture(folder, records, args.strict_size)

        for r in rows:
            r["subculture"] = sub
            r["source_txt"] = txt_path.name
        all_rows.extend(rows)

        line = (f"  {sub:>14s}  expected={expected_counts.get(sub,0):>4d}  "
                f"txt_records={len(records):>4d}  matched={len(rows):>4d}  "
                f"unmatched_files={len(unmatched_files):>3d}  unmatched_txt={len(unmatched_txt):>3d}")
        summary_lines.append(line)
        if unmatched_files or unmatched_txt:
            any_gap = True
            print(line, file=sys.stderr)
            if unmatched_files[:5]:
                print(f"      sample unmatched files: {unmatched_files[:5]}", file=sys.stderr)
            if unmatched_txt[:3]:
                print(f"      sample unmatched txt UUIDs: {[r['uuid'] for r in unmatched_txt[:3]]}", file=sys.stderr)

    print("\n[summary]")
    for line in summary_lines:
        print(line)

    if any_gap:
        print("\n[FAIL] some images or TXT records went unmatched. Refusing to write a partial CSV.", file=sys.stderr)
        print("       Inspect with --strict-size to see exactly which size groups collide.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows, columns=["filename", "subculture", "prompt", "image_uuid",
                                         "dimensions_bytes", "source_txt"])

    # Sanity: every row in clip_metadata must appear in the parsed mapping.
    canonical = set(zip(metadata["filename"], metadata["subculture"]))
    parsed = set(zip(df["filename"], df["subculture"]))
    missing = canonical - parsed
    extra = parsed - canonical
    if missing or extra:
        print(f"[FAIL] coverage mismatch vs clip_metadata.csv: missing={len(missing)} extra={len(extra)}",
              file=sys.stderr)
        if missing: print(f"   sample missing: {list(missing)[:5]}", file=sys.stderr)
        if extra:   print(f"   sample extra:   {list(extra)[:5]}", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"\n[done] wrote {args.out}  ({len(df)} rows, {df['prompt'].str.len().mean():.0f} chars/prompt avg)")
    print("       columns: filename, subculture, prompt, image_uuid, dimensions_bytes, source_txt")


if __name__ == "__main__":
    main()
