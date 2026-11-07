"""Check the 33 CLIP-cosine-1 near-duplicate pairs for byte identity.

Read-only on data/images and the near-duplicate pair list. Writes
output/dedup_byte_identical.txt and refuses to overwrite.
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


import hashlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = DATASET
PAIRS = DATASET / "near_duplicates" / "near_duplicate_pairs.csv"
IMG = DATASET / "data" / "images"
OUT = OUTPUT / "dedup_byte_identical.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    pairs = pd.read_csv(PAIRS)
    exact = pairs[pairs.cosine >= 0.999999]
    same = diff = missing = 0
    lines = [
        "Pairs with cosine >= 0.999999 in dedup_dupe_pairs.csv.",
        "Each file is data/images/<subculture>/<filename> as named in that row.",
        f"exact pairs: {len(exact)}; cross-subculture: {int((exact.sub_a != exact.sub_b).sum())}",
        "",
    ]
    for _, r in exact.iterrows():
        a = IMG / r.sub_a / r.file_a
        b = IMG / r.sub_b / r.file_b
        if not a.is_file() or not b.is_file():
            missing += 1
            lines.append(f"MISSING {r.sub_a}/{r.file_a}  {r.sub_b}/{r.file_b}")
            continue
        ha, hb = sha256(a), sha256(b)
        if ha == hb:
            same += 1
        else:
            diff += 1
            lines.append(f"DIFF {r.sub_a}/{r.file_a} {a.stat().st_size}  {r.sub_b}/{r.file_b} {b.stat().st_size}")
    lines += ["", f"byte-identical: {same}", f"different: {diff}", f"missing: {missing}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-4:]))
    if diff or missing or same != 33:
        raise SystemExit("[FAIL] expected 33 byte-identical pairs")


if __name__ == "__main__":
    main()
