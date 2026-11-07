#!/usr/bin/env python3
"""Check that a local copy of the SubCulture-2.6K dataset is complete (folders, file counts, columns)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "paths.py").is_file():
        sys.path.insert(0, str(_p))
        break
from paths import DATASET


def main() -> None:
    root = DATASET
    errors = []

    def need(rel: str, kind: str) -> Path:
        path = root / rel
        if kind == "dir" and not path.is_dir():
            errors.append(f"missing directory {rel}")
        elif kind == "file" and not path.is_file():
            errors.append(f"missing file {rel}")
        return path

    for name in ("data", "matched_text", "near_duplicates", "label_check", "photographs", "cross_generator", "bootstrap"):
        need(name, "dir")
    need("croissant.json", "file")
    need("data/clip_metadata.csv", "file")
    need("data/clip_embeddings.npy", "file")
    need("data/interpretable_features.csv", "file")
    need("data/prompts/prompts_with_flags.csv", "file")
    for s in ("laion_v2", "pickscore", "image_reward", "hps_v2"):
        need(f"data/scores/scores_{s}.csv", "file")
    need("near_duplicates/near_duplicate_pairs.csv", "file")
    need("label_check/label_audit_sample.csv", "file")
    need("label_check/labels_coder1.csv", "file")
    need("label_check/labels_coder2.csv", "file")
    need("photographs/photographs.csv", "file")
    need("cross_generator/prompts.csv", "file")
    need("cross_generator/scores.csv", "file")
    need("cross_generator/generation_status.csv", "file")

    images = list((root / "data" / "images").rglob("*.jpg")) if (root / "data" / "images").is_dir() else []
    gen_png = list((root / "cross_generator").rglob("*.png")) if (root / "cross_generator").is_dir() else []
    matched = list((root / "matched_text").glob("scores_*.csv")) if (root / "matched_text").is_dir() else []
    if len(images) != 2610:
        errors.append(f"corpus jpeg count {len(images)} != 2610")
    if len(gen_png) != 182:
        errors.append(f"generator png count {len(gen_png)} != 182")
    if len(matched) != 9:
        errors.append(f"matched-text score files {len(matched)} != 9")

    photo = root / "photographs" / "photographs.csv"
    if photo.is_file():
        header = photo.read_text().splitlines()[0].split(",")
        expected = ["filename", "style", "source_url", "license", "provider", "date_evidence", "query", "laion_score"]
        if header != expected:
            errors.append(f"photograph columns {header}")

    croissant = root / "croissant.json"
    if croissant.is_file():
        version = json.loads(croissant.read_text()).get("version")
        if version != "1.1.0":
            errors.append(f"croissant version {version} != 1.1.0")

    if errors:
        print(f"{root}")
        for item in errors:
            print("FAIL", item)
        sys.exit(1)
    print(f"OK {root}")
    print(f"jpeg {len(images)}  generator png {len(gen_png)}  matched-text score files {len(matched)}")


if __name__ == "__main__":
    main()
