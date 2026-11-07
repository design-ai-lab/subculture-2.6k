"""
Label check — blinded manual label-verification kit. Stratified sample of 50 images per subculture (seed 42), copied
under anonymized names so coders never see the platform label.

Produces (under label_audit/):
  sample_images/sample_NNN.jpg      300 anonymized copies
  coding_sheet.csv                  sample_id + blank coder1/coder2 columns
  answer_key.csv                    sample_id -> true (platform) label; DO NOT
                                    open until both coders are done
  contact_sheet.html                all 300 images with their sample_id, in
                                    shuffled order, no label information

Coding protocol (report agreement in the audit): each coder assigns one of
the six subculture labels (or "none/unclear") per image, independently. Then
compute per-coder accuracy vs the platform label and inter-coder Cohen's kappa.
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


import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = DATASET
SEED = 42
PER_SUB = 50

out = HERE / "label_audit"
img_out = out / "sample_images"
img_out.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(REPO_ROOT / "data" / "clip_metadata.csv")
rng = np.random.default_rng(SEED)

sampled = pd.concat([g.sample(PER_SUB, random_state=SEED)
                     for _, g in meta.groupby("subculture")])
sampled = sampled.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
sampled["sample_id"] = [f"sample_{i:03d}" for i in range(len(sampled))]

for _, r in sampled.iterrows():
    shutil.copyfile(REPO_ROOT / r["relative_path"], img_out / f"{r['sample_id']}.jpg")

sampled[["sample_id"]].assign(coder1_label="", coder2_label="", notes="") \
    .to_csv(out / "coding_sheet.csv", index=False)
sampled[["sample_id", "subculture", "filename", "relative_path"]] \
    .to_csv(out / "answer_key.csv", index=False)

cells = "\n".join(
    f'<figure><img src="sample_images/{sid}.jpg" loading="lazy">'
    f"<figcaption>{sid}</figcaption></figure>"
    for sid in sampled["sample_id"])
(out / "contact_sheet.html").write_text(
    "<!doctype html><meta charset='utf-8'><title>Label audit — blinded sample</title>"
    "<style>body{font-family:sans-serif;display:grid;"
    "grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;padding:10px}"
    "figure{margin:0}img{width:100%;height:220px;object-fit:cover}"
    "figcaption{font-size:12px;text-align:center}</style>"
    f"<h1 style='grid-column:1/-1'>Blinded label-audit sample (n={len(sampled)})</h1>"
    + cells)

print(f"[done] {len(sampled)} images -> {img_out}")
print(f"[done] coding_sheet.csv, answer_key.csv, contact_sheet.html in {out}")
print("Open contact_sheet.html to code; do NOT open answer_key.csv until done.")
