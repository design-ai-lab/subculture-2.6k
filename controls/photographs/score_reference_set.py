"""
Real-photograph reference set (Section 5.4): membership pre-rank and LAION-A scoring.

Does the Y2K penalty persist on real photographs that no Stable Diffusion or
LAION-filter pipeline touched? LAION-A is vision-only, so no text input is involved.
The pipeline matches the corpus: CLIP ViT-L-14 ("openai") via open_clip and the
LAION-A MLP head from models/, applied to L2-normalized embeddings.

Photographs are released as URLs only (photographs/photographs.csv). Download them
into controls/photographs/images/{y2k,cyberpunk}/ before running --stage score.

Stages:
  --stage curate   embed candidates/{y2k,cyberpunk}, train the paper's random forest
                   (300 trees, seed 42) on the 2,610 corpus embeddings, and write
                   curation_ranks.csv with the predicted subculture and the
                   probability of the intended class for each candidate.
  --stage score    embed images/{y2k,cyberpunk}, score with LAION-A, and report
                   per-class statistics and the gap against the SD corpus.

Outputs go to output/.
"""

from __future__ import annotations


import sys as _sys
from pathlib import Path as _Path
for _p in _Path(__file__).resolve().parents:
    if (_p / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from paths import DATASET, CODE_ROOT, OUTPUT, under, laion_head
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
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = DATASET
SEED = 42
CLASSES = ["y2k", "cyberpunk"]


def embed_folder(root: Path, device: str) -> tuple[list[str], list[str], np.ndarray]:
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval()

    names, classes, feats = [], [], []
    for cls in CLASSES:
        files = sorted((root / cls).glob("*.jpg"))
        for i in range(0, len(files), 16):
            batch_files = files[i:i + 16]
            tensors = [preprocess(Image.open(f).convert("RGB")) for f in batch_files]
            with torch.inference_mode():
                out = model.encode_image(torch.stack(tensors).to(device))
            feats.append(out.float().cpu().numpy())
            names += [f.name for f in batch_files]
            classes += [cls] * len(batch_files)
        print(f"  embedded {cls}: {len(files)} images", flush=True)
    if not names:
        raise SystemExit(f"[FAIL] no images under {root}")
    return names, classes, np.concatenate(feats, axis=0).astype(np.float32)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def pick_device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else \
           ("cuda" if torch.cuda.is_available() else "cpu")


def stage_curate(device: str) -> None:
    from sklearn.ensemble import RandomForestClassifier
    corpus = l2(np.load(REPO_ROOT / "data" / "clip_embeddings.npy").astype(np.float32))
    y = pd.read_csv(REPO_ROOT / "data" / "clip_metadata.csv")["subculture"].values
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(corpus, y)

    names, classes, feats = embed_folder(HERE / "candidates", device)
    proba = rf.predict_proba(l2(feats))
    pred = rf.classes_[proba.argmax(axis=1)]
    p_intended = [proba[i, list(rf.classes_).index(c)] for i, c in enumerate(classes)]

    out = pd.DataFrame({"filename": names, "intended_class": classes,
                        "predicted": pred, "p_intended": np.round(p_intended, 3),
                        "match": pred == np.array(classes)})
    out = out.sort_values(["intended_class", "p_intended"], ascending=[True, False])
    OUTPUT.mkdir(exist_ok=True)
    out.to_csv(OUTPUT / "curation_ranks.csv", index=False)
    for cls in CLASSES:
        sub = out[out.intended_class == cls]
        print(f"[{cls}] {len(sub)} candidates | classifier agrees: {sub.match.sum()} "
              f"| p_intended>=0.5: {(sub.p_intended >= 0.5).sum()}")
    print("[done] output/curation_ranks.csv — sort by p_intended; low rows first to discard")


def stage_score(device: str) -> None:
    import torch
    import torch.nn as nn
    from scipy import stats

    names, classes, feats = embed_folder(HERE / "images", device)
    x = l2(feats)

    class AestheticMLP(nn.Module):
        def __init__(self, input_size: int = 768) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 1024), nn.Dropout(0.2),
                nn.Linear(1024, 128),        nn.Dropout(0.2),
                nn.Linear(128, 64),          nn.Dropout(0.2),
                nn.Linear(64, 16),
                nn.Linear(16, 1))
        def forward(self, t): return self.layers(t)

    head = AestheticMLP(768)
    state = torch.load(laion_head(),
                       map_location="cpu", weights_only=True)
    head.load_state_dict(state)
    head.eval()
    with torch.inference_mode():
        scores = head(torch.from_numpy(x)).squeeze(-1).numpy()

    df = pd.DataFrame({"filename": names, "class": classes, "laion_score": scores})
    OUTPUT.mkdir(exist_ok=True)
    df.to_csv(OUTPUT / "photo_scores.csv", index=False)

    yk = df[df["class"] == "y2k"].laion_score
    cy = df[df["class"] == "cyberpunk"].laion_score
    u, p = stats.mannwhitneyu(cy, yk, alternative="greater")
    d = (cy.mean() - yk.mean()) / np.sqrt((cy.var() + yk.var()) / 2)

    # SD-corpus reference gap (released scores)
    sd = pd.read_csv(REPO_ROOT / "data" / "scores" / "scores_laion_v2.csv")
    sd_cy, sd_yk = (sd[sd.subculture == s].score for s in ("cyberpunk", "y2k"))
    sd_d = (sd_cy.mean() - sd_yk.mean()) / np.sqrt((sd_cy.var() + sd_yk.var()) / 2)

    report = "\n".join([
        "Real-photograph reference set",
        f"n: cyberpunk {len(cy)}, y2k {len(yk)} (real photographs, pre-2021 uploads)",
        "",
        f"REAL PHOTOS:  cyberpunk mean {cy.mean():.3f} (sd {cy.std():.3f})   "
        f"y2k mean {yk.mean():.3f} (sd {yk.std():.3f})",
        f"  gap {cy.mean() - yk.mean():+.3f}   Cohen's d {d:.2f}   "
        f"Mann-Whitney (cy>y2k) p {p:.3g}",
        f"  y2k fraction >= 6.5 cutoff: {(yk >= 6.5).mean():.3f}   "
        f"cyberpunk: {(cy >= 6.5).mean():.3f}",
        "",
        f"SD CORPUS (paper): cyberpunk mean {sd_cy.mean():.3f}   y2k mean {sd_yk.mean():.3f}   "
        f"gap {sd_cy.mean() - sd_yk.mean():+.3f}   Cohen's d {sd_d:.2f}",
        "",
        "Reading: a persistent gap on real photos is evidence the penalty follows",
        "the STYLE, not the generator; a vanished gap supports the generator story.",
        "",
    ])
    (OUTPUT / "photo_report.txt").write_text(report)
    print(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("curate", "score"))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or pick_device()
    print(f"[info] stage={args.stage} device={device}")
    (stage_curate if args.stage == "curate" else stage_score)(device)


if __name__ == "__main__":
    main()
