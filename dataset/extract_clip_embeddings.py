"""
Extract CLIP ViT-L/14 embeddings for all SubCulture-2.6K images.

Outputs (released copies in data/):
  output/clip_embeddings.npy   — (N, 768) float32, RAW (un-normalized) embeddings
  output/clip_metadata.csv     — index, filename, subculture, relative_path

Notes:
  - Uses OpenAI CLIP ViT-L/14 weights via open_clip. This is the backbone the
    LAION-Aesthetics predictor expects; do NOT swap to SigLIP/MetaCLIP/etc.
  - Embeddings are saved RAW. L2-normalize at use-time for cosine similarity;
    keep raw for the aesthetic predictor.
  - Row i in the .npy corresponds to row i in the .csv (matched by `index`).
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
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import open_clip


REPO_ROOT = DATASET
IMAGES_DIR = REPO_ROOT / "data" / "images"
OUT_EMBED = OUTPUT / "clip_embeddings.npy"
OUT_META = OUTPUT / "clip_metadata.csv"

SUBCULTURES = ["cottagecore", "cyberpunk", "dark_academia", "goblincore", "synthwave", "y2k"]
MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
BATCH_SIZE = 32
SEED = 42


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_image_paths() -> list[tuple[str, Path]]:
    """Return list of (subculture_label, absolute_path) sorted deterministically."""
    items: list[tuple[str, Path]] = []
    for sub in SUBCULTURES:
        sub_dir = IMAGES_DIR / sub
        if not sub_dir.is_dir():
            sys.exit(f"Missing image folder: {sub_dir}")
        files = sorted(p for p in sub_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            sys.exit(f"No images found in {sub_dir}")
        items.extend((sub, p) for p in files)
    return items


def main() -> None:
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"[info] device = {device}")

    print(f"[info] loading {MODEL_NAME} ({PRETRAINED}) — first run downloads ~890MB to ~/.cache")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device).eval()

    items = collect_image_paths()
    n = len(items)
    print(f"[info] found {n} images across {len(SUBCULTURES)} subcultures")

    embeddings = np.zeros((n, 768), dtype=np.float32)
    metadata: list[dict] = []

    autocast_dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    t0 = time.time()

    with torch.inference_mode():
        for start in range(0, n, BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            tensors = []
            for sub, path in batch:
                with Image.open(path) as img:
                    tensors.append(preprocess(img.convert("RGB")))
            batch_tensor = torch.stack(tensors).to(device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=device.type != "cpu"):
                feats = model.encode_image(batch_tensor)

            feats = feats.float().cpu().numpy()
            embeddings[start : start + len(batch)] = feats

            for offset, (sub, path) in enumerate(batch):
                idx = start + offset
                metadata.append({
                    "index": idx,
                    "filename": path.name,
                    "subculture": sub,
                    "relative_path": str(under(REPO_ROOT, path)),
                })

            done = start + len(batch)
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            eta = (n - done) / max(rate, 1e-6)
            print(f"[progress] {done}/{n}  ({rate:5.1f} img/s, ETA {eta:5.1f}s)")

    OUT_EMBED.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_EMBED, embeddings)
    with OUT_META.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "filename", "subculture", "relative_path"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"[done] wrote {OUT_EMBED}  shape={embeddings.shape}  dtype={embeddings.dtype}")
    print(f"[done] wrote {OUT_META}  rows={len(metadata)}")
    print(f"[done] total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
