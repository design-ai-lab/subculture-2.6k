"""
Per-backbone separability — embedding extraction for the peer scorers' own representation spaces.

The paper shows 92.4% / 92.7% subculture separability in CLIP ViT-L/14 space
(results/separability/clip_subculture_analysis.py). The peers read other spaces:
PickScore and HPS-v2 a CLIP ViT-H/14 fine-tune, ImageReward a BLIP fine-tune.
This script extracts per-image embeddings from those spaces:

    --backbone vith_pick   PickScore_v1's own fine-tuned ViT-H/14 image tower
                           (run in the subc-pick environment). Also the closest available
                           proxy for HPS-v2, which fine-tunes the same base tower.
    --backbone blip_ir     ImageReward-v1.0's BLIP visual encoder, CLS token
                           (run in the subc-ir environment).

Rows are aligned to data/clip_metadata.csv order (the canonical manifest), same
as data/clip_embeddings.npy. Outputs go to backbone_per_backbone/embeddings/:
    <backbone>.npy   float32 (N, D)
    <backbone>.meta.json
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
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = DATASET
META_PATH = REPO_ROOT / "data" / "clip_metadata.csv"
OUT_DIR = OUTPUT / "embeddings"


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_vith_pick(meta: pd.DataFrame, device: str, batch_size: int) -> tuple[np.ndarray, dict]:
    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").to(device).eval()

    out = []
    for start in range(0, len(meta), batch_size):
        chunk = meta.iloc[start:start + batch_size]
        imgs = [Image.open(REPO_ROOT / r).convert("RGB") for r in chunk["relative_path"]]
        inputs = proc(images=imgs, return_tensors="pt").to(device)
        with torch.inference_mode():
            feats = model.get_image_features(**inputs)
            if hasattr(feats, "image_embeds"):
                feats = feats.image_embeds
        out.append(feats.float().cpu().numpy())
        if (start // batch_size) % 20 == 0:
            print(f"  [vith_pick] {start + len(chunk)}/{len(meta)}", flush=True)
    emb = np.concatenate(out, axis=0)
    info = {
        "backbone": "vith_pick",
        "checkpoint_id": "yuvalkirstain/PickScore_v1 (image tower, get_image_features)",
        "processor": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "note": "PickScore's own fine-tuned ViT-H/14; closest proxy for HPS-v2 (same base tower)",
        "torch": torch.__version__,
    }
    return emb, info


def extract_blip_ir(meta: pd.DataFrame, device: str, batch_size: int) -> tuple[np.ndarray, dict]:
    import torch
    import ImageReward as RM
    from PIL import Image

    model = RM.load("ImageReward-v1.0", device=device)
    model.blip.visual_encoder.eval()

    out = []
    for start in range(0, len(meta), batch_size):
        chunk = meta.iloc[start:start + batch_size]
        imgs = [model.preprocess(Image.open(REPO_ROOT / r).convert("RGB"))
                for r in chunk["relative_path"]]
        batch = torch.stack(imgs).to(device)
        with torch.inference_mode():
            image_embeds = model.blip.visual_encoder(batch)   # (B, tokens, D)
            cls = image_embeds[:, 0, :]                       # CLS token
        out.append(cls.float().cpu().numpy())
        if (start // batch_size) % 20 == 0:
            print(f"  [blip_ir] {start + len(chunk)}/{len(meta)}", flush=True)
    emb = np.concatenate(out, axis=0)
    info = {
        "backbone": "blip_ir",
        "checkpoint_id": "THUDM/ImageReward ImageReward-v1.0 (BLIP visual encoder, CLS token)",
        "note": "ImageReward's own fine-tuned BLIP vision tower",
        "torch": torch.__version__,
    }
    return emb, info


EXTRACTORS = {"vith_pick": extract_vith_pick, "blip_ir": extract_blip_ir}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(EXTRACTORS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Debug: first N images only")
    ap.add_argument("--out-suffix", default="", help="e.g. '_smoke' to avoid clobbering a full run")
    args = ap.parse_args()

    meta = pd.read_csv(META_PATH)
    if args.limit:
        meta = meta.head(args.limit).copy()
    device = args.device or pick_device()
    OUT_DIR.mkdir(exist_ok=True)

    print(f"[info] backbone={args.backbone}  device={device}  N={len(meta)}")
    t0 = time.time()
    emb, info = EXTRACTORS[args.backbone](meta, device, args.batch_size)
    assert emb.shape[0] == len(meta), f"row mismatch: {emb.shape[0]} vs {len(meta)}"
    assert not np.isnan(emb).any(), "NaNs in embeddings"

    out_npy = OUT_DIR / f"{args.backbone}{args.out_suffix}.npy"
    np.save(out_npy, emb.astype(np.float32))
    payload = {
        **info,
        "n": int(emb.shape[0]), "dim": int(emb.shape[1]),
        "device": device, "wall_clock_s": round(time.time() - t0, 1),
        "npy_sha256": hashlib.sha256(out_npy.read_bytes()).hexdigest(),
        "row_order": "data/clip_metadata.csv (canonical)",
    }
    (OUT_DIR / f"{args.backbone}{args.out_suffix}.meta.json").write_text(json.dumps(payload, indent=2))
    print(f"[done] wrote {out_npy}  shape={emb.shape}")
    print(json.dumps({k: payload[k] for k in ('n', 'dim', 'device', 'wall_clock_s')}, indent=2))


if __name__ == "__main__":
    main()
