"""
Extract the interpretable per-image features used in the LAION-A regression (Table 4).

For each of the 2,610 images we compute three families of features:

  Palette / color
    sat_mean, sat_std            HSV saturation, channel-wise
    lum_mean, lum_std            HSV value channel
    colorfulness                 Hasler & Süsstrunk (2003) image-colorfulness metric
    palette_diversity            max pairwise CIEDE2000 across the 5 dominant
                                 dominant colors in data/color_analysis/
    palette_top_pct              % of pixels claimed by the dominant color

  Polish proxies
    laplacian_var                sharpness via Laplacian variance
    edge_density                 fraction of pixels marked by Canny(50, 150)
    local_contrast_std           std of local-contrast map (5x5 stdfilt)
    bytes_per_pixel              file_size / (W*H) — JPEG-quality proxy

  Composition / faces
    aspect_ratio                 W / H
    face_count                   OpenCV Haar cascade frontal-face detections

Inputs:
  data/clip_metadata.csv         canonical image list (2,610 rows)
  data/images/<sub>/image_<N>.jpg
  data/color_analysis/<sub>.json (K-means dominant colors, for palette_diversity)

Output:
  data/interpretable_features.csv with the 12 columns above plus
  index, filename, subculture, relative_path.

Runtime estimate: ~0.5-1.0 sec/image on CPU = ~25-45 min for 2,610.
Stateless and embarrassingly parallel; if needed, split via --shard / --of.
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
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


REPO_ROOT = DATASET
META_PATH = REPO_ROOT / "data" / "clip_metadata.csv"
COLOR_DIR = REPO_ROOT / "data" / "color_analysis"
OUT_PATH  = OUTPUT / "interpretable_features.csv"

FEATURE_COLS = [
    "sat_mean", "sat_std", "lum_mean", "lum_std",
    "colorfulness", "palette_diversity", "palette_top_pct",
    "laplacian_var", "edge_density", "local_contrast_std", "bytes_per_pixel",
    "aspect_ratio", "face_count",
]


# --------------------------------------------------------------------------- #
# Palette diversity from the K-means dominant colors
# --------------------------------------------------------------------------- #

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Single-color RGB (0-255) -> LAB. cv2 expects uint8 array."""
    arr = np.array([[rgb]], dtype=np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)


def _ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """Standard CIEDE2000 distance — formula reproduced from Sharma et al. 2005."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    avg_L = (L1 + L2) / 2.0
    C1 = math.hypot(a1, b1); C2 = math.hypot(a2, b2)
    avg_C = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    a1p = (1 + G) * a1; a2p = (1 + G) * a2
    C1p = math.hypot(a1p, b1); C2p = math.hypot(a2p, b2)
    avg_Cp = (C1p + C2p) / 2.0
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    if abs(dhp) > 180: dhp -= math.copysign(360, dhp)
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp) / 2)
    avg_hp = (h1p + h2p) / 2 if abs(h1p - h2p) <= 180 else (h1p + h2p + 360) / 2
    T = (1 - 0.17 * math.cos(math.radians(avg_hp - 30))
            + 0.24 * math.cos(math.radians(2 * avg_hp))
            + 0.32 * math.cos(math.radians(3 * avg_hp + 6))
            - 0.20 * math.cos(math.radians(4 * avg_hp - 63)))
    SL = 1 + (0.015 * (avg_L - 50) ** 2) / math.sqrt(20 + (avg_L - 50) ** 2)
    SC = 1 + 0.045 * avg_Cp
    SH = 1 + 0.015 * avg_Cp * T
    delta_theta = 30 * math.exp(-((avg_hp - 275) / 25) ** 2)
    RC = 2 * math.sqrt(avg_Cp ** 7 / (avg_Cp ** 7 + 25 ** 7))
    RT = -math.sin(math.radians(2 * delta_theta)) * RC
    return math.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
                     + RT * (dCp / SC) * (dHp / SH))


def load_color_analysis() -> dict[tuple[str, str], dict]:
    """(filename, subculture) -> {colors: [...], top_pct: float}"""
    out = {}
    for jpath in sorted(COLOR_DIR.glob("*.json")):
        sub = jpath.stem.replace("dark_academia", "dark_academia").lower()
        # JSON keys vary; just pull image_analysis records
        data = json.loads(jpath.read_text())
        for rec in data.get("image_analysis", []):
            colors = rec.get("colors", [])
            if not colors:
                continue
            out[(rec["filename"], sub)] = {
                "rgbs": [c["rgb"] for c in colors],
                "top_pct": colors[0].get("percentage", 0.0),
            }
    return out


def palette_features(filename: str, subculture: str,
                     color_index: dict) -> tuple[float, float]:
    """(palette_diversity, palette_top_pct).

    diversity = max pairwise CIEDE2000 across the 5 dominant LAB colors.
    """
    rec = color_index.get((filename, subculture))
    if rec is None:
        return float("nan"), float("nan")
    rgbs = rec["rgbs"]
    labs = [_rgb_to_lab(np.array(rgb)) for rgb in rgbs]
    if len(labs) < 2:
        return 0.0, float(rec["top_pct"])
    diversity = max(_ciede2000(labs[i], labs[j])
                    for i in range(len(labs))
                    for j in range(i + 1, len(labs)))
    return float(diversity), float(rec["top_pct"])


# --------------------------------------------------------------------------- #
# Per-image image features
# --------------------------------------------------------------------------- #

# OpenCV's bundled frontal-face cascade, loaded once and shared.
HAAR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def colorfulness_hasler(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) image colorfulness metric."""
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma = math.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mu    = math.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(sigma + 0.3 * mu)


def per_image_features(path: Path, byte_size: int) -> dict:
    pil = Image.open(path).convert("RGB")
    rgb = np.array(pil)
    H, W = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    # Local-contrast std: stdev of pixel values in a 5x5 sliding window, then std of that map
    g32 = gray.astype(np.float32)
    mean5 = cv2.boxFilter(g32, -1, (5, 5))
    sq5   = cv2.boxFilter(g32 * g32, -1, (5, 5))
    var5  = np.clip(sq5 - mean5 ** 2, 0, None)
    local_contrast = np.sqrt(var5)
    faces = HAAR.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return {
        "sat_mean": float(sat.mean()),
        "sat_std":  float(sat.std()),
        "lum_mean": float(val.mean()),
        "lum_std":  float(val.std()),
        "colorfulness": colorfulness_hasler(rgb),
        "laplacian_var": laplacian_var,
        "edge_density": edge_density,
        "local_contrast_std": float(local_contrast.std()),
        "bytes_per_pixel": byte_size / (H * W) if H * W > 0 else float("nan"),
        "aspect_ratio": W / H if H > 0 else float("nan"),
        "face_count": int(len(faces) if isinstance(faces, np.ndarray) else 0),
    }


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0, help="0-indexed shard to compute")
    ap.add_argument("--of", type=int, default=1, help="total shards")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    metadata = pd.read_csv(META_PATH)
    if args.limit:
        metadata = metadata.head(args.limit).copy()
    if args.of > 1:
        metadata = metadata.iloc[args.shard::args.of].copy()

    color_index = load_color_analysis()
    print(f"[info] color_analysis: {len(color_index)} (filename, subculture) entries")

    rows = []
    n = len(metadata)
    for i, row in enumerate(metadata.itertuples(index=False), start=1):
        path = REPO_ROOT / row.relative_path
        try:
            byte_size = path.stat().st_size
            feats = per_image_features(path, byte_size)
        except Exception as exc:
            print(f"[FAIL] {row.filename} ({row.subculture}): {exc}")
            feats = {c: float("nan") for c in FEATURE_COLS if c not in {"palette_diversity", "palette_top_pct"}}
        diversity, top_pct = palette_features(row.filename, row.subculture, color_index)
        feats["palette_diversity"] = diversity
        feats["palette_top_pct"]   = top_pct
        rows.append({
            "index": int(row.index),
            "filename": row.filename,
            "subculture": row.subculture,
            "relative_path": row.relative_path,
            **feats,
        })
        if i % 100 == 0 or i == n:
            print(f"[progress] {i}/{n}")

    df = pd.DataFrame(rows, columns=["index", "filename", "subculture", "relative_path"] + FEATURE_COLS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[done] wrote {args.out}  ({len(df)} rows)")
    print(df[FEATURE_COLS].describe().round(4).to_string())


if __name__ == "__main__":
    main()
