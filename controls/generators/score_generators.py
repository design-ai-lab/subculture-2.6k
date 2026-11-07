"""
Score cross-generator images with LAION-A V2 and compare orderings.

Pipeline identity with the corpus and photograph scoring: CLIP ViT-L-14 ("openai") via
open_clip, L2-normalized embeddings, pinned LAION-A MLP head from models/
(this pipeline reproduces the released corpus embeddings at cosine 1.0000).
Vision-only, so the text prompt plays no role.

Input:  $SUBCULTURE_DATA/cross_generator/images/<generator>/<subculture>/*.png
Output: output/generator_scores.csv        per-image scores
        output/generator_report.txt        per-generator per-subculture means,
                                           ranks, Spearman rho vs the SD corpus
                                           ordering, Y2K-bottom check

Usage:  python3 controls/generators/score_generators.py
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


from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = DATASET
SUBCULTURES = ["cottagecore", "cyberpunk", "dark_academia",
               "goblincore", "synthwave", "y2k"]


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def pick_device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else \
           ("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    import torch
    import torch.nn as nn
    import open_clip
    from PIL import Image
    from scipy import stats

    device = pick_device()
    print(f"[info] device={device}")

    gens = sorted(d.name for d in (DATASET / "cross_generator" / "images").iterdir()
                  if d.is_dir() and any(next((d / s).glob("*"), None) is not None
                                        for s in SUBCULTURES if (d / s).is_dir()))
    if not gens:
        raise SystemExit("[FAIL] no images found under images/<generator>/<subculture>/")
    print(f"[info] generators found: {gens}")

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    model = model.to(device).eval()

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

    rows = []
    for g in gens:
        for sub in SUBCULTURES:
            files = sorted(p for p in (DATASET / "cross_generator" / "images" / g / sub).glob("*")
                           if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
            for i in range(0, len(files), 16):
                batch = files[i:i + 16]
                tensors = [preprocess(Image.open(f).convert("RGB")) for f in batch]
                with torch.inference_mode():
                    emb = model.encode_image(torch.stack(tensors).to(device)).float().cpu().numpy()
                    scores = head(torch.from_numpy(l2(emb))).squeeze(-1).numpy()
                rows += [{"generator": g, "subculture": sub, "filename": f.name,
                          "laion_score": float(s)} for f, s in zip(batch, np.atleast_1d(scores))]
            if files:
                print(f"  [{g}] {sub}: {len(files)} images", flush=True)

    df = pd.DataFrame(rows)
    OUTPUT.mkdir(exist_ok=True)
    df.to_csv(OUTPUT / "generator_scores.csv", index=False)

    sd = pd.read_csv(REPO_ROOT / "data" / "scores" / "scores_laion_v2.csv")
    sd_mean = sd.groupby("subculture")["score"].mean()

    def ranks(means: pd.Series) -> pd.Series:
        return means.rank(ascending=False).astype(int)  # 1 = highest score

    sd_rank = ranks(sd_mean)
    lines = ["Cross-generator check",
             f"Scorer: LAION-A V2 vision-only (photograph pipeline identity)", ""]

    header = f"{'subculture':<14} {'SD corpus':>12}"
    for g in gens:
        header += f" {g:>16}"
    lines += [header + "   (mean | rank, 1 = highest)"]
    gen_means = {g: df[df.generator == g].groupby("subculture")["laion_score"].mean()
                 for g in gens}
    for sub in sd_mean.sort_values(ascending=False).index:
        line = f"{sub:<14} {sd_mean[sub]:>7.3f} r{sd_rank[sub]}"
        for g in gens:
            m = gen_means[g]
            if sub in m.index:
                line += f" {m[sub]:>11.3f} r{ranks(m)[sub]}"
            else:
                line += f" {'—':>14}"
        lines.append(line)
    lines.append("")

    for g in gens:
        m = gen_means[g]
        common = [s for s in SUBCULTURES if s in m.index]
        rho, p = stats.spearmanr(sd_mean[common], m[common])
        n = len(df[df.generator == g])
        y2k_bottom = m.idxmin() == "y2k" if "y2k" in m.index else None
        lines += [f"[{g}] n={n} images | Spearman rho vs SD ordering: {rho:.2f} (p={p:.3f}, "
                  f"k={len(common)} classes) | y2k lowest: {y2k_bottom}"]
        if "y2k" in m.index:
            others = m.drop("y2k")
            lines += [f"[{g}] y2k mean {m['y2k']:.3f} vs other-class mean "
                      f"{others.mean():.3f} (gap {others.mean() - m['y2k']:+.3f})"]
    lines += ["", "Reading: rho near 1 with y2k at/near the bottom means the SD-corpus",
              "ordering is not SD-specific; disagreement localizes it to the generator.",
              "Claim level stays qualitative: n=8 per class per generator by design.", ""]

    report = "\n".join(lines)
    (OUTPUT / "generator_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
