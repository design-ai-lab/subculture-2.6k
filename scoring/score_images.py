"""
Score the 2,610 SubCulture-2.6K images with one of the four aesthetic predictors.

    python scoring/score_images.py --scorer laion
    python scoring/score_images.py --scorer pick --prompts $SUBCULTURE_DATA/data/prompts/prompts_with_flags.csv
    python scoring/score_images.py --scorer ir   --prompts $SUBCULTURE_DATA/data/prompts/prompts_with_flags.csv
    python scoring/score_images.py --scorer hps  --prompts $SUBCULTURE_DATA/data/prompts/prompts_with_flags.csv

The image list and row order come from `data/clip_metadata.csv`. Each run writes
one CSV (index, filename, subculture, relative_path, score) and a `.meta.json`
sidecar (checkpoint, device, library versions, score range) to `output/scores/`
unless `--out-dir` is given. PickScore, ImageReward and HPS-v2 need different
library versions; run each from its environment in `environments/`.

LAION-A is vision-only. The other three take a text input: `--prompts` (a CSV
with filename, subculture, prompt) gives the original generation prompts, and
`--prompt-fallback` replaces them with the subculture label (`subculture`), an
empty string (`empty`) or the fixed text "an image" (`const`). Use `--out-suffix`
to keep these runs apart (see results/matched_text/run_matched_text.sh).

If the output CSV already exists with the full row count, the script exits
unless `--overwrite` is given.
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
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = DATASET
META_PATH = REPO_ROOT / "data" / "clip_metadata.csv"
SCORES_DIR = OUTPUT / "scores"

SCORER_CHOICES = ("laion", "pick", "ir", "hps")

OUTPUT_COLUMNS = ["index", "filename", "subculture", "relative_path", "score"]


# --------------------------------------------------------------------------- #
# Scorer adapter contract
# --------------------------------------------------------------------------- #

@dataclass
class ScorerInfo:
    """Metadata written to the sidecar JSON for reproducibility (Appendix I)."""
    scorer: str
    checkpoint_id: str           # HuggingFace repo or URL
    checkpoint_commit: str       # pinned commit hash (or "pending" if not yet set)
    score_scale: str             # human-readable description of the output scale
    expects_prompt: bool
    preprocessing: str           # short description of the pipeline applied
    library_versions: dict       # filled at runtime by the adapter


class Scorer:
    """Minimal interface every scorer adapter implements.

    Implementations live below; they import heavy deps locally so that
    `python score_images.py --scorer pick` in `subc-pick` does not choke on
    ImageReward's dependencies (and vice-versa).
    """

    info: ScorerInfo

    def load(self, device: str) -> None: ...
    def score(
        self,
        rows: pd.DataFrame,
        prompt_lookup: Callable[[str], str] | None,
        batch_size: int,
    ) -> np.ndarray:
        """Return a float32 array of scores aligned to `rows` order."""
        ...


# --------------------------------------------------------------------------- #
# LAION-Aesthetics V2 (vision-only, uses the released CLIP embeddings)
# --------------------------------------------------------------------------- #

class LaionScorer(Scorer):
    info = ScorerInfo(
        scorer="laion_v2",
        checkpoint_id="christophschuhmann/improved-aesthetic-predictor :: sac+logos+ava1-l14-linearMSE.pth",
        checkpoint_commit="improved-aesthetic-predictor@main",
        score_scale="1-10 (AVA-style, continuous)",
        expects_prompt=False,
        preprocessing="L2-normalize CLIP ViT-L/14 embeddings from data/clip_embeddings.npy",
        library_versions={},
    )

    def load(self, device: str) -> None:
        import torch
        import torch.nn as nn

        self.torch = torch
        self.device = torch.device(device)
        self.info.library_versions = {"torch": torch.__version__}

        class AestheticMLP(nn.Module):
            def __init__(self, input_size: int = 768) -> None:
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 1024), nn.Dropout(0.2),
                    nn.Linear(1024, 128),        nn.Dropout(0.2),
                    nn.Linear(128, 64),          nn.Dropout(0.2),
                    nn.Linear(64, 16),
                    nn.Linear(16, 1),
                )
            def forward(self, x): return self.layers(x)

        weights = laion_head()
        model = AestheticMLP(768)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()

        emb_path = REPO_ROOT / "data" / "clip_embeddings.npy"
        self.embeddings = np.load(emb_path).astype(np.float32)

    def score(self, rows, prompt_lookup, batch_size):
        idx = rows["index"].to_numpy()
        x = self.embeddings[idx]
        norms = np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
        x = x / norms
        with self.torch.inference_mode():
            t = self.torch.from_numpy(x).to(self.device)
            out = self.model(t).squeeze(-1).float().cpu().numpy()
        return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# PickScore (CLIP ViT-H/14, pairwise-preference fine-tune on Pick-a-Pic)
# --------------------------------------------------------------------------- #

class PickScorer(Scorer):
    info = ScorerInfo(
        scorer="pickscore",
        checkpoint_id="yuvalkirstain/PickScore_v1",
        checkpoint_commit="unset",  # fill on first run via huggingface_hub.model_info
        score_scale="logit (unbounded, ~0-30 typical for absolute use)",
        expects_prompt=True,
        preprocessing="AutoProcessor from yuvalkirstain/PickScore_v1 (CLIP-H/14, 224, OpenAI norm)",
        library_versions={},
    )

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoProcessor, AutoModel
        self.torch = torch
        self.device = torch.device(device)
        proc_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.processor = AutoProcessor.from_pretrained(proc_id)
        self.model = AutoModel.from_pretrained(self.info.checkpoint_id).to(self.device).eval()
        import transformers
        self.info.library_versions = {"torch": torch.__version__, "transformers": transformers.__version__}

    def score(self, rows, prompt_lookup, batch_size):
        from PIL import Image
        assert prompt_lookup is not None, "PickScore requires --prompts or --prompt-fallback"
        scores = np.empty(len(rows), dtype=np.float32)
        for start in range(0, len(rows), batch_size):
            chunk = rows.iloc[start:start + batch_size]
            imgs = [Image.open(REPO_ROOT / r).convert("RGB") for r in chunk["relative_path"]]
            prompts = [prompt_lookup(f, s) for f, s in zip(chunk["filename"], chunk["subculture"])]
            image_inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            text_inputs = self.processor(text=prompts, padding=True, truncation=True,
                                         max_length=77, return_tensors="pt").to(self.device)
            with self.torch.inference_mode():
                img_emb = self.model.get_image_features(**image_inputs)
                txt_emb = self.model.get_text_features(**text_inputs)
                # transformers>=5 returns a ModelOutput wrapper; <5 returns a Tensor.
                if hasattr(img_emb, "image_embeds"): img_emb = img_emb.image_embeds
                if hasattr(txt_emb, "text_embeds"):  txt_emb = txt_emb.text_embeds
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
                # PickScore's absolute-use convention: per-pair logit_scale * <img,txt>
                logit_scale = self.model.logit_scale.exp()
                batch_scores = (logit_scale * (img_emb * txt_emb).sum(dim=-1)).float().cpu().numpy()
            scores[start:start + len(chunk)] = batch_scores
        return scores


# --------------------------------------------------------------------------- #
# ImageReward (BLIP reward model)
# --------------------------------------------------------------------------- #

class ImageRewardScorer(Scorer):
    info = ScorerInfo(
        scorer="image_reward",
        checkpoint_id="THUDM/ImageReward",
        checkpoint_commit="unset",
        score_scale="~N(0,1), unbounded (typical range [-3, +3])",
        expects_prompt=True,
        preprocessing="ImageReward.load('ImageReward-v1.0') built-in BLIP transform",
        library_versions={},
    )

    def load(self, device: str) -> None:
        import torch
        import ImageReward as RM  # from `pip install image-reward`
        self.torch = torch
        self.device = device
        self.model = RM.load("ImageReward-v1.0", device=device)
        import ImageReward as _ir_pkg
        self.info.library_versions = {
            "torch": torch.__version__,
            "image-reward": getattr(_ir_pkg, "__version__", "unknown"),
        }

    def score(self, rows, prompt_lookup, batch_size):
        assert prompt_lookup is not None, "ImageReward requires --prompts or --prompt-fallback"
        scores = np.empty(len(rows), dtype=np.float32)
        for i, (_, r) in enumerate(rows.iterrows()):
            prompt = prompt_lookup(r["filename"], r["subculture"])
            with self.torch.inference_mode():
                scores[i] = float(self.model.score(prompt, str(REPO_ROOT / r["relative_path"])))
        return scores


# --------------------------------------------------------------------------- #
# HPS-v2 (CLIP ViT-H/14, pairwise-preference on HPD-v2)
# --------------------------------------------------------------------------- #

class HPSv2Scorer(Scorer):
    info = ScorerInfo(
        scorer="hps_v2",
        checkpoint_id="xswu/HPSv2",
        checkpoint_commit="unset",
        score_scale="small positive scalar (relative)",
        expects_prompt=True,
        preprocessing="hpsv2 package built-in (CLIP-H/14, 224)",
        library_versions={},
    )

    def load(self, device: str) -> None:
        import torch
        import hpsv2
        self.torch = torch
        self.device = device
        self.hpsv2 = hpsv2
        self.info.library_versions = {
            "torch": torch.__version__,
            "hpsv2": getattr(hpsv2, "__version__", "unknown"),
        }

    def score(self, rows, prompt_lookup, batch_size):
        assert prompt_lookup is not None, "HPS-v2 requires --prompts or --prompt-fallback"
        scores = np.empty(len(rows), dtype=np.float32)
        for i, (_, r) in enumerate(rows.iterrows()):
            prompt = prompt_lookup(r["filename"], r["subculture"])
            img_path = str(REPO_ROOT / r["relative_path"])
            s = self.hpsv2.score(img_path, prompt, hps_version="v2.1")
            # hpsv2.score returns list[float] for list input, float for single
            scores[i] = float(s[0] if isinstance(s, (list, tuple, np.ndarray)) else s)
        return scores


SCORER_REGISTRY: dict[str, type[Scorer]] = {
    "laion": LaionScorer,
    "pick":  PickScorer,
    "ir":    ImageRewardScorer,
    "hps":   HPSv2Scorer,
}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def build_prompt_lookup(
    prompts_csv: Path | None,
    fallback: str | None,
    metadata: pd.DataFrame,
) -> Callable[[str, str], str] | None:
    """Returns a callable (filename, subculture) -> prompt, or None for vision-only scorers.

    Keying on (filename, subculture) is required because filenames like
    image_32.jpg are reused across the six subculture folders — 484 such
    collisions in the canonical list. A bare filename lookup would silently
    return the wrong prompt for ~18% of rows.
    """
    if prompts_csv is not None:
        df = pd.read_csv(prompts_csv)
        required = {"filename", "subculture", "prompt"}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise ValueError(f"{prompts_csv} is missing columns: {missing_cols}")
        mapping = {(f, s): p for f, s, p in zip(df["filename"], df["subculture"],
                                                df["prompt"].fillna(""))}
        canonical = set(zip(metadata["filename"], metadata["subculture"]))
        uncovered = canonical - set(mapping)
        if uncovered:
            raise ValueError(
                f"Prompts CSV is missing {len(uncovered)} (filename, subculture) pairs "
                f"(e.g., {sorted(uncovered)[:3]}). Use data/prompts/prompts_with_flags.csv."
            )
        return lambda fn, sub: mapping[(fn, sub)]
    if fallback == "subculture":
        return lambda fn, sub: sub.replace("_", " ")
    if fallback == "empty":
        return lambda fn, sub: ""
    if fallback == "const":
        return lambda fn, sub: "an image"
    return None


def pick_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available(): return "mps"
        if torch.cuda.is_available():          return "cuda"
    except Exception:
        pass
    return "cpu"


def sanity_check(scores: np.ndarray, scorer_name: str) -> dict:
    stats = {
        "n": int(scores.size),
        "min":  float(scores.min()),
        "max":  float(scores.max()),
        "mean": float(scores.mean()),
        "std":  float(scores.std()),
        "n_nan": int(np.isnan(scores).sum()),
    }
    if stats["n_nan"] > 0:
        print(f"[WARN] {scorer_name}: {stats['n_nan']} NaN scores", file=sys.stderr)
    if stats["std"] < 1e-6:
        print(f"[WARN] {scorer_name}: score std≈0 — check preprocessing", file=sys.stderr)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Score SubCulture-2.6K with one deployed aesthetic predictor.")
    ap.add_argument("--scorer", required=True, choices=SCORER_CHOICES)
    ap.add_argument("--prompts", type=Path, default=None,
                    help="CSV with columns filename,prompt. Required for pick|ir|hps unless --prompt-fallback is set.")
    ap.add_argument("--prompt-fallback", choices=("subculture", "empty", "const"), default=None,
                    help="Use subculture label, empty string, or the constant 'an image' as prompt "
                         "when --prompts is unavailable. Recorded in sidecar meta.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for the output CSV + sidecar (default: output/scores).")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix appended to the output stem, e.g. '__empty' -> scores_pickscore__empty.csv")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None, help="mps|cuda|cpu (auto-detected if omitted)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Debug: score only the first N images.")
    args = ap.parse_args()

    metadata = pd.read_csv(META_PATH)
    if args.limit:
        metadata = metadata.head(args.limit).copy()

    scorer_cls = SCORER_REGISTRY[args.scorer]
    scorer = scorer_cls()

    out_dir  = args.out_dir if args.out_dir is not None else SCORES_DIR
    stem     = f"scores_{scorer.info.scorer}{args.out_suffix}"
    out_csv  = out_dir / f"{stem}.csv"
    out_meta = out_dir / f"{stem}.meta.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not args.overwrite:
        existing = pd.read_csv(out_csv)
        if len(existing) == len(metadata):
            print(f"[skip] {out_csv} already complete ({len(existing)} rows). --overwrite to rerun.")
            return

    prompt_lookup = build_prompt_lookup(args.prompts, args.prompt_fallback, metadata)
    if scorer.info.expects_prompt and prompt_lookup is None:
        ap.error(f"{args.scorer} requires --prompts or --prompt-fallback")

    device = args.device or pick_device()
    print(f"[info] scorer={args.scorer}  device={device}  N={len(metadata)}")
    t0 = time.time()
    scorer.load(device)
    t_load = time.time() - t0

    t0 = time.time()
    scores = scorer.score(metadata, prompt_lookup, args.batch_size)
    t_score = time.time() - t0
    assert scores.shape == (len(metadata),), f"score shape mismatch: {scores.shape}"

    out = metadata[["index", "filename", "subculture", "relative_path"]].copy()
    out["score"] = scores
    out = out[OUTPUT_COLUMNS]
    out.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}  ({len(out)} rows)")

    meta_payload = {
        **asdict(scorer.info),
        "device": device,
        "wall_clock_load_s":  round(t_load, 2),
        "wall_clock_score_s": round(t_score, 2),
        "prompts_source": (
            str(args.prompts) if args.prompts
            else f"fallback:{args.prompt_fallback}" if args.prompt_fallback
            else "none (vision-only scorer)"
        ),
        "conditioning_arm": (
            "orig" if args.prompts
            else {"empty": "empty", "const": "const", "subculture": "label"}.get(args.prompt_fallback)
            or "vision-only"
        ),
        "output_csv": str(out_csv),
        "score_summary": sanity_check(scores, scorer.info.scorer),
    }
    out_meta.write_text(json.dumps(meta_payload, indent=2))
    print(f"[done] wrote {out_meta}")
    print(json.dumps(meta_payload["score_summary"], indent=2))


if __name__ == "__main__":
    main()
