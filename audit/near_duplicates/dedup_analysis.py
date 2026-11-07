"""
Near-duplicate audit on the released CLIP ViT-L/14 embeddings, plus stability of the paper's
headline numbers after excluding near-duplicates.

Method: cosine similarity on L2-normalized data/clip_embeddings.npy.
Pairs >= tau are near-duplicates; connected components collapsed keeping the
lowest canonical index. Primary tau = 0.95, sensitivity at 0.90/0.925/0.97/0.99.

Headline stats recomputed on the deduplicated subset (original conditioning):
  - LAION-A eta^2 by subculture                     (paper: 0.379)
  - inter-peer mean rank rho A / LAION-peer mean B  (paper: 0.70 / 0.30)
  - >=6.5 cutoff retention, Cyberpunk vs Y2K ratio  (paper: ~22x)
  - ViT-L separability, LR + RF, 5-fold stratified  (paper: 92.4 / 92.7)
    (near-dupes could inflate CV accuracy via fold leakage — the key check)

Outputs: output/dedup_report.txt, output/dedup_dupe_pairs.csv,
         output/dedup_headline_stability.csv
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


import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO_ROOT = DATASET
import cross_scorer_analysis as xsa

TAUS = [0.90, 0.925, 0.95, 0.97, 0.99]
TAU_PRIMARY = 0.95
SEED = 42
SCORERS = {"LAION-A V2": "scores_laion_v2.csv", "PickScore": "scores_pickscore.csv",
           "ImageReward": "scores_image_reward.csv", "HPS-v2": "scores_hps_v2.csv"}
PEERS = ["PickScore", "ImageReward", "HPS-v2"]


def components(n: int, pairs: np.ndarray) -> np.ndarray:
    parent = np.arange(n)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


def ab_from_frames(frames: dict) -> tuple[float, float]:
    means = xsa.per_subculture_means(frames)
    ranks = xsa.per_subculture_ranks(means)
    spear = xsa.spearman_subculture_ranks(ranks)
    A = float(np.mean([spear.loc[a, b] for i, a in enumerate(PEERS) for b in PEERS[i + 1:]]))
    B = float(np.mean([spear.loc["LAION-A V2", p] for p in PEERS]))
    return A, B


def main() -> None:
    out_dir = OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = np.load(REPO_ROOT / "data" / "clip_embeddings.npy").astype(np.float32)
    meta = pd.read_csv(REPO_ROOT / "data" / "clip_metadata.csv")
    n = len(meta)
    emb_n = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    sim = emb_n @ emb_n.T
    iu = np.triu_indices(n, k=1)

    # --- duplicate counts per threshold -------------------------------------
    curve = []
    for tau in TAUS:
        mask = sim[iu] >= tau
        idx_a, idx_b = iu[0][mask], iu[1][mask]
        cross = (meta.subculture.values[idx_a] != meta.subculture.values[idx_b]).sum()
        comp = components(n, np.stack([idx_a, idx_b], axis=1))
        n_excluded = n - len(np.unique(comp))
        curve.append({"tau": tau, "n_pairs": int(mask.sum()),
                      "n_cross_subculture_pairs": int(cross),
                      "n_images_excluded": int(n_excluded)})
    curve_df = pd.DataFrame(curve)

    # --- primary threshold: pair list + keep mask ---------------------------
    mask = sim[iu] >= TAU_PRIMARY
    idx_a, idx_b = iu[0][mask], iu[1][mask]
    pairs = pd.DataFrame({
        "index_a": idx_a, "file_a": meta.filename.values[idx_a],
        "sub_a": meta.subculture.values[idx_a],
        "index_b": idx_b, "file_b": meta.filename.values[idx_b],
        "sub_b": meta.subculture.values[idx_b],
        "cosine": np.round(sim[iu][mask], 4),
    }).sort_values("cosine", ascending=False)
    pairs.to_csv(out_dir / "dedup_dupe_pairs.csv", index=False)

    comp = components(n, np.stack([idx_a, idx_b], axis=1))
    keep = np.zeros(n, dtype=bool)
    keep[np.unique(comp)] = True          # component representative = lowest index
    print(f"[info] tau={TAU_PRIMARY}: {mask.sum()} pairs, excluding {n - keep.sum()} images "
          f"-> {keep.sum()} remain")

    # --- headline stability on deduped subset -------------------------------
    frames_full, frames_dd = {}, {}
    for name, fname in SCORERS.items():
        df = pd.read_csv(REPO_ROOT / "data" / "scores" / fname)
        frames_full[name] = df
        frames_dd[name] = df[keep].reset_index(drop=True)

    rows = []
    # eta^2 (LAION)
    for tag, fr in [("full", frames_full), ("dedup", frames_dd)]:
        eta = xsa.per_scorer_anova(fr).loc["LAION-A V2", "eta_sq"]
        rows.append({"stat": "laion_eta_sq", "set": tag, "value": round(float(eta), 4)})
    # A / B
    for tag, fr in [("full", frames_full), ("dedup", frames_dd)]:
        A, B = ab_from_frames(fr)
        rows.append({"stat": "inter_peer_mean_rho_A", "set": tag, "value": round(A, 3)})
        rows.append({"stat": "laion_peer_mean_rho_B", "set": tag, "value": round(B, 3)})
    # retention ratio at 6.5
    for tag, fr in [("full", frames_full), ("dedup", frames_dd)]:
        laion = fr["LAION-A V2"]
        ret = laion.groupby("subculture").score.apply(lambda s: (s >= 6.5).mean())
        rows.append({"stat": "retention_6.5_cyberpunk", "set": tag, "value": round(float(ret["cyberpunk"]), 4)})
        rows.append({"stat": "retention_6.5_y2k", "set": tag, "value": round(float(ret["y2k"]), 4)})
        rows.append({"stat": "retention_ratio_cyb_over_y2k", "set": tag,
                     "value": round(float(ret["cyberpunk"] / max(ret["y2k"], 1e-9)), 1)})
    # separability (the leakage check)
    y = meta["subculture"].values
    for tag, m in [("full", np.ones(n, dtype=bool)), ("dedup", keep)]:
        X, yv = emb_n[m], y[m]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        lr = cross_val_score(LogisticRegression(max_iter=2000),
                             StandardScaler().fit_transform(X), yv, cv=cv).mean()
        rf = cross_val_score(RandomForestClassifier(n_estimators=300, random_state=SEED,
                                                    n_jobs=-1), X, yv, cv=cv).mean()
        rows.append({"stat": "vitl_logreg_acc", "set": tag, "value": round(float(lr), 4)})
        rows.append({"stat": "vitl_rf_acc", "set": tag, "value": round(float(rf), 4)})

    stab = pd.DataFrame(rows).pivot(index="stat", columns="set", values="value")[["full", "dedup"]]
    stab.to_csv(out_dir / "dedup_headline_stability.csv")

    report = "\n".join([
        f"Near-duplicate audit — generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "Embeddings: data/clip_embeddings.npy, cosine on L2-normalized rows.",
        "",
        "=== duplicate counts by threshold ===", curve_df.to_string(index=False), "",
        f"=== primary threshold tau={TAU_PRIMARY} ===",
        f"pairs: {len(pairs)}   images excluded: {n - int(keep.sum())}   remaining: {int(keep.sum())}/{n}",
        f"cross-subculture pairs (label-contamination signal): "
        f"{int((pairs.sub_a != pairs.sub_b).sum())}",
        "",
        "=== headline stability (full vs deduplicated) ===", stab.to_string(), "",
        "Reference paper values: eta_sq 0.379 | A 0.70 | B 0.30 | retention ratio ~22x | LR/RF 92.4/92.7",
        "",
    ])
    (out_dir / "dedup_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
