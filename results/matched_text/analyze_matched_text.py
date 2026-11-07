"""
Matched-text analysis — the three text-conditioned predictors rescored under
neutral or label text.

Compares the cross-scorer agreement structure under a matched text input
against the original (prompt-conditioned) scores. Statistics are computed by
importing `results/laion_profile/cross_scorer_analysis.py`, so matched-text
numbers and original-prompt numbers come from the identical code path.

Inputs
------
* LAION-A V2: `data/scores/scores_laion_v2.csv` (vision-only, input-invariant).
* Peers (PickScore, ImageReward, HPS-v2): `matched_text/scores_<scorer>__<arm>.csv`
  in the dataset. The arm `const` reads the files suffixed `__an_image`.

Usage
-----
    python analyze_matched_text.py --arm empty
    python analyze_matched_text.py --arm const            # the fixed text "an image"
    python analyze_matched_text.py --arm label
    # self-validation on the original peer scores; reproduces the paper's
    # original-prompt values (A ~0.70, B ~0.30):
    python analyze_matched_text.py --arm origcheck --scores-dir "$SUBCULTURE_DATA/data/scores" --suffix ""

Outputs (output/)
-----------------
    matched_<arm>_per_subculture_means.csv
    matched_<arm>_per_subculture_ranks.csv     (1 = lowest, 6 = highest, within scorer)
    matched_<arm>_anova.csv                    (per-scorer F, p, eta_sq)
    matched_<arm>_spearman_matrix.csv          (4x4 subculture-rank agreement)
    matched_<arm>_orig_vs_arm_per_image.csv    (per-peer Spearman, orig vs arm scores)
    matched_<arm>_vs_laion_per_image.csv       (per-peer Spearman vs LAION-A, this arm)
    matched_<arm>_report.txt                   (all tables, summary statistics A/B/C and a
                                             markdown table)

Summary statistics (original-prompt baselines as reported in the paper):
    A = mean inter-peer subculture-rank Spearman rho          (baseline 0.70)
    B = mean LAION-vs-peer subculture-rank Spearman rho       (baseline 0.30)
    C = Goblincore ranked top by all three peers AND Cyberpunk top on LAION-A
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
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = DATASET

import numpy as np
import pandas as pd
from scipy import stats

import cross_scorer_analysis as xsa

N_EXPECTED = 2610
BASELINE_A = 0.70   # inter-peer mean rank rho, original prompts, as reported in the paper
BASELINE_B = 0.30   # LAION-vs-peer mean rank rho, original prompts
BASELINE_VS_LAION = {"PickScore": 0.14, "ImageReward": 0.16, "HPS-v2": 0.34}

PEERS = [  # display name, file stem (matches score_images.py ScorerInfo.scorer)
    ("PickScore",   "pickscore"),
    ("ImageReward", "image_reward"),
    ("HPS-v2",      "hps_v2"),
]
LAION = "LAION-A V2"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scores(path: Path, ref: pd.DataFrame | None) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[FAIL] missing scores file: {path} (scoring not finished?)")
    df = pd.read_csv(path)
    assert len(df) == N_EXPECTED, f"{path.name}: {len(df)} rows, expected {N_EXPECTED}"
    assert df.score.isna().sum() == 0, f"{path.name}: NaN scores"
    if ref is not None:  # row alignment with the canonical manifest order
        same = (df["filename"].values == ref["filename"].values).all() and \
               (df["subculture"].values == ref["subculture"].values).all()
        assert same, f"{path.name}: row order differs from original scores"
    return df


def mean_offdiag(mat: pd.DataFrame, rows: list[str], cols: list[str]) -> float:
    vals = [mat.loc[a, b] for i, a in enumerate(rows) for b in cols[i + 1:]] \
        if rows == cols else [mat.loc[a, b] for a in rows for b in cols]
    return float(np.mean(vals))


def to_markdown_ranks(ranks: pd.DataFrame, arm: str) -> str:
    subs = list(ranks.index)
    lines = ["| scorer (conditioning) | " + " | ".join(s.replace("_", " ") for s in subs) + " |",
             "|" + "---|" * (len(subs) + 1)]
    for name in ranks.columns:
        cond = "image only" if name == LAION else arm
        row = " | ".join(str(int(ranks.loc[s, name])) for s in subs)
        lines.append(f"| {name} ({cond}) | {row} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="matched text neutral-prompt analysis")
    ap.add_argument("--arm", required=True,
                    help="empty | const | label | origcheck (free-form; used in filenames)")
    ap.add_argument("--scores-dir", type=Path, default=DATASET / "matched_text")
    ap.add_argument("--suffix", default=None,
                    help="filename suffix of the arm CSVs (default '__<arm>', '__an_image' for const; '' for origcheck)")
    args = ap.parse_args()
    if args.suffix is not None:
        suffix = args.suffix
    else:
        suffix = "__an_image" if args.arm == "const" else f"__{args.arm}"

    out_dir = OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_dir = REPO_ROOT / "data" / "scores"
    laion_df = load_scores(orig_dir / "scores_laion_v2.csv", ref=None)

    frames, orig_frames, inputs = {LAION: laion_df}, {}, {}
    for name, stem in PEERS:
        orig_frames[name] = load_scores(orig_dir / f"scores_{stem}.csv", ref=laion_df)
        p = args.scores_dir / f"scores_{stem}{suffix}.csv"
        frames[name] = load_scores(p, ref=laion_df)
        inputs[name] = {"path": str(p), "sha256": sha256_of(p)}

    # --- core tables via the paper code ----------------------------------------
    means  = xsa.per_subculture_means(frames)
    ranks  = xsa.per_subculture_ranks(means)
    anova  = xsa.per_scorer_anova(frames)
    spear  = xsa.spearman_subculture_ranks(ranks)
    vs_laion = xsa.per_image_corr_vs_laion(frames)

    # --- orig-vs-arm per-image correlation, per peer -------------------------
    rows = []
    for name, _ in PEERS:
        rho, p = stats.spearmanr(orig_frames[name].score, frames[name].score)
        r, _p  = stats.pearsonr(orig_frames[name].score, frames[name].score)
        rows.append({"peer": name, "spearman_rho_orig_vs_arm": round(rho, 3),
                     "spearman_p": p, "pearson_r": round(r, 3)})
    orig_vs_arm = pd.DataFrame(rows).set_index("peer")

    # --- decision rules ------------------------------------------------------
    peer_names = [n for n, _ in PEERS]
    A = mean_offdiag(spear, peer_names, peer_names)
    B = float(np.mean([spear.loc[LAION, p] for p in peer_names]))
    top = ranks.idxmax(axis=0)            # subculture with rank 6, per scorer
    C_peers = all(top[p] == "goblincore" for p in peer_names)
    C_laion = top[LAION] == "cyberpunk"
    C = C_peers and C_laion

    if C and (A - B) >= 0.30:
        outcome = "Outcome 1 — divergence SURVIVES (conditioning exonerated)"
    elif (not C) or (A - B) < 0.10:
        outcome = "Outcome 2 — divergence COLLAPSES (STOP, flag to author before writing)"
    else:
        outcome = "Outcome 3 — partial (concede conditioned share, quantify residual)"

    # --- persist -------------------------------------------------------------
    arm = args.arm
    means.to_csv(out_dir / f"matched_{arm}_per_subculture_means.csv")
    ranks.to_csv(out_dir / f"matched_{arm}_per_subculture_ranks.csv")
    anova.to_csv(out_dir / f"matched_{arm}_anova.csv")
    spear.to_csv(out_dir / f"matched_{arm}_spearman_matrix.csv")
    orig_vs_arm.to_csv(out_dir / f"matched_{arm}_orig_vs_arm_per_image.csv")
    vs_laion.to_csv(out_dir / f"matched_{arm}_vs_laion_per_image.csv")

    vs_laion_line = "  ".join(
        f"{p}: {vs_laion.loc[p, 'spearman_rho']:.2f} (paper {BASELINE_VS_LAION[p]:.2f})"
        for p in peer_names)
    report = "\n".join([
        f"matched text neutral-prompt analysis — arm={arm}",
        f"generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"inputs: {json.dumps(inputs, indent=2)}",
        "",
        "=== SUMMARY ===",
        f"A  mean inter-peer rank rho:  {A:.3f}   (paper baseline {BASELINE_A:.2f})",
        f"B  mean LAION-vs-peer rho:    {B:.3f}   (paper baseline {BASELINE_B:.2f})",
        f"C  Goblincore top on all peers: {C_peers};  Cyberpunk top on LAION-A: {C_laion}",
        f"=> {outcome}",
        "   (threshold-based suggestion only — author judges the call)",
        "",
        f"top-ranked subculture per scorer: {top.to_dict()}",
        "",
        "=== per-image Spearman, orig vs arm (how much the caption mattered) ===",
        orig_vs_arm.to_string(),
        "",
        f"=== per-image Spearman vs LAION-A, this arm (paper: 0.14/0.16/0.34) ===",
        vs_laion_line,
        "",
        "=== per-subculture means ===", means.round(4).to_string(),
        "", "=== per-subculture ranks (1=lowest, 6=highest) ===", ranks.to_string(),
        "", "=== one-way ANOVA by subculture ===",
        anova[["F", "p_anova", "eta_sq"]].round(4).to_string(),
        "", "=== 4x4 subculture-rank Spearman matrix ===", spear.to_string(),
        "",
        "=== markdown table (ranks) ===",
        to_markdown_ranks(ranks, arm),
        "",
    ])
    (out_dir / f"matched_{arm}_report.txt").write_text(report)
    print(report)
    print(f"[done] wrote 6 CSVs + matched_{arm}_report.txt to {out_dir}")


if __name__ == "__main__":
    main()
