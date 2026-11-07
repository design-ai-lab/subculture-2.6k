"""
Scrape-order check (corpus independence).

Question: does Lexica search-result position correlate with LAION-A score within
each subculture? High correlation would suggest the corpus itself is
pre-selected along the polish axes LAION rewards; low correlation argues
against it.

Rank proxy: the filename number N in image_N.jpg encodes scrape/collection
order, which is monotone in the Lexica API cursor (the collection script, not
released, pages with strictly forward-moving cursors; random offsets skip positions but never
reorder). Caveats reported alongside: (a) proxy assumes search order tracks
Lexica's ranking; (b) per-subculture prompt files are "combined" reports, so
concatenated scrape sessions may add noise (which biases rho toward zero).

Outputs: output/order_rank_correlations.csv, output/order_report.txt
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


import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO_ROOT = DATASET
SUBCULTURES = ["cottagecore", "cyberpunk", "dark_academia", "goblincore", "synthwave", "y2k"]

SCORE_FILES = {          # original (paper) scores; LAION-A is the primary target
    "LAION-A V2":  "scores_laion_v2.csv",
    "PickScore":   "scores_pickscore.csv",
    "ImageReward": "scores_image_reward.csv",
    "HPS-v2":      "scores_hps_v2.csv",
}


def main() -> None:
    out_dir = OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for name, fname in SCORE_FILES.items():
        df = pd.read_csv(REPO_ROOT / "data" / "scores" / fname)
        assert len(df) == 2610 and df.score.isna().sum() == 0, fname
        df["scrape_rank"] = df["filename"].str.extract(r"image_(\d+)\.jpg")[0].astype(int)
        assert df["scrape_rank"].notna().all(), f"{fname}: unparseable filenames"
        frames[name] = df

    rows = []
    for name, df in frames.items():
        for sub in SUBCULTURES:
            g = df[df.subculture == sub]
            rho, p = stats.spearmanr(g["scrape_rank"], g["score"])
            rows.append({"scorer": name, "subculture": sub, "n": len(g),
                         "spearman_rho": round(rho, 3), "p": p})
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "order_rank_correlations.csv", index=False)

    piv = res.pivot(index="subculture", columns="scorer", values="spearman_rho").loc[
        SUBCULTURES, list(SCORE_FILES)]
    laion = res[res.scorer == "LAION-A V2"]
    abs_max = laion.spearman_rho.abs().max()
    abs_mean = laion.spearman_rho.abs().mean()
    n_sig = int((laion.p < 0.05).sum())

    md = ["| subculture | n | LAION-A rho | p |", "|---|---|---|---|"]
    for _, r in laion.iterrows():
        md.append(f"| {r.subculture.replace('_', ' ')} | {r.n} | {r.spearman_rho:+.3f} | {r.p:.3g} |")

    report = "\n".join([
        f"Scrape-order check (corpus independence) — generated {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "Proxy: filename number N (scrape order, monotone in Lexica cursor).",
        "Caveats: proxy assumes search order ~ ranking; combined scrape sessions add",
        "noise that biases rho toward zero.",
        "",
        "=== HEADLINE (LAION-A vs scrape rank, within subculture) ===",
        f"mean |rho| = {abs_mean:.3f}   max |rho| = {abs_max:.3f}   "
        f"significant at p<0.05: {n_sig}/6 subcultures",
        "",
        "=== Spearman rho, all scorers (original conditioning) ===",
        piv.to_string(), "",
        "=== markdown table (LAION-A) ===", *md, "",
    ])
    (out_dir / "order_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
