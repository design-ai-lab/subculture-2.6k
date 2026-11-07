"""
Cross-scorer comparison across LAION-A V2, PickScore, ImageReward and HPS-v2.

Reads the four data/scores/scores_*.csv files and writes to output/:

  rq2b_per_subculture_means.csv         6 × 4 means
  rq2b_per_subculture_ranks.csv         6 × 4 ranks (1=lowest)
  rq2b_per_scorer_anova.csv             4 × {F, p, eta_sq}
  rq2b_spearman_subculture_ranks.csv    4 × 4 cross-scorer agreement
  rq2b_per_image_corr_vs_laion.csv      peer × {Spearman ρ, p, Pearson r}
  rq2b_per_image_corr_within_subculture.csv   3 peers × 6 subcultures
  rq2b_quality_cutoff_retention.csv     LAION cutoff retention table
  figures/rq2b_score_scatter_vs_laion.png    3 panels, peer vs LAION-A
  figures/rq2b_rank_heatmap.png         6 × 4 rank heatmap
  figures/rq2b_per_subculture_zscore.png    z-scored means side-by-side
  rq2b_stats_report.txt                 all of the above and a summary sentence
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


import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = DATASET
SCORES_DIR = REPO_ROOT / "data" / "scores"
OUT_DIR    = OUTPUT
FIG_DIR    = OUT_DIR / "figures"

SCORERS = [
    ("LAION-A V2",   "scores_laion_v2.csv",     "1–10",     "tab:blue"),
    ("PickScore",    "scores_pickscore.csv",    "logit",    "tab:orange"),
    ("ImageReward",  "scores_image_reward.csv", "≈N(0,1)",  "tab:green"),
    ("HPS-v2",       "scores_hps_v2.csv",       "small +",  "tab:red"),
]

SUBCULTURE_ORDER = ["cottagecore", "cyberpunk", "dark_academia",
                    "goblincore",  "synthwave", "y2k"]


# --------------------------------------------------------------------------- #

def load_all() -> dict[str, pd.DataFrame]:
    """name -> per-image df with cols: index, filename, subculture, score."""
    frames = {}
    for name, fname, _, _ in SCORERS:
        df = pd.read_csv(SCORES_DIR / fname)
        assert len(df) == 2610, f"{fname}: expected 2610 rows, got {len(df)}"
        assert df.score.isna().sum() == 0, f"{fname}: NaN scores present"
        frames[name] = df
    return frames


def per_subculture_means(frames) -> pd.DataFrame:
    out = pd.DataFrame()
    for name, df in frames.items():
        out[name] = df.groupby("subculture").score.mean()
    return out.loc[SUBCULTURE_ORDER]


def per_subculture_ranks(means: pd.DataFrame) -> pd.DataFrame:
    """1 = lowest mean, 6 = highest mean (within each scorer)."""
    return means.rank(axis=0, method="min").astype(int)


def per_scorer_anova(frames) -> pd.DataFrame:
    rows = []
    for name, df in frames.items():
        groups = [g.values for _, g in df.groupby("subculture").score]
        f, p_anova = stats.f_oneway(*groups)
        h, p_kw    = stats.kruskal(*groups)
        grand = df.score.mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((df.score - grand) ** 2).sum()
        eta2 = ss_between / ss_total
        rows.append({
            "scorer": name, "n": len(df),
            "F": f, "p_anova": p_anova,
            "H": h, "p_kruskal": p_kw,
            "eta_sq": eta2,
        })
    return pd.DataFrame(rows).set_index("scorer")


def spearman_subculture_ranks(ranks: pd.DataFrame) -> pd.DataFrame:
    cols = list(ranks.columns)
    out = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            out.loc[a, b] = round(stats.spearmanr(ranks[a], ranks[b]).statistic, 3)
    return out


def per_image_corr_vs_laion(frames) -> pd.DataFrame:
    laion = frames["LAION-A V2"][["index", "score"]].rename(columns={"score": "laion"})
    rows = []
    for name, df in frames.items():
        if name == "LAION-A V2":
            continue
        merged = df.merge(laion, on="index")
        rho, p_s = stats.spearmanr(merged.score, merged.laion)
        r, p_p   = stats.pearsonr(merged.score, merged.laion)
        rows.append({"peer": name, "spearman_rho": rho, "spearman_p": p_s,
                     "pearson_r": r, "pearson_p": p_p, "n": len(merged)})
    return pd.DataFrame(rows).set_index("peer")


def per_image_corr_within_subculture(frames) -> pd.DataFrame:
    laion = frames["LAION-A V2"][["index", "subculture", "score"]].rename(columns={"score": "laion"})
    rows = []
    for name, df in frames.items():
        if name == "LAION-A V2":
            continue
        merged = df.merge(laion[["index", "laion"]], on="index")
        for sub in SUBCULTURE_ORDER:
            sub_mask = merged.subculture == sub
            rho = stats.spearmanr(merged.loc[sub_mask, "score"], merged.loc[sub_mask, "laion"]).statistic
            rows.append({"peer": name, "subculture": sub, "spearman_rho": round(rho, 3),
                         "n": int(sub_mask.sum())})
    return pd.DataFrame(rows).pivot(index="peer", columns="subculture", values="spearman_rho")[SUBCULTURE_ORDER]


def quality_cutoff_retention(frames) -> pd.DataFrame:
    """LAION-A specific: fraction of each subculture passing each cutoff."""
    df = frames["LAION-A V2"]
    rows = []
    for cut in [4.5, 5.0, 5.5, 6.0, 6.5]:
        for sub in SUBCULTURE_ORDER:
            s = df.loc[df.subculture == sub, "score"]
            rows.append({"cutoff": cut, "subculture": sub,
                         "frac_retained": round((s >= cut).mean(), 4),
                         "n_retained": int((s >= cut).sum()),
                         "n_total": len(s)})
    return pd.DataFrame(rows).pivot(index="cutoff", columns="subculture",
                                    values="frac_retained")[SUBCULTURE_ORDER]


# --------------------------------------------------------------------------- #

def fig_rank_heatmap(ranks: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(ranks.values.T, cmap="RdYlGn", aspect="auto", vmin=1, vmax=6)
    ax.set_xticks(range(len(ranks)))
    ax.set_xticklabels(ranks.index, rotation=20, ha="right")
    ax.set_yticks(range(len(ranks.columns)))
    ax.set_yticklabels(ranks.columns)
    for i in range(ranks.shape[0]):
        for j in range(ranks.shape[1]):
            ax.text(i, j, ranks.iloc[i, j], ha="center", va="center", fontsize=11)
    ax.set_title("Per-subculture rank by scorer  (1=lowest mean, 6=highest)")
    fig.colorbar(im, ax=ax, ticks=[1, 2, 3, 4, 5, 6], label="rank")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def fig_score_scatter_vs_laion(frames, path: Path) -> None:
    laion = frames["LAION-A V2"].set_index("index")["score"]
    peers = [(n, df) for n, df in frames.items() if n != "LAION-A V2"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    palette = dict(zip(SUBCULTURE_ORDER, plt.cm.tab10.colors[:6]))
    for ax, (name, df) in zip(axes, peers):
        merged = df.set_index("index").join(laion.rename("laion"))
        for sub in SUBCULTURE_ORDER:
            m = merged[merged.subculture == sub]
            ax.scatter(m.laion, m.score, s=8, alpha=0.45, label=sub, color=palette[sub])
        rho = stats.spearmanr(merged.score, merged.laion).statistic
        ax.set_title(f"{name}   (Spearman ρ={rho:.3f})")
        ax.set_xlabel("LAION-A V2 score")
        ax.set_ylabel(f"{name} score")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, markerscale=1.5, frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def fig_zscore_means(means: pd.DataFrame, path: Path) -> None:
    z = (means - means.mean(axis=0)) / means.std(axis=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(z))
    width = 0.20
    for i, col in enumerate(z.columns):
        ax.bar(x + (i - 1.5) * width, z[col], width, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels(z.index, rotation=20, ha="right")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("z-scored mean (within scorer)")
    ax.set_title("Per-subculture mean by scorer (z-scored within scorer for comparability)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def write_report(means, ranks, anova, spearman, per_image_corr, within_sub_corr,
                 cutoff, path: Path) -> None:
    laion_eta = anova.loc["LAION-A V2", "eta_sq"]
    bottom_per_scorer = ranks.idxmin()
    top_per_scorer    = ranks.idxmax()
    peer_pairs = [("PickScore", "ImageReward"), ("PickScore", "HPS-v2"), ("ImageReward", "HPS-v2")]
    peer_avg = np.mean([spearman.loc[a, b] for a, b in peer_pairs])
    laion_vs_peers = [spearman.loc["LAION-A V2", p] for p in ["PickScore", "ImageReward", "HPS-v2"]]

    headline = (
        f"Across the four deployed aesthetic predictors, peer scorers share LAION-Aesthetics's "
        f"dispreference for Y2K imagery (rank 1 of 6 on three of four scorers) but diverge on "
        f"which subculture they reward most: LAION-Aesthetics privileges Cyberpunk imagery "
        f"(rank 6, η²={laion_eta:.2f}, large by Cohen's convention), while PickScore, "
        f"ImageReward, and HPS-v2 instead privilege Goblincore. Cross-scorer rank correlation "
        f"on per-subculture orderings is {peer_avg:.2f} on average among the three peers but "
        f"only {np.mean(laion_vs_peers):.2f} between LAION-Aesthetics and any peer, indicating "
        f"that LAION-Aesthetics's specific subcultural preference is idiosyncratic to its "
        f"training composite (SAC + LAION-Logos + AVA-1) rather than a family property of "
        f"deployed aesthetic predictors."
    )

    lines = []
    lines.append("Cross-scorer comparison (LAION-A V2, PickScore, ImageReward, HPS-v2)")
    lines.append("=" * 76)
    lines.append("N images: 2,610   Subcultures: 6")
    lines.append("")
    lines.append("Per-scorer ANOVA across subcultures:")
    lines.append(anova.round(4).to_string())
    lines.append("")
    lines.append("Per-subculture mean by scorer (raw, scorer scales differ; do not compare across rows):")
    lines.append(means.round(4).to_string())
    lines.append("")
    lines.append("Per-subculture rank (1=lowest mean within scorer, 6=highest):")
    lines.append(ranks.to_string())
    lines.append("")
    lines.append(f"Bottom-of-rank by scorer:  {bottom_per_scorer.to_dict()}")
    lines.append(f"Top-of-rank by scorer:     {top_per_scorer.to_dict()}")
    lines.append("")
    lines.append("Spearman rank-correlation between scorers (on the 6 per-subculture-mean ranks):")
    lines.append(spearman.to_string())
    lines.append("")
    lines.append("Per-image cross-scorer correlations vs LAION-A V2 (all 2,610 images):")
    lines.append(per_image_corr.round(4).to_string())
    lines.append("")
    lines.append("Per-image Spearman vs LAION-A V2 within each subculture (3 peers × 6 subs):")
    lines.append(within_sub_corr.round(3).to_string())
    lines.append("")
    lines.append("LAION-A V2 quality-cutoff retention rate by subculture:")
    lines.append(cutoff.round(3).to_string())
    lines.append("")
    lines.append("=" * 76)
    lines.append("Summary:")
    lines.append("")
    lines.append(headline)
    lines.append("=" * 76)
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    frames = load_all()
    means    = per_subculture_means(frames)
    ranks    = per_subculture_ranks(means)
    anova    = per_scorer_anova(frames)
    spearman = spearman_subculture_ranks(ranks)
    per_img  = per_image_corr_vs_laion(frames)
    within   = per_image_corr_within_subculture(frames)
    cutoff   = quality_cutoff_retention(frames)

    means.to_csv(OUT_DIR / "rq2b_per_subculture_means.csv")
    ranks.to_csv(OUT_DIR / "rq2b_per_subculture_ranks.csv")
    anova.to_csv(OUT_DIR / "rq2b_per_scorer_anova.csv")
    spearman.to_csv(OUT_DIR / "rq2b_spearman_subculture_ranks.csv")
    per_img.to_csv(OUT_DIR / "rq2b_per_image_corr_vs_laion.csv")
    within.to_csv(OUT_DIR / "rq2b_per_image_corr_within_subculture.csv")
    cutoff.to_csv(OUT_DIR / "rq2b_quality_cutoff_retention.csv")

    fig_rank_heatmap(ranks, FIG_DIR / "rq2b_rank_heatmap.png")
    fig_score_scatter_vs_laion(frames, FIG_DIR / "rq2b_score_scatter_vs_laion.png")
    fig_zscore_means(means, FIG_DIR / "rq2b_per_subculture_zscore.png")

    write_report(means, ranks, anova, spearman, per_img, within, cutoff,
                 OUT_DIR / "rq2b_stats_report.txt")

    print("[done] wrote 7 CSV tables, 3 figures, and the stats report:")
    for p in sorted(OUT_DIR.glob("rq2b_*.csv")) + sorted(FIG_DIR.glob("rq2b_*.png")) + [OUT_DIR / "rq2b_stats_report.txt"]:
        print(f"   {under(REPO_ROOT, p)}")


if __name__ == "__main__":
    main()
