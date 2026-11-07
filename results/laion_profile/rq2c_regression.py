"""
Interpretable-feature regression (Table 4).

Asks: which interpretable image properties predict LAION-A V2's score? And —
because the multi-scorer audit changed the headline — does the *same* feature
set predict each peer scorer's score, or do the scorers attend to different
axes of the image?

Pipeline
--------
1. Linear regression with standardized coefficients (every feature on the same
   z-scale so coefficients are directly comparable). Reported per scorer.
2. HistGradientBoostingRegressor + sklearn permutation importance for non-linear
   robustness — also per scorer.
3. Within-LAION-A regression run independently inside each subculture, so we
   can ask whether the dominant axes generalise across subcultures or whether
   the Y2K penalty is driven by a Y2K-specific interaction.
4. Residual-eta-squared ANOVA: standardize-residualise the LAION-A score
   against the top-K features, rerun the one-way subculture ANOVA on residuals.
   Reports residual eta² alongside raw eta². Drop = mediation share.

Outputs (under output/)
  rq2c_linear_coefs.csv                4 scorers × 13 standardized coefficients + R^2
  rq2c_permutation_importance.csv      4 scorers × 13 features (mean importance)
  rq2c_within_subculture_coefs.csv     6 subcultures × 13 standardized coefs (LAION-A)
  rq2c_residual_anova.csv              raw vs residual eta^2 across top-K choices
  rq2c_stats_report.txt                all of the above and a summary
  figures/rq2c_top_axes_per_scorer.png horizontal bar plot of |std-coef| or perm-imp
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


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


REPO_ROOT = DATASET
SCORES_DIR = REPO_ROOT / "data" / "scores"
FEATS_PATH = REPO_ROOT / "data" / "interpretable_features.csv"
OUT_DIR    = OUTPUT
FIG_DIR    = OUT_DIR / "figures"

FEATURES = [
    "sat_mean", "sat_std", "lum_mean", "lum_std",
    "colorfulness", "palette_diversity", "palette_top_pct",
    "laplacian_var", "edge_density", "local_contrast_std", "bytes_per_pixel",
    "aspect_ratio", "face_count",
]
SCORERS = [
    ("LAION-A V2",  "scores_laion_v2.csv"),
    ("PickScore",   "scores_pickscore.csv"),
    ("ImageReward", "scores_image_reward.csv"),
    ("HPS-v2",      "scores_hps_v2.csv"),
]
SUBCULTURE_ORDER = ["cottagecore", "cyberpunk", "dark_academia",
                    "goblincore",  "synthwave", "y2k"]


# --------------------------------------------------------------------------- #

def load_joined() -> pd.DataFrame:
    feats = pd.read_csv(FEATS_PATH)
    df = feats[["index", "filename", "subculture"] + FEATURES].copy()
    for name, fname in SCORERS:
        s = pd.read_csv(SCORES_DIR / fname)[["index", "score"]].rename(columns={"score": name})
        df = df.merge(s, on="index")
    return df


def standardized_linear(df: pd.DataFrame, target: str) -> tuple[pd.Series, float]:
    X = df[FEATURES].values
    y = df[target].values
    Xs = StandardScaler().fit_transform(X)
    ys = (y - y.mean()) / y.std()
    model = LinearRegression().fit(Xs, ys)
    coefs = pd.Series(model.coef_, index=FEATURES)
    r2 = model.score(Xs, ys)
    return coefs, r2


def permutation_imp(df: pd.DataFrame, target: str, n_repeats: int = 6,
                    seed: int = 0) -> pd.Series:
    X = df[FEATURES].values
    y = df[target].values
    model = HistGradientBoostingRegressor(max_iter=300, random_state=seed).fit(X, y)
    imp = permutation_importance(model, X, y, n_repeats=n_repeats,
                                 random_state=seed, n_jobs=-1)
    return pd.Series(imp.importances_mean, index=FEATURES)


def within_subculture_linear(df: pd.DataFrame, target: str = "LAION-A V2") -> pd.DataFrame:
    rows = {}
    for sub in SUBCULTURE_ORDER:
        sub_df = df[df.subculture == sub]
        coefs, r2 = standardized_linear(sub_df, target)
        coefs.loc["__R2__"] = r2
        coefs.loc["__n__"]  = len(sub_df)
        rows[sub] = coefs
    return pd.DataFrame(rows).T[FEATURES + ["__R2__", "__n__"]]


def residual_eta_squared(df: pd.DataFrame, target: str, top_k_list=(2, 4, 6, 13)) -> pd.DataFrame:
    """Partial out top-K standardized features from `target`, recompute eta²."""
    coefs, _ = standardized_linear(df, target)
    ranking = coefs.abs().sort_values(ascending=False).index.tolist()
    grand_groups = [g.values for _, g in df.groupby("subculture")[target]]
    grand_mean = df[target].mean()
    ss_between_raw = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in grand_groups)
    ss_total_raw   = ((df[target] - grand_mean) ** 2).sum()
    eta_raw = ss_between_raw / ss_total_raw
    rows = []
    for k in top_k_list:
        cols = ranking[:k]
        Xs = StandardScaler().fit_transform(df[cols].values)
        ys = (df[target].values - df[target].mean()) / df[target].std()
        residual = ys - LinearRegression().fit(Xs, ys).predict(Xs)
        df_res = df.copy(); df_res["__resid__"] = residual
        groups = [g.values for _, g in df_res.groupby("subculture")["__resid__"]]
        gm = df_res["__resid__"].mean()
        ss_b = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
        ss_t = ((df_res["__resid__"] - gm) ** 2).sum()
        eta_res = ss_b / ss_t if ss_t > 0 else 0.0
        rows.append({"top_k": k, "features_partialled": ", ".join(cols),
                     "eta_raw": eta_raw, "eta_residual": eta_res,
                     "mediation_share": (eta_raw - eta_res) / eta_raw})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #

def fig_top_axes_per_scorer(coefs_df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(coefs_df.columns), figsize=(4 * len(coefs_df.columns), 5),
                              sharey=True)
    if len(coefs_df.columns) == 1: axes = [axes]
    for ax, col in zip(axes, coefs_df.columns):
        s = coefs_df[col].sort_values()
        colors = ["tab:red" if v < 0 else "tab:blue" for v in s.values]
        ax.barh(s.index, s.values, color=colors)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_title(f"{col}  (β std)")
        ax.set_xlabel("standardized coefficient")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def write_report(linear, perm_imp, within, resid, path: Path) -> None:
    lines = []
    lines.append("Interpretable-feature regression on the four scorers")
    lines.append("=" * 76)
    lines.append("")
    lines.append("Standardized linear coefficients (β std; positive = higher feature → higher score):")
    lines.append(linear.round(4).to_string())
    lines.append("")
    lines.append("Permutation importance from gradient-boosted regressor (mean Δ R²):")
    lines.append(perm_imp.round(4).to_string())
    lines.append("")

    lines.append("Within-subculture standardized coefficients (LAION-A V2 only):")
    lines.append(within.round(4).to_string())
    lines.append("")

    lines.append("Residual eta² ANOVA — partialling out top-K features then rerunning the subculture ANOVA:")
    lines.append(resid.round(4).to_string(index=False))
    lines.append("")

    # Headline synthesis
    laion_top = linear["LAION-A V2"].abs().sort_values(ascending=False).head(3).index.tolist()
    full_med  = float(resid.iloc[-1]["mediation_share"])
    lines.append("=" * 76)
    lines.append("Summary:")
    lines.append("")
    if full_med >= 0.5:
        verdict = ("largely reducible" if full_med >= 0.75 else "substantially reducible")
        lines.append(
            f"LAION-Aesthetics V2's per-image score is {verdict} to the interpretable "
            f"image properties measured here: {', '.join(laion_top)} are the dominant "
            f"axes (top-3 by standardized coefficient). Partialling all 13 features out "
            f"of the score collapses the cross-subcultural eta² from {resid.iloc[-1]['eta_raw']:.2f} "
            f"to {resid.iloc[-1]['eta_residual']:.2f} (mediation share {full_med*100:.0f}%)."
        )
    else:
        lines.append(
            f"LAION-Aesthetics V2's preference is *not* fully reducible to the interpretable "
            f"axes measured here. The top axes are {', '.join(laion_top)}, but partialling "
            f"all 13 features out of the score only reduces cross-subcultural eta² from "
            f"{resid.iloc[-1]['eta_raw']:.2f} to {resid.iloc[-1]['eta_residual']:.2f} "
            f"(mediation share {full_med*100:.0f}%) — a residual subcultural taste remains."
        )
    lines.append("")
    lines.append("=" * 76)
    path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_joined()
    print(f"[info] joined frame: {len(df)} rows, "
          f"NaN per scorer: {df[[s[0] for s in SCORERS]].isna().sum().to_dict()}")

    # 1. Standardized linear coefficients per scorer
    linear = pd.DataFrame()
    r2_row = {}
    for name, _ in SCORERS:
        coefs, r2 = standardized_linear(df, name)
        linear[name] = coefs
        r2_row[name] = r2
    linear.loc["__R2__"] = pd.Series(r2_row)
    linear.to_csv(OUT_DIR / "rq2c_linear_coefs.csv")
    print("[done] rq2c_linear_coefs.csv")

    # 2. Permutation importance per scorer
    perm = pd.DataFrame()
    for name, _ in SCORERS:
        perm[name] = permutation_imp(df, name)
        print(f"   permutation importance: {name} ✓")
    perm.to_csv(OUT_DIR / "rq2c_permutation_importance.csv")

    # 3. Within-subculture for LAION-A V2
    within = within_subculture_linear(df, "LAION-A V2")
    within.to_csv(OUT_DIR / "rq2c_within_subculture_coefs.csv")
    print("[done] rq2c_within_subculture_coefs.csv")

    # 4. Residual eta² for LAION-A V2
    resid = residual_eta_squared(df, "LAION-A V2")
    resid.to_csv(OUT_DIR / "rq2c_residual_anova.csv", index=False)
    print("[done] rq2c_residual_anova.csv")

    # Figure
    fig_top_axes_per_scorer(linear.drop("__R2__"), FIG_DIR / "rq2c_top_axes_per_scorer.png")
    print("[done] rq2c_top_axes_per_scorer.png")

    write_report(linear, perm, within, resid, OUT_DIR / "rq2c_stats_report.txt")
    print("[done] rq2c_stats_report.txt")


if __name__ == "__main__":
    main()
