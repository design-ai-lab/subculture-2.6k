"""Matched-conditioning analysis of two prompt-conditioned results.

Recomputes, for each conditioning arm, (1) the linear R^2 of each
text-conditioned predictor on the thirteen interpretable features (Section 5.2)
and (2) the per-image Spearman correlation between LAION-Aesthetics V2
and each predictor within each subculture (the within-Y2K LAION/HPS-v2
result in the Discussion).

Arms: "orig" = data/scores (original user prompts);
"empty", "const" ("an image"), "label" = matched text rescoring.

Reuses the paper's regression function unchanged. Reads only; writes to
output/ and refuses to overwrite. Run after rq2c_regression.py (see README.md):
    python3 results/matched_text/matched_arm_axes.py
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
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd
from scipy import stats

REPO_ROOT = DATASET
from rq2c_regression import FEATURES, SUBCULTURE_ORDER, standardized_linear  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import write_manifest  # noqa: E402

FEATS_PATH = REPO_ROOT / "data" / "interpretable_features.csv"
ORIG_DIR = REPO_ROOT / "data" / "scores"
MATCHED_DIR = DATASET / "matched_text"
FROZEN_R2 = OUTPUT / "rq2c_linear_coefs.csv"
FROZEN_WITHIN = OUTPUT / "rq2b_per_image_corr_within_subculture.csv"
OUT_DIR = OUTPUT
PEERS = [("PickScore", "pickscore"), ("ImageReward", "image_reward"), ("HPS-v2", "hps_v2")]
ARMS = ["orig", "empty", "const", "label"]


def score_path(slug: str, arm: str) -> Path:
    if arm == "orig":
        return ORIG_DIR / f"scores_{slug}.csv"
    return MATCHED_DIR / f"scores_{slug}__{'an_image' if arm == 'const' else arm}.csv"


def load_scores(path: Path, name: str) -> pd.DataFrame:
    s = pd.read_csv(path)[["index", "filename", "subculture", "score"]]
    return s.rename(columns={"score": name})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_r2 = OUT_DIR / "matched_arm_r2.csv"
    out_within = OUT_DIR / "matched_arm_within_subculture_rho.csv"
    out_report = OUT_DIR / "matched_arm_report.md"
    out_manifest = OUT_DIR / "matched_arm_manifest.json"
    for p in (out_r2, out_within, out_report, out_manifest):
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
    inputs = [FEATS_PATH, ORIG_DIR / "scores_laion_v2.csv", FROZEN_R2, FROZEN_WITHIN]
    inputs += [score_path(slug, arm) for arm in ARMS for _, slug in PEERS]

    feats = pd.read_csv(FEATS_PATH)[["index", "filename", "subculture"] + FEATURES]
    laion = load_scores(ORIG_DIR / "scores_laion_v2.csv", "LAION-A V2")
    base = feats.merge(laion, on=["index", "filename", "subculture"], validate="one_to_one")
    assert len(base) == 2610, len(base)

    _, laion_r2 = standardized_linear(base, "LAION-A V2")
    r2_rows = [{"arm": "vision-only", "scorer": "LAION-A V2", "r2": laion_r2, "n": len(base)}]
    within_rows = []
    for arm in ARMS:
        for name, slug in PEERS:
            df = base.merge(load_scores(score_path(slug, arm), name),
                            on=["index", "filename", "subculture"], validate="one_to_one")
            assert len(df) == 2610 and df[name].notna().all(), (arm, name)
            _, r2 = standardized_linear(df, name)
            r2_rows.append({"arm": arm, "scorer": name, "r2": r2, "n": len(df)})
            row = {"arm": arm, "scorer": name,
                   "all_images": stats.spearmanr(df[name], df["LAION-A V2"]).statistic}
            for sub in SUBCULTURE_ORDER:
                m = df.subculture == sub
                row[sub] = stats.spearmanr(df.loc[m, name], df.loc[m, "LAION-A V2"]).statistic
            within_rows.append(row)

    r2 = pd.DataFrame(r2_rows)
    within = pd.DataFrame(within_rows)

    frozen_r2 = pd.read_csv(FROZEN_R2, index_col=0).loc["__R2__"]
    frozen_within = pd.read_csv(FROZEN_WITHIN, index_col=0)
    orig_r2 = r2[r2.arm.isin(["orig", "vision-only"])].set_index("scorer").r2
    orig_within = within[within.arm == "orig"].set_index("scorer")[SUBCULTURE_ORDER]
    r2_diff = float((orig_r2 - frozen_r2[orig_r2.index]).abs().max())
    within_diff = float((orig_within - frozen_within.loc[orig_within.index]).abs().max().max())
    checks = {
        "orig_arm_r2_max_abs_diff_vs_rq2c_linear_coefs": r2_diff,
        "orig_arm_within_rho_max_abs_diff_vs_rq2b_within_subculture (rounded to 3 dp)": within_diff,
    }
    assert r2_diff < 1e-9 and within_diff <= 5e-4 + 1e-9, checks

    released = {"const": "an_image"}
    labels = [released.get(a, a) for a in ARMS]
    r2 = r2.replace({"arm": released})
    within = within.replace({"arm": released})
    r2.to_csv(out_r2, index=False)
    within.to_csv(out_within, index=False)

    r2_wide = r2[r2.scorer != "LAION-A V2"].pivot(index="scorer", columns="arm", values="r2")[labels]
    within_y2k = within.pivot(index="scorer", columns="arm", values="y2k")[labels]
    lines = [
        "# Matched-conditioning analysis",
        "",
        "Generated by `matched_arm_axes.py`. Inputs: `data/interpretable_features.csv`,",
        "`data/scores/` (orig arm and LAION-A V2) and `matched_text/` (empty, an_image, label).",
        "Method: `results/laion_profile/rq2c_regression.standardized_linear`; per-image Spearman",
        "within each subculture as in `results/laion_profile/cross_scorer_analysis.py`.",
        "",
        f"LAION-A V2 (vision-only, all arms): R^2 = {laion_r2:.3f}",
        "",
        "Reproduction check: the orig arm matches `output/rq2c_linear_coefs.csv` and",
        f"`output/rq2b_per_image_corr_within_subculture.csv` (R^2 max |diff| = {r2_diff:.1e};",
        f"within-subculture rho max |diff| = {within_diff:.1e}, rq2b values rounded to 3 dp).",
        "",
        "## R^2 on the thirteen interpretable features",
        "",
        r2_wide.round(3).to_markdown(),
        "",
        "## Within-Y2K per-image Spearman with LAION-A V2",
        "",
        within_y2k.round(3).to_markdown(),
        "",
        "## Per-image Spearman with LAION-A V2, all subcultures",
        "",
        within.set_index(["scorer", "arm"])[["all_images"] + SUBCULTURE_ORDER].round(3).to_markdown(),
        "",
    ]
    sd = base.groupby("subculture")["LAION-A V2"].std().reindex(SUBCULTURE_ORDER)
    lines += [
        "## Score spread by subculture (standard deviation)",
        "",
        "Higher spread within a subculture raises correlations by itself.",
        "",
        "| scorer | arm | " + " | ".join(SUBCULTURE_ORDER) + " |",
        "|---|---|" + "---|" * len(SUBCULTURE_ORDER),
        "| LAION-A V2 | vision-only | " + " | ".join(f"{v:.3f}" for v in sd) + " |",
    ]
    for arm in ARMS:
        for name, slug in PEERS:
            s = load_scores(score_path(slug, arm), name).groupby("subculture")[name].std()
            lines.append(f"| {name} | {released.get(arm, arm)} | " + " | ".join(
                f"{s[sub]:.3f}" for sub in SUBCULTURE_ORDER) + " |")
    lines.append("")
    out_report.write_text("\n".join(lines))
    print("\n".join(lines))

    write_manifest(
        out_manifest, script=Path(__file__),
        purpose=("Recompute the R^2 on the interpretable features and the within-subculture per-image Spearman "
                 "with LAION-A V2 for the three text-conditioned predictors under each "
                 "conditioning arm."),
        inputs=inputs, outputs=[out_r2, out_within, out_report],
        parameters={"arms": labels, "features": FEATURES, "n_images": int(len(base)),
                    "regression": "results/laion_profile/rq2c_regression.standardized_linear (unchanged)",
                    "correlation": "scipy.stats.spearmanr, per image"},
        checks=checks,
    )
    print(f"[done] {out_manifest.name}")


if __name__ == "__main__":
    main()
