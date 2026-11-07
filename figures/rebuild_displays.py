"""Rebuild Table 3, the text-input agreement table and Figure 1 from the released scores.

Table 3 and Figure 1 use the fixed text "an image" for the three
text-conditioned predictors (LAION-Aesthetics V2 is vision-only and identical
under every text input). The agreement table reports, for each text input, the
agreement between the predictors' six-subculture orderings.

Read-only on the dataset. Writes to ``output/`` and refuses to overwrite.
Run after ``results/laion_profile/cross_scorer_analysis.py`` and
``results/matched_text/analyze_matched_text.py --arm {empty,const,label}``:

    python3 figures/rebuild_displays.py

Checks performed (all must pass, otherwise the script stops):
  * the original-prompt arm reproduces ``output/rq2b_*`` (means, ranks, eta^2,
    Spearman matrix);
  * per-subculture means, ranks, eta^2 and Spearman matrices for each matched
    text input reproduce ``output/matched_<arm>_*``;
  * the inter-peer / LAION-peer means match the values reported in the paper.
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

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = DATASET
from _provenance import write_manifest  # noqa: E402

OUT = OUTPUT
SCORES = ROOT / "data" / "scores"
MATCHED = ROOT / "matched_text"
RQ2B = OUTPUT
ARM_FILE = {"empty": "empty", "const": "an_image", "label": "label"}

SUBS = ["cottagecore", "cyberpunk", "dark_academia", "goblincore", "synthwave", "y2k"]
SUB_LABEL = {"cottagecore": "Cottagecore", "cyberpunk": "Cyberpunk", "dark_academia": "Dark Academia",
             "goblincore": "Goblincore", "synthwave": "Synthwave", "y2k": "Y2K"}
SCORERS = ["LAION-A V2", "PickScore", "ImageReward", "HPS-v2"]
PEERS = SCORERS[1:]
FILE_STEM = {"PickScore": "scores_pickscore", "ImageReward": "scores_image_reward", "HPS-v2": "scores_hps_v2"}
ARMS = {"orig": "Original prompts", "empty": "Empty string", "const": "\u201can image\u201d", "label": "Subculture label"}
ARM_SHORT = {"orig": "orig", "empty": "empty", "const": "const", "label": "label"}


def fail(msg: str) -> None:
    raise SystemExit("CHECK FAILED: " + msg)


def load_arm(arm: str) -> pd.DataFrame:
    """One row per image with the four scorers' scores for the given arm."""
    laion = pd.read_csv(SCORES / "scores_laion_v2.csv")[["index", "filename", "subculture", "relative_path", "score"]]
    df = laion.rename(columns={"score": "LAION-A V2"})
    for peer in PEERS:
        if arm == "orig":
            path = SCORES / f"{FILE_STEM[peer]}.csv"
        else:
            path = MATCHED / f"{FILE_STEM[peer]}__{ARM_FILE[arm]}.csv"
        p = pd.read_csv(path)[["index", "filename", "subculture", "score"]].rename(columns={"score": peer})
        df = df.merge(p, on=["index", "filename", "subculture"], how="inner", validate="one_to_one")
    if len(df) != 2610:
        fail(f"{arm}: expected 2,610 joined rows, got {len(df)}")
    return df


def summarise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    means = df.groupby("subculture")[SCORERS].mean().loc[SUBS]
    ranks = means.rank(method="min").astype(int)  # 1 = lowest mean
    eta = {}
    for s in SCORERS:
        groups = [g[s].to_numpy() for _, g in df.groupby("subculture")]
        grand = df[s].mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_total = ((df[s] - grand) ** 2).sum()
        eta[s] = ss_between / ss_total
    eta = pd.Series(eta)
    rho = pd.DataFrame(index=SCORERS, columns=SCORERS, dtype=float)
    for a in SCORERS:
        for b in SCORERS:
            rho.loc[a, b] = 1.0 if a == b else stats.spearmanr(ranks[a], ranks[b]).statistic
    return means, ranks, eta, rho


def inputs_for(arm: str) -> list[Path]:
    if arm == "orig":
        return [SCORES / f"{FILE_STEM[p]}.csv" for p in PEERS]
    return [MATCHED / f"{FILE_STEM[p]}__{ARM_FILE[arm]}.csv" for p in PEERS]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table3": OUT / "table3_an_image.csv",
        "cond": OUT / "text_input_agreement.csv",
        "rho": OUT / "spearman_by_text_input.csv",
        "cells": OUT / "fig1_cells.csv",
        "fig": OUT / "fig1_an_image.png",
        "report": OUT / "displays_report.md",
        "manifest": OUT / "displays_manifest.json",
    }
    for p in outputs.values():
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")

    checks: dict[str, str] = {}
    arms: dict[str, dict] = {}
    for arm in ARMS:
        df = load_arm(arm)
        means, ranks, eta, rho = summarise(df)
        arms[arm] = {"df": df, "means": means, "ranks": ranks, "eta": eta, "rho": rho}

    # ---- Check 1: original arm reproduces the rq2b tables -------------------------------
    ref_means = pd.read_csv(RQ2B / "rq2b_per_subculture_means.csv", index_col=0).loc[SUBS]
    ref_ranks = pd.read_csv(RQ2B / "rq2b_per_subculture_ranks.csv", index_col=0).loc[SUBS]
    ref_eta = pd.read_csv(RQ2B / "rq2b_per_scorer_anova.csv", index_col=0)["eta_sq"]
    ref_rho = pd.read_csv(RQ2B / "rq2b_spearman_subculture_ranks.csv", index_col=0)
    d_means = float((arms["orig"]["means"] - ref_means[SCORERS]).abs().max().max())
    d_eta = float((arms["orig"]["eta"] - ref_eta[SCORERS]).abs().max())
    d_rho = float((arms["orig"]["rho"].astype(float) - ref_rho.loc[SCORERS, SCORERS].astype(float)).abs().max().max())
    if d_means > 1e-9 or d_eta > 1e-9 or not arms["orig"]["ranks"].equals(ref_ranks[SCORERS].astype(int)) or d_rho > 1e-3:
        fail(f"orig arm vs rq2b: means {d_means}, eta {d_eta}, rho {d_rho}")
    checks["orig_vs_rq2b"] = f"means max|diff| {d_means:.1e}; eta max|diff| {d_eta:.1e}; ranks equal; rho max|diff| {d_rho:.1e} (rq2b rho has 3 decimals)"

    # ---- Check 2: matched arms reproduce the matched-text tables -------------------------------
    for arm in ("empty", "const", "label"):
        m = pd.read_csv(OUTPUT / f"matched_{arm}_per_subculture_means.csv", index_col=0).loc[SUBS]
        r = pd.read_csv(OUTPUT / f"matched_{arm}_per_subculture_ranks.csv", index_col=0).loc[SUBS]
        e = pd.read_csv(OUTPUT / f"matched_{arm}_anova.csv", index_col=0)["eta_sq"]
        s = pd.read_csv(OUTPUT / f"matched_{arm}_spearman_matrix.csv", index_col=0)
        dm = float((arms[arm]["means"] - m[SCORERS]).abs().max().max())
        de = float((arms[arm]["eta"] - e[SCORERS]).abs().max())
        ds = float((arms[arm]["rho"].astype(float) - s.loc[SCORERS, SCORERS].astype(float)).abs().max().max())
        if dm > 1e-9 or de > 1e-9 or not arms[arm]["ranks"].equals(r[SCORERS].astype(int)) or ds > 1e-3:
            fail(f"{arm} arm vs matched files: means {dm}, eta {de}, rho {ds}")
        checks[f"{arm}_vs_matched"] = f"means max|diff| {dm:.1e}; eta max|diff| {de:.1e}; ranks equal; rho max|diff| {ds:.1e}"

    # ---- Table 3: "an image" for the three predictors --------------------------------------
    const = arms["const"]
    rows = []
    for sub in SUBS:
        row = {"subculture": SUB_LABEL[sub]}
        for s in SCORERS:
            row[f"{s} mean"] = const["means"].loc[sub, s]
            row[f"{s} rank"] = int(const["ranks"].loc[sub, s])
        rows.append(row)
    eta_row = {"subculture": "eta_sq"}
    for s in SCORERS:
        eta_row[f"{s} mean"] = const["eta"][s]
        eta_row[f"{s} rank"] = ""
    rows.append(eta_row)
    pd.DataFrame(rows).to_csv(outputs["table3"], index=False)

    # ---- Text-input agreement table ---------------------------------------------
    summary_ref = {"orig": (0.695, 0.30), "empty": (0.352, 0.505), "const": (0.333, 0.448), "label": (0.829, 0.467)}
    cond_rows = []
    rho_rows = []
    for arm, label in ARMS.items():
        rho = arms[arm]["rho"].astype(float)
        inter = np.mean([rho.loc["PickScore", "ImageReward"], rho.loc["PickScore", "HPS-v2"], rho.loc["ImageReward", "HPS-v2"]])
        lp = np.mean([rho.loc["LAION-A V2", p] for p in PEERS])
        ra, rb = summary_ref[arm]
        if abs(inter - ra) > 0.006 or abs(lp - rb) > 0.006:
            fail(f"{arm}: inter-peer {inter:.3f} vs summary {ra}; LAION-peer {lp:.3f} vs {rb}")
        ranks = arms[arm]["ranks"]
        top = {s: SUB_LABEL[ranks[s].idxmax()] for s in SCORERS}
        y2k = {s: int(ranks.loc["y2k", s]) for s in SCORERS}
        eta = arms[arm]["eta"]
        cond_rows.append({
            "arm": arm, "text input": label,
            "mean inter-peer rho": round(inter, 3), "mean LAION-peer rho": round(lp, 3),
            "top subculture (LAION/Pick/IR/HPS)": " / ".join(top[s] for s in SCORERS),
            "Y2K rank (LAION/Pick/IR/HPS)": " / ".join(str(y2k[s]) for s in SCORERS),
            "n scorers ranking Y2K lowest": sum(v == 1 for v in y2k.values()),
            "n scorers ranking Cyberpunk highest": sum(v == "Cyberpunk" for v in top.values()),
            "eta_sq PickScore": round(eta["PickScore"], 3), "eta_sq ImageReward": round(eta["ImageReward"], 3),
            "eta_sq HPS-v2": round(eta["HPS-v2"], 3), "eta_sq LAION-A V2": round(eta["LAION-A V2"], 3),
        })
        for a in SCORERS:
            for b in SCORERS:
                if SCORERS.index(a) < SCORERS.index(b):
                    rho_rows.append({"arm": arm, "scorer_a": a, "scorer_b": b, "rho": round(rho.loc[a, b], 3)})
    checks["cross_arm_summary"] = "inter-peer and LAION-peer means within 0.006 of the paper's values for all four arms"
    released = {"const": "an_image"}
    pd.DataFrame(cond_rows).replace({"arm": released}).to_csv(outputs["cond"], index=False)
    pd.DataFrame(rho_rows).replace({"arm": released}).to_csv(outputs["rho"], index=False)

    # ---- Figure 1: representative image per cell ------------------------------------------
    df = const["df"]
    cells = []
    for s in SCORERS:
        for sub in SUBS:
            g = df[df["subculture"] == sub]
            mu = const["means"].loc[sub, s]
            i = (g[s] - mu).abs().idxmin()
            cells.append({"scorer": s, "arm": "vision-only" if s == "LAION-A V2" else "an_image",
                          "subculture": sub, "mean": mu, "rank": int(const["ranks"].loc[sub, s]),
                          "image": g.loc[i, "relative_path"], "image_score": g.loc[i, s],
                          "abs_diff": abs(g.loc[i, s] - mu)})
    cells_df = pd.DataFrame(cells)
    cells_df.to_csv(outputs["cells"], index=False)
    draw_figure(cells_df, outputs["fig"])

    # ---- Report ----------------------------------------------------------------------------
    fmt = {"LAION-A V2": "{:.2f}", "PickScore": "{:.2f}", "ImageReward": "{:+.2f}", "HPS-v2": "{:.3f}"}
    lines = ["# Displays rebuilt from the released scores", "",
             "Table 3 under the constant text \u201can image\u201d (LAION-Aesthetics V2 is vision-only). Rank 1 = lowest mean.", "",
             "| Subculture | " + " | ".join(f"{s} mean | rank" for s in SCORERS) + " |",
             "|---|" + "---|---|" * 4]
    for sub in SUBS:
        lines.append(f"| {SUB_LABEL[sub]} | " + " | ".join(
            f"{fmt[s].format(const['means'].loc[sub, s])} | {int(const['ranks'].loc[sub, s])}" for s in SCORERS) + " |")
    lines.append("| eta^2 | " + " | ".join(f"{const['eta'][s]:.3f} | " for s in SCORERS) + " |")
    lines += ["", "Conditioning table (agreement = Spearman rho between six-subculture orderings):", "",
              "| Text input | inter-peer rho | LAION-peer rho | top (LAION/Pick/IR/HPS) | Y2K rank (LAION/Pick/IR/HPS) | eta^2 Pick/IR/HPS |", "|---|---|---|---|---|---|"]
    for r in cond_rows:
        lines.append(f"| {r['text input']} | {r['mean inter-peer rho']:.2f} | {r['mean LAION-peer rho']:.2f} | {r['top subculture (LAION/Pick/IR/HPS)']} | "
                     f"{r['Y2K rank (LAION/Pick/IR/HPS)']} | {r['eta_sq PickScore']:.3f} / {r['eta_sq ImageReward']:.3f} / {r['eta_sq HPS-v2']:.3f} |")
    lines += ["", "Checks:", ""] + [f"- {k}: {v}" for k, v in checks.items()] + ["",
              "Figure 1 cells: one representative image per scorer and subculture, the image whose score is closest to the subculture mean "
              f"(`{outputs['cells'].name}`)."]
    outputs["report"].write_text("\n".join(lines) + "\n")

    inputs = [SCORES / "scores_laion_v2.csv"] + sum((inputs_for(a) for a in ARMS), []) + [
        RQ2B / "rq2b_per_subculture_means.csv", RQ2B / "rq2b_per_subculture_ranks.csv",
        RQ2B / "rq2b_per_scorer_anova.csv", RQ2B / "rq2b_spearman_subculture_ranks.csv"]
    for arm in ("empty", "const", "label"):
        inputs += [OUTPUT / f"matched_{arm}_{k}.csv" for k in ("per_subculture_means", "per_subculture_ranks", "anova", "spearman_matrix")]
    write_manifest(outputs["manifest"], script=Path(__file__), purpose=__doc__.strip().splitlines()[0],
                   inputs=inputs, outputs=[p for k, p in outputs.items() if k != "manifest"],
                   parameters={"table3_arm": "const (\u201can image\u201d)", "rank_convention": "1 = lowest mean",
                               "figure_cell_rule": "image with score closest to the subculture mean, same arm as the column"},
                   checks=checks)
    print("\n".join(lines))


def draw_figure(cells: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    fmt = {"LAION-A V2": "{:.2f}", "PickScore": "{:.2f}", "ImageReward": "{:+.2f}", "HPS-v2": "{:.3f}"}
    row_label = {"LAION-A V2": "LAION-A V2\n(vision-only)", "PickScore": "PickScore\n(\u201can image\u201d)",
                 "ImageReward": "ImageReward\n(\u201can image\u201d)", "HPS-v2": "HPS-v2\n(\u201can image\u201d)"}
    fig, axes = plt.subplots(4, 6, figsize=(12.4, 9.4))
    plt.subplots_adjust(left=0.11, right=0.995, top=0.86, bottom=0.085, wspace=0.08, hspace=0.42)
    for i, s in enumerate(SCORERS):
        for j, sub in enumerate(SUBS):
            ax = axes[i, j]
            c = cells[(cells.scorer == s) & (cells.subculture == sub)].iloc[0]
            im = Image.open(ROOT / c["image"]).convert("RGB")
            w, h = im.size
            side = min(w, h)
            im = im.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side)).resize((256, 256))
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            edge = None
            if c["rank"] == 6:
                edge, tag = "#1f8f3a", "highest in row"
            elif c["rank"] == 1:
                edge, tag = "#c0392b", "lowest in row"
            if edge:
                ax.add_patch(Rectangle((0, 0), 255, 255, fill=False, lw=4, ec=edge, transform=ax.transData, clip_on=False))
            label = fmt[s].format(c["mean"])
            ax.set_title(label + (f"\n{tag}" if edge else ""), fontsize=11, pad=4,
                         color=edge or "black", fontweight="bold" if edge else "normal")
            if j == 0:
                ax.text(-0.12, 0.5, row_label[s], transform=ax.transAxes, ha="right", va="center", fontsize=12)
    # Column headers at one fixed height, above the tallest title of the first row.
    top_edge = max(ax.get_position().y1 for ax in axes[0])
    for j, sub in enumerate(SUBS):
        pos = axes[0, j].get_position()
        fig.text((pos.x0 + pos.x1) / 2, top_edge + 0.065, SUB_LABEL[sub], ha="center", va="bottom",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.965, "Per-subculture mean score of four aesthetic predictors, SubCulture-2.6K (N = 2,610)",
             ha="center", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.012, "PickScore, ImageReward and HPS-v2 scored with the constant text \u201can image\u201d; LAION-A V2 reads pixels only.\n"
             "Each cell shows the image whose score is closest to the subculture mean. Green = highest mean in the row, red = lowest.",
             ha="center", va="bottom", fontsize=10, linespacing=1.5)
    fig.savefig(out, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
