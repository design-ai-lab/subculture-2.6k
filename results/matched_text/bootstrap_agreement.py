"""Bootstrap intervals for cross-scorer agreement on subculture orderings.

The agreement figures in the conditioning table (matched text) are Spearman rho
between two scorers' rankings of the six per-subculture mean scores, averaged
over the three pairs among PickScore, ImageReward and HPS-v2 ("inter-peer")
or over the three pairs with LAION-Aesthetics V2 ("LAION-peer").

Procedure, per replicate b = 1..B:
  1. Resample images with replacement within each subculture, keeping each
     subculture's size (stratified bootstrap; sizes are fixed by design).
     The same resampled images are used for every scorer and every arm,
     because all scorers scored the same images.
  2. Recompute the six per-subculture means for each scorer and arm.
  3. Rank them within each scorer, compute Spearman rho for the six scorer
     pairs, and average into the inter-peer and LAION-peer figures.
  4. Record differences between arms (paired, same resample).
Report the full-sample value and the 2.5th/97.5th percentiles (percentile
interval). Also records, per scorer and arm, the share of replicates in which
Y2K ranks lowest and Cyberpunk highest.

Scope: the interval reflects which images were sampled, given these six
subcultures and these scorers. It does not cover the choice of subcultures.

Writes to output/ and refuses to overwrite. Run after cross_scorer_analysis.py
and analyze_matched_text.py (see README.md):
    python3 results/matched_text/bootstrap_agreement.py
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
from itertools import combinations
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _provenance import REPO_ROOT, write_manifest  # noqa: E402

ORIG_DIR = REPO_ROOT / "data" / "scores"
MATCHED_DIR = DATASET / "matched_text"
FROZEN_ORIG_MATRIX = OUTPUT / "rq2b_spearman_subculture_ranks.csv"
OUT_DIR = OUTPUT
SEED = 20261021
B = 2000
ARMS = ["orig", "empty", "const", "label"]
SCORERS = [("LAION-A V2", "laion_v2"), ("PickScore", "pickscore"),
           ("ImageReward", "image_reward"), ("HPS-v2", "hps_v2")]
SUBCULTURES = ["cottagecore", "cyberpunk", "dark_academia", "goblincore", "synthwave", "y2k"]
PEER_PAIRS = [(1, 2), (1, 3), (2, 3)]
LAION_PAIRS = [(0, 1), (0, 2), (0, 3)]
DIFFERENCES = [("label", "const"), ("label", "empty"), ("orig", "const"), ("orig", "empty")]


def score_path(slug: str, arm: str) -> Path:
    if slug == "laion_v2" or arm == "orig":
        return ORIG_DIR / f"scores_{slug}.csv"
    return MATCHED_DIR / f"scores_{slug}__{'an_image' if arm == 'const' else arm}.csv"


def reference_matrix(arm: str) -> Path:
    if arm == "orig":
        return FROZEN_ORIG_MATRIX
    return OUTPUT / f"matched_{arm}_spearman_matrix.csv"


def load_arm(arm: str, ref: pd.DataFrame) -> np.ndarray:
    """Return an (n_images, 4) score matrix aligned to `ref` order."""
    cols = []
    for name, slug in SCORERS:
        s = pd.read_csv(score_path(slug, arm))[["index", "filename", "subculture", "score"]]
        m = ref.merge(s, on=["index", "filename", "subculture"], how="left", validate="one_to_one")
        assert m.score.notna().all() and len(m) == len(ref), (arm, name)
        cols.append(m.score.to_numpy())
    return np.column_stack(cols)


def group_means(scores: np.ndarray, groups: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(6, 4) per-subculture means over the images in `idx`."""
    g = groups[idx]
    counts = np.bincount(g, minlength=len(SUBCULTURES))
    sums = np.stack([np.bincount(g, weights=scores[idx, j], minlength=len(SUBCULTURES))
                     for j in range(scores.shape[1])], axis=1)
    return sums / counts[:, None]


def pair_rhos(means: np.ndarray) -> dict[tuple[int, int], float]:
    ranks = np.column_stack([rankdata(means[:, j]) for j in range(means.shape[1])])
    return {(a, b): float(np.corrcoef(ranks[:, a], ranks[:, b])[0, 1])
            for a, b in combinations(range(means.shape[1]), 2)}


def summarize(rhos: dict[tuple[int, int], float]) -> dict[str, float]:
    return {"inter_peer": float(np.mean([rhos[p] for p in PEER_PAIRS])),
            "laion_peer": float(np.mean([rhos[p] for p in LAION_PAIRS]))}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outs = {k: OUT_DIR / f"bootstrap_{k}.{ext}" for k, ext in
            [("agreement", "csv"), ("pairs", "csv"), ("differences", "csv"),
             ("rank_stability", "csv"), ("report", "md"), ("manifest", "json")]}
    for p in outs.values():
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")

    ref = pd.read_csv(ORIG_DIR / "scores_laion_v2.csv")[["index", "filename", "subculture"]]
    assert len(ref) == 2610
    groups = ref.subculture.map({s: i for i, s in enumerate(SUBCULTURES)}).to_numpy()
    data = {arm: load_arm(arm, ref) for arm in ARMS}
    inputs = sorted({score_path(slug, arm) for arm in ARMS for _, slug in SCORERS})
    inputs += [reference_matrix(arm) for arm in ARMS]

    full_idx = np.arange(len(ref))
    point_pairs = {arm: pair_rhos(group_means(data[arm], groups, full_idx)) for arm in ARMS}
    point = {arm: summarize(point_pairs[arm]) for arm in ARMS}

    names = [n for n, _ in SCORERS]
    max_diff = 0.0
    for arm in ARMS:
        refm = pd.read_csv(reference_matrix(arm), index_col=0)
        for (a, b), r in point_pairs[arm].items():
            max_diff = max(max_diff, abs(r - float(refm.loc[names[a], names[b]])))
    checks = {"full_sample_pair_rho_max_abs_diff_vs_reference_matrices (3 dp files)": max_diff}
    assert max_diff <= 5e-4 + 1e-9, checks

    rng = np.random.default_rng(SEED)
    members = [np.flatnonzero(groups == g) for g in range(len(SUBCULTURES))]
    boot_summary = {arm: {"inter_peer": [], "laion_peer": []} for arm in ARMS}
    boot_pairs = {arm: {p: [] for p in point_pairs[arm]} for arm in ARMS}
    y2k_low = {arm: np.zeros(4) for arm in ARMS}
    cyb_high = {arm: np.zeros(4) for arm in ARMS}
    y2k, cyb = SUBCULTURES.index("y2k"), SUBCULTURES.index("cyberpunk")
    for _ in range(B):
        idx = np.concatenate([rng.choice(m, size=len(m), replace=True) for m in members])
        for arm in ARMS:
            means = group_means(data[arm], groups, idx)
            rhos = pair_rhos(means)
            for k, v in summarize(rhos).items():
                boot_summary[arm][k].append(v)
            for p, v in rhos.items():
                boot_pairs[arm][p].append(v)
            y2k_low[arm] += means.argmin(axis=0) == y2k
            cyb_high[arm] += means.argmax(axis=0) == cyb

    def ci(values) -> tuple[float, float]:
        lo, hi = np.percentile(values, [2.5, 97.5])
        return float(lo), float(hi)

    agreement = []
    for arm in ARMS:
        for metric in ("inter_peer", "laion_peer"):
            lo, hi = ci(boot_summary[arm][metric])
            agreement.append({"arm": arm, "metric": metric, "point": point[arm][metric],
                              "ci95_low": lo, "ci95_high": hi, "B": B})
    agreement = pd.DataFrame(agreement)

    pairs = []
    for arm in ARMS:
        for (a, b), vals in boot_pairs[arm].items():
            lo, hi = ci(vals)
            pairs.append({"arm": arm, "scorer_a": names[a], "scorer_b": names[b],
                          "point": point_pairs[arm][(a, b)], "ci95_low": lo, "ci95_high": hi})
    pairs = pd.DataFrame(pairs)

    differences = []
    for a1, a2 in DIFFERENCES:
        for metric in ("inter_peer", "laion_peer"):
            d = np.array(boot_summary[a1][metric]) - np.array(boot_summary[a2][metric])
            lo, hi = ci(d)
            differences.append({"contrast": f"{a1} - {a2}", "metric": metric,
                                "point": point[a1][metric] - point[a2][metric],
                                "ci95_low": lo, "ci95_high": hi,
                                "share_of_replicates_above_0": float((d > 0).mean())})
    differences = pd.DataFrame(differences)

    stability = pd.DataFrame([{"arm": arm, "scorer": names[j],
                               "share_y2k_lowest": y2k_low[arm][j] / B,
                               "share_cyberpunk_highest": cyb_high[arm][j] / B}
                              for arm in ARMS for j in range(4)])

    released = {"const": "an_image"}
    labels = [released.get(a, a) for a in ARMS]
    agreement = agreement.replace({"arm": released})
    pairs = pairs.replace({"arm": released})
    stability = stability.replace({"arm": released})
    differences["contrast"] = differences["contrast"].str.replace("const", "an_image")
    agreement.to_csv(outs["agreement"], index=False)
    pairs.to_csv(outs["pairs"], index=False)
    differences.to_csv(outs["differences"], index=False)
    stability.to_csv(outs["rank_stability"], index=False)

    fmt = lambda r: f"{r.point:.2f} [{r.ci95_low:.2f}, {r.ci95_high:.2f}]"  # noqa: E731
    table = agreement.assign(v=agreement.apply(fmt, axis=1)).pivot(
        index="arm", columns="metric", values="v").reindex(labels)
    lines = [
        "# Bootstrap intervals for subculture-ordering agreement",
        "",
        f"Generated by `bootstrap_agreement.py`: stratified image bootstrap, B = {B}, "
        f"seed = {SEED}, 95% percentile intervals. Arms: orig = original prompts; "
        "empty = empty string; an_image = the fixed text \"an image\"; label = subculture label. LAION-A V2 is "
        "vision-only (same scores in every arm).",
        "",
        f"Reproduction check: full-sample pair rho matches the reference matrices "
        f"(max |diff| = {max_diff:.1e}; reference files rounded to 3 dp).",
        "",
        "## Mean agreement, value [95% interval]",
        "",
        table.to_markdown(),
        "",
        "## Paired differences between arms",
        "",
        differences.round(3).to_markdown(index=False),
        "",
        "## Each scorer pair",
        "",
        pairs.round(3).to_markdown(index=False),
        "",
        "## Ranking stability (share of replicates)",
        "",
        stability.round(3).to_markdown(index=False),
        "",
    ]
    outs["report"].write_text("\n".join(lines))
    print("\n".join(lines))

    write_manifest(
        outs["manifest"], script=Path(__file__),
        purpose=("Bootstrap 95% intervals for mean inter-peer and LAION-peer Spearman "
                 "agreement on six-subculture orderings, per conditioning arm, plus paired "
                 "arm differences and ranking stability."),
        inputs=inputs, outputs=[outs[k] for k in ("agreement", "pairs", "differences",
                                                  "rank_stability", "report")],
        parameters={"B": B, "seed": SEED, "rng": "numpy.random.default_rng",
                    "resampling": "within subculture, with replacement, sizes fixed",
                    "interval": "percentile 2.5/97.5", "arms": labels,
                    "n_images": int(len(ref))},
        checks=checks,
    )
    print(f"[done] {outs['manifest'].name}")


if __name__ == "__main__":
    main()
