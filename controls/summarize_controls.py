"""
Section 5.4 control statistics from the released score tables (no model weights needed).

Real photographs: Cyberpunk minus Y2K mean LAION-A score with a normal 95% interval,
the same gap in the SD corpus, their ratio, and how many photographs reach 6.5.

Other generators (Table 6): per generator, n images, the gap between the mean of the
other five class means and the Y2K mean, whether Y2K is the lowest class, and the
Spearman rho (with p) between the generator's six class means and the corpus means.

Inputs:  photographs/photographs.csv, cross_generator/scores.csv,
         data/scores/scores_laion_v2.csv
Outputs: output/controls_summary.txt, output/table6_generators.csv

Usage:   python3 controls/summarize_controls.py
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
for _p in _Path(__file__).resolve().parents:
    if (_p / "paths.py").is_file():
        if str(_p) not in _sys.path:
            _sys.path.insert(0, str(_p))
        break
from paths import DATASET, OUTPUT

import numpy as np
import pandas as pd
from scipy import stats

CUTOFF = 6.5


def gap_ci(high: pd.Series, low: pd.Series) -> tuple[float, float, float]:
    gap = high.mean() - low.mean()
    se = np.sqrt(high.var() / len(high) + low.var() / len(low))
    return gap, gap - 1.96 * se, gap + 1.96 * se


def main() -> None:
    corpus = pd.read_csv(DATASET / "data" / "scores" / "scores_laion_v2.csv")
    corpus_means = corpus.groupby("subculture")["score"].mean()

    photos = pd.read_csv(DATASET / "photographs" / "photographs.csv")
    cy = photos.loc[photos["style"] == "cyberpunk", "laion_score"]
    yk = photos.loc[photos["style"] == "y2k", "laion_score"]
    gap, lo, hi = gap_ci(cy, yk)
    corpus_gap = corpus_means["cyberpunk"] - corpus_means["y2k"]
    lines = [
        "Real photographs (LAION-A V2)",
        f"  n: cyberpunk {len(cy)}, y2k {len(yk)}",
        f"  means: cyberpunk {cy.mean():.2f}, y2k {yk.mean():.2f}",
        f"  gap {gap:+.2f} (95% CI {lo:.2f} to {hi:.2f})",
        f"  corpus gap {corpus_gap:+.2f}; photo/corpus ratio {gap / corpus_gap:.2f}",
        f"  photographs at or above {CUTOFF}: {(photos['laion_score'] >= CUTOFF).sum()} "
        f"of {len(photos)}",
        "",
    ]

    gens = pd.read_csv(DATASET / "cross_generator" / "scores.csv")
    corpus_other = corpus_means.drop("y2k").mean() - corpus_means["y2k"]
    rows = []
    for (gen, model), g in gens.groupby(["generator", "model_name"], sort=False):
        means = g.groupby("subculture")["laion_score"].mean()
        rho, p = stats.spearmanr(means, corpus_means[means.index])
        rows.append({
            "generator": gen,
            "model_name": model,
            "n": len(g),
            "gap_others_minus_y2k": means.drop("y2k").mean() - means["y2k"],
            "y2k_lowest": means.idxmin() == "y2k",
            "spearman_rho": rho,
            "spearman_p": p,
        })
    table = pd.DataFrame(rows)
    OUTPUT.mkdir(exist_ok=True)
    table.to_csv(OUTPUT / "table6_generators.csv", index=False)

    lines += ["Other generators (Table 6)",
              f"  corpus gap (other five minus y2k) {corpus_other:+.2f}"]
    for r in rows:
        lines.append(f"  {r['model_name']:<26} n={r['n']:<3} gap {r['gap_others_minus_y2k']:+.2f}  "
                     f"y2k lowest {r['y2k_lowest']}  rho {r['spearman_rho']:.2f} "
                     f"(p {r['spearman_p']:.2f})")
    report = "\n".join(lines) + "\n"
    (OUTPUT / "controls_summary.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
