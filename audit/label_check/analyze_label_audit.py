"""
Label check — analyze blinded coding results against the platform labels.

Reads label_check/label_audit_sample.csv (sample_id -> platform label) plus
every labels_coder*.csv in label_check/ of the dataset. Reports per coder:
n coded, agreement with the platform label (overall and per subculture), and,
when two or more coders exist, pairwise Cohen's kappa on jointly coded items
("unclear" kept as a category).

Output: output/label_check_report.txt (also printed).
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


import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
AUDIT = DATASET / "label_check"
SUBS = ["cottagecore", "cyberpunk", "dark_academia", "goblincore", "synthwave", "y2k"]


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(df) == 0:
        return float("nan")
    x, y = df.iloc[:, 0], df.iloc[:, 1]
    po = (x == y).mean()
    pe = sum((x == c).mean() * (y == c).mean() for c in set(x) | set(y))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    key = pd.read_csv(AUDIT / "label_audit_sample.csv").set_index("sample_id")["subculture"]

    files = sorted(AUDIT.glob("labels_coder*.csv"))
    if not files:
        raise SystemExit(f"[FAIL] no labels_coder*.csv found in {AUDIT}")

    coders = {}
    for f in files:
        name = f.stem.replace("labels_", "")
        coders[name] = pd.read_csv(f).set_index("sample_id")["label"]

    lines = [f"Blinded label check — generated "
             f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             f"sample: {len(key)} images, 50/subculture, anonymized, seed 42", ""]

    for name, lab in coders.items():
        joined = pd.concat([lab, key], axis=1, join="inner")
        joined.columns = ["coded", "platform"]
        strict = joined[joined.coded != "unclear"]
        acc = (strict.coded == strict.platform).mean()
        n_unclear = int((joined.coded == "unclear").sum())
        lines.append(f"== coder {name} ==")
        lines.append(f"coded {len(joined)}/{len(key)}   unclear: {n_unclear}   "
                     f"agreement with platform label (excl. unclear): {acc:.3f}")
        per = (strict.assign(ok=strict.coded == strict.platform)
               .groupby("platform")["ok"].mean().round(3))
        lines.append("per-subculture agreement: " +
                     "  ".join(f"{s}: {per.get(s, float('nan'))}" for s in SUBS))
        conf = strict[strict.coded != strict.platform]
        if len(conf):
            top = conf.groupby(["platform", "coded"]).size().sort_values(ascending=False).head(5)
            lines.append("top confusions (platform -> coded): " +
                         "; ".join(f"{a}->{b}: {n}" for (a, b), n in top.items()))
        lines.append("")

    names = list(coders)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k = cohen_kappa(coders[names[i]], coders[names[j]])
            lines.append(f"Cohen's kappa {names[i]} vs {names[j]}: {k:.3f}")

    if len(names) == 1:
        lines.append("(single coder — report as preliminary; second coder pending)")

    report = "\n".join(lines) + "\n"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "label_check_report.txt").write_text(report)
    print(report)
    print("[done] output/label_check_report.txt")


if __name__ == "__main__":
    main()
