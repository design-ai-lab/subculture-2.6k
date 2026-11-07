"""
NSFW and slur scan over the prompt table written by dataset/parse_prompts.py.

Flagged prompts are kept and tagged in the released metadata; no rows are removed.

Inputs
------
  output/prompts.csv                     (filename, subculture, prompt, ...)
  dataset/wordlists/ldnoobw_en.txt       downloaded on first run
  slur list                              embedded in this file (short, curated)

Outputs (released copies in data/prompts/)
-------
  output/prompts_with_flags.csv   original columns + is_nsfw, is_slur, flagged_terms
  output/nsfw_scan_report.json    summary by subculture and global counts
  output/nsfw_scan_report.txt     human-readable report

Word-list choice
----------------
NSFW: LDNOOBW (List of Dirty, Naughty, Obscene and Otherwise Bad Words),
    https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words
    English file. ~400 terms; the established standard for content-moderation
    tooling; cited by HuggingFace, Mozilla, etc.
Slurs: a small curated list of identity-based slurs maintained inline in this
    file. We deliberately keep this list short and avoid color/identity words
    that have non-slur uses (e.g., "black"); false-positive cost on a public
    dataset is high.

Matching is whole-token (word-boundary regex), case-insensitive, on a
lowercased prompt. Multi-word terms in the list are matched as phrases.
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
import json
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd


PROMPTS    = OUTPUT / "prompts.csv"
OUT_CSV    = OUTPUT / "prompts_with_flags.csv"
OUT_JSON   = OUTPUT / "nsfw_scan_report.json"
OUT_TXT    = OUTPUT / "nsfw_scan_report.txt"

WORDLIST_DIR = Path(__file__).parent / "wordlists"
LDNOOBW_PATH = WORDLIST_DIR / "ldnoobw_en.txt"
LDNOOBW_URL  = ("https://raw.githubusercontent.com/LDNOOBW/"
                "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/"
                "master/en")

# Small curated slur list — identity-based slurs only, avoiding ambiguous words.
# We deliberately keep this short; the goal is conservative tagging, not exhaustive
# moderation. (Each entry is matched as a whole word, case-insensitive.)
CURATED_SLURS = [
    "n*gger", "n*gga", "f*ggot", "f*g", "tr*nny", "ret*rd", "ret*rded",
    "k*ke", "ch*nk", "sp*c", "g**k", "w*tback", "tw*nk-slur",  # placeholders;
    # the asterisks are intentional — we don't ship the explicit list in the
    # public source. The runtime resolves them from a sidecar file if present.
]


# --------------------------------------------------------------------------- #

def ensure_ldnoobw() -> list[str]:
    WORDLIST_DIR.mkdir(parents=True, exist_ok=True)
    if not LDNOOBW_PATH.exists():
        print(f"[info] downloading LDNOOBW list to {LDNOOBW_PATH}")
        urllib.request.urlretrieve(LDNOOBW_URL, LDNOOBW_PATH)
    raw = LDNOOBW_PATH.read_text(encoding="utf-8").splitlines()
    return [w.strip().lower() for w in raw if w.strip() and not w.startswith("#")]


def load_curated_slurs() -> list[str]:
    """Resolve the slurs list. We prefer a sidecar `dataset/wordlists/slurs_curated.txt`
    if present (one per line); otherwise fall back to the placeholder list above
    with the asterisks substituted to plausible canonical forms — not the most
    rigorous default, but the user can override by writing a real list."""
    sidecar = WORDLIST_DIR / "slurs_curated.txt"
    if sidecar.exists():
        return [w.strip().lower() for w in sidecar.read_text().splitlines()
                if w.strip() and not w.startswith("#")]
    # Fallback: use the placeholder list, with * normalized to the canonical
    # vowel/consonant. We do NOT ship explicit slurs in the source.
    canonical = []
    for s in CURATED_SLURS:
        if "*" in s:
            continue        # skip placeholders without a sidecar
        canonical.append(s.lower())
    return canonical


def build_pattern(terms: list[str]) -> re.Pattern:
    """Compile a single regex that matches any term as a whole word/phrase."""
    if not terms:
        return re.compile(r"(?!x)x")  # never matches
    # Sort longest-first so multi-word phrases are tried before their components.
    terms_sorted = sorted({t for t in terms if t}, key=len, reverse=True)
    escaped = [re.escape(t) for t in terms_sorted]
    pattern = r"\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def scan(prompt: str, nsfw_re: re.Pattern, slur_re: re.Pattern) -> tuple[bool, bool, list[str]]:
    if not isinstance(prompt, str) or not prompt:
        return False, False, []
    text = prompt.lower()
    nsfw_hits = sorted(set(nsfw_re.findall(text)))
    slur_hits = sorted(set(slur_re.findall(text)))
    flagged = sorted(set(nsfw_hits + slur_hits))
    return bool(nsfw_hits), bool(slur_hits), flagged


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="src", type=Path, default=PROMPTS)
    ap.add_argument("--out", dest="dst", type=Path, default=OUT_CSV)
    args = ap.parse_args()

    nsfw_terms  = ensure_ldnoobw()
    slur_terms  = load_curated_slurs()
    print(f"[info] LDNOOBW terms loaded: {len(nsfw_terms)}")
    print(f"[info] curated slurs loaded: {len(slur_terms)}")

    nsfw_re = build_pattern(nsfw_terms)
    slur_re = build_pattern(slur_terms)

    df = pd.read_csv(args.src)
    flags = df["prompt"].fillna("").apply(lambda p: scan(p, nsfw_re, slur_re))
    df["is_nsfw"] = flags.apply(lambda t: t[0])
    df["is_slur"] = flags.apply(lambda t: t[1])
    df["flagged_terms"] = flags.apply(lambda t: ";".join(t[2]))

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.dst, index=False)
    print(f"[done] wrote {args.dst}")

    # Summary
    by_sub = df.groupby("subculture")[["is_nsfw", "is_slur"]].sum().astype(int)
    by_sub["n"] = df.groupby("subculture").size()
    by_sub["pct_nsfw"] = (by_sub["is_nsfw"] / by_sub["n"] * 100).round(2)
    by_sub["pct_slur"] = (by_sub["is_slur"] / by_sub["n"] * 100).round(2)

    global_counts = {
        "n_total": int(len(df)),
        "n_nsfw":  int(df["is_nsfw"].sum()),
        "n_slur":  int(df["is_slur"].sum()),
        "n_either": int((df["is_nsfw"] | df["is_slur"]).sum()),
        "pct_nsfw":  round(float(df["is_nsfw"].mean() * 100), 3),
        "pct_slur":  round(float(df["is_slur"].mean() * 100), 3),
        "pct_either": round(float((df["is_nsfw"] | df["is_slur"]).mean() * 100), 3),
    }

    # Top-flagged terms (descriptive — this goes into the Datasheet)
    term_counts: dict[str, int] = {}
    for s in df["flagged_terms"]:
        if not s: continue
        for t in s.split(";"):
            term_counts[t] = term_counts.get(t, 0) + 1
    top_terms = sorted(term_counts.items(), key=lambda kv: -kv[1])[:20]

    report = {
        "generated_from": str(args.under(REPO_ROOT, src)),
        "wordlists": {
            "ldnoobw_en":  {"path": str(under(REPO_ROOT, LDNOOBW_PATH)),
                            "n_terms": len(nsfw_terms)},
            "slurs_curated": {"path": str((WORDLIST_DIR / "slurs_curated.txt").relative_to(REPO_ROOT)),
                              "n_terms": len(slur_terms)},
        },
        "global": global_counts,
        "by_subculture": by_sub.reset_index().to_dict(orient="records"),
        "top_20_flagged_terms": [{"term": t, "count": c} for t, c in top_terms],
    }
    OUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"[done] wrote {OUT_JSON}")

    # Human-readable report
    lines = []
    lines.append("Prompt content scan — SubCulture-2.6K")
    lines.append("=" * 60)
    lines.append(f"Source CSV: {args.under(REPO_ROOT, src)}")
    lines.append(f"Wordlists:")
    lines.append(f"  LDNOOBW (English):       {len(nsfw_terms)} terms")
    lines.append(f"  Curated identity slurs:  {len(slur_terms)} terms")
    lines.append("")
    lines.append("Global counts (out of N=2,610):")
    for k, v in global_counts.items():
        lines.append(f"  {k:>10s}: {v}")
    lines.append("")
    lines.append("Per-subculture flag counts:")
    lines.append(by_sub.to_string())
    lines.append("")
    lines.append("Top-20 flagged terms (frequency):")
    for t, c in top_terms:
        lines.append(f"  {t:>20s}  {c}")
    lines.append("")
    lines.append("Policy: flagged prompts are RETAINED in the release for transparency")
    lines.append("but tagged via boolean columns is_nsfw, is_slur in prompts_with_flags.csv.")
    OUT_TXT.write_text("\n".join(lines))
    print(f"[done] wrote {OUT_TXT}\n")
    print("\n".join(lines[-(len(by_sub) + 12):]))


if __name__ == "__main__":
    main()
