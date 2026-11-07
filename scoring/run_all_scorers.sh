#!/usr/bin/env bash
# Score the 2,610 corpus images with the four aesthetic predictors.
# Usage: bash scoring/run_all_scorers.sh [laion|pick|ir|hps ...]   (default: all four)
# Writes output/scores/; the released scores in the dataset are not touched.
# PickScore, ImageReward and HPS-v2 have conflicting dependencies: run each in the
# environment pinned in environments/ (see README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SUBCULTURE_DATA:?Set SUBCULTURE_DATA to the dataset root}"
export SUBCULTURE_DATA
OUT=output/scores
PROMPTS="$SUBCULTURE_DATA/data/prompts/prompts_with_flags.csv"
mkdir -p "$OUT"
SCORERS=("$@")
[ ${#SCORERS[@]} -eq 0 ] && SCORERS=(laion pick ir hps)
for S in "${SCORERS[@]}"; do
  if [ "$S" = laion ]; then
    python3 scoring/score_images.py --scorer laion --out-dir "$OUT"
  else
    python3 scoring/score_images.py --scorer "$S" --prompts "$PROMPTS" --out-dir "$OUT"
  fi
done
