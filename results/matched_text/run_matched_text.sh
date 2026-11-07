#!/usr/bin/env bash
# Rescore PickScore, ImageReward, and HPS-v2 under one text input.
# Usage: bash results/matched_text/run_matched_text.sh [empty|const|label] [pick ir hps]
# const is the fixed text "an image", stored as the file suffix an_image.
# Writes output/matched_text/; the released matched-text scores are not touched.
# Run each scorer in its own environment from environments/ (see README.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${SUBCULTURE_DATA:?Set SUBCULTURE_DATA to the dataset root}"
export SUBCULTURE_DATA
ARM="${1:-empty}"
case "$ARM" in
  empty) FALLBACK=empty; SUFFIX=__empty ;;
  const) FALLBACK=const; SUFFIX=__an_image ;;
  label) FALLBACK=subculture; SUFFIX=__label ;;
  *) echo "usage: run_matched_text.sh [empty|const|label]" >&2; exit 1 ;;
esac
shift || true
SCORERS=("$@")
[ ${#SCORERS[@]} -eq 0 ] && SCORERS=(pick ir hps)
OUT=output/matched_text
mkdir -p "$OUT"
for S in "${SCORERS[@]}"; do
  python3 scoring/score_images.py --scorer "$S" --prompt-fallback "$FALLBACK" \
    --out-dir "$OUT" --out-suffix "$SUFFIX"
done
