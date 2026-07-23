#!/usr/bin/env bash
# ODACE training runner (portable). Trains the editable UNet and writes the
# checkpoint + provenance manifest under $OUT.
#
# Evaluation (ASR / FID-CLIP) is NOT part of this repository; it used the shared
# evaluation harness described in the paper. See external_eval/README.md.
#
# Usage:  bash experiments/run_odace.sh [STEPS] [OUT_DIR]
#   PY=/path/to/python  bash experiments/run_odace.sh 1500 outputs/odace_nudity
set -uo pipefail

# repo root = parent of this script's directory (portable, no hardcoded paths)
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PY="${PY:-python}"
STEPS="${1:-400}"
OUT="${2:-outputs/odace_nudity}"
CONFIG="${CONFIG:-configs/nudity_odace_benign_n1.yaml}"
LOG="${LOG:-outputs/odace_run.log}"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
echo "ODACE train start steps=$STEPS config=$CONFIG out=$OUT $(date -u +%FT%TZ)" | tee -a "$LOG"

"$PY" train_odace.py --config "$CONFIG" --num_steps "$STEPS" --output_dir "$OUT" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}

if [ "$rc" -eq 0 ] && [ -f "$OUT/final/config.json" ]; then
  echo "ODACE_TRAIN_DONE ok out=$OUT $(date -u +%FT%TZ)" | tee -a "$LOG"
else
  echo "ODACE_TRAIN_DONE fail rc=$rc $(date -u +%FT%TZ)" | tee -a "$LOG"
  exit 1
fi
