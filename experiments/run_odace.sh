#!/usr/bin/env bash
# ODACE train + ASR eval through the shared lsse harness (unet swap).
# tee -> tmux pane shows LIVE progress AND log captures everything.
set -uo pipefail
PY="$HOME/miniconda3/envs/lsse/bin/python"
LOG=/mnt/d/unlearning/SD_unlearning/odace/outputs/odace_run.log
STEPS="${1:-400}"; NIMG="${2:-50}"; ATTACKS="${3:-all}"; OUT="${4:-outputs/odace_nudity}"
EVALOUT="outputs/eval/$(basename "$OUT")"
: > "$LOG"
cd /mnt/d/unlearning/SD_unlearning/odace
echo "ODACE train start steps=$STEPS $(date -u +%FT%TZ)" | tee -a "$LOG"
"$PY" train_odace.py --config configs/nudity_odace.yaml --num_steps "$STEPS" --output_dir "$OUT" 2>&1 | tee -a "$LOG"
echo "TRAIN_DONE rc=${PIPESTATUS[0]} $(date -u +%FT%TZ)" | tee -a "$LOG"
if [ ! -f "$OUT/final/config.json" ]; then
  echo "ODACE_ALL_DONE fail=1 (no checkpoint)" | tee -a "$LOG"; exit 1
fi
echo "ODACE eval start nimg=$NIMG attacks=$ATTACKS out=$EVALOUT $(date -u +%FT%TZ)" | tee -a "$LOG"
"$PY" evaluate_odace.py --unet_dir "$OUT/final" --output_dir "$EVALOUT" \
    --num_images "$NIMG" --attacks "$ATTACKS" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ $rc -eq 0 ] && [ -f "$EVALOUT/eval_results.json" ]; then
  echo "ODACE_ALL_DONE fail=0 $(date -u +%FT%TZ)" | tee -a "$LOG"
else
  echo "ODACE_ALL_DONE fail=1 rc=$rc $(date -u +%FT%TZ)" | tee -a "$LOG"
fi
