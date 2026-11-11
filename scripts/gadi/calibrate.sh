#!/bin/bash
# Login node, no GPU, a few seconds: threshold calibration of finished eval runs
# (docs/unified_config.md section 9). Reads $OUT/eval/<run>/margins.jsonl.
#   bash scripts/gadi/calibrate.sh                                  # every run in $OUT/eval
#   bash scripts/gadi/calibrate.sh base binary_sft_full binary_dpo_lr1e-4_full
# Writes $OUT/eval/<run>/calibrated.csv and $OUT/eval/calibration_summary.csv.
set -euo pipefail
export VENV=${VENV:-/scratch/jp09/$USER/venv_qwen3}   # any env with numpy + torch; only reads files
source scripts/gadi/env.sh
EH=()
if [ -d "$EH_ROOT/questions" ]; then EH=(--eh_root "$EH_ROOT"); fi
python -m opadpo_video.calibrate --vh "$DATA/eval_vh.jsonl" --eval_dir "$OUT/eval" "${EH[@]}" --runs "$@"
