#!/bin/bash
# Login node, no GPU, a few seconds: only the yes/no baseline data (prepare.sh also rebuilds the
# OPA-DPO prompts, which this leaves alone).
#   bash scripts/gadi/prepare_binary.sh
#       -> $OUT/work_binary_all: every TRAIN question of temporal + object_relation, i.e. the whole
#          Adaptive-SEED TRAIN split (150 pairs). Used by the FULL=1 runs of train_opa / train_dpo.
#   TRAIN_SUBSETS=temporal BIN_WORK=$OUT/work_binary bash scripts/gadi/prepare_binary.sh
#       -> the older temporal-only set (130 questions)
set -euo pipefail
source scripts/gadi/env.sh
# shellcheck disable=SC2086  (TRAIN_SUBSETS is a space-separated list)
python -m opadpo_video.build_binary --root "$VH_ROOT" --subsets temporal object_relation \
  --train_subsets ${TRAIN_SUBSETS:-temporal object_relation} work_dir="${BIN_WORK:-$OUT/work_binary_all}"
