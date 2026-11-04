#!/bin/bash
# Login node, no GPU: split identical to Adaptive-SEED + training prompts + eval items,
# plus the expert-free yes/no baseline data in $OUT/work_binary.
#   bash scripts/gadi/prepare.sh
#   TRAIN_SUBSETS="temporal object_relation" DESCRIBE=2 bash scripts/gadi/prepare.sh
set -euo pipefail
source scripts/gadi/env.sh
# shellcheck disable=SC2086  (TRAIN_SUBSETS is a space-separated list)
python -m opadpo_video.prepare_videohallucer --root "$VH_ROOT" --out "$DATA" \
  --subsets temporal object_relation --train_subsets ${TRAIN_SUBSETS:-temporal} --describe "${DESCRIBE:-4}"
# shellcheck disable=SC2086
python -m opadpo_video.build_binary --root "$VH_ROOT" --subsets temporal object_relation \
  --train_subsets ${TRAIN_SUBSETS:-temporal} work_dir="$OUT/work_binary"
