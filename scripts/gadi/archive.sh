#!/bin/bash
# Copy what is worth keeping from scratch ($OUT) to gdata ($ARCHIVE): final adapters, eval results,
# data splits and train jsonl. Skips checkpoint-* and ref_logps_* (re-creatable, large / many files).
# Safe to re-run; called at the end of every train / eval job, or by hand on a login node:
#   bash scripts/gadi/archive.sh
set -uo pipefail
source scripts/gadi/env.sh
if [ "$(readlink -f "$OUT")" = "$(readlink -f "$ARCHIVE")" ]; then echo "OUT == ARCHIVE, nothing to copy"; exit 0; fi
mkdir -p "$ARCHIVE"
rsync -a --prune-empty-dirs \
  --exclude 'checkpoint-*' --exclude 'ref_logps_*' --exclude 'rollouts/' --exclude 'logs/' \
  --exclude '*.pt' --exclude '__pycache__/' \
  "$OUT/" "$ARCHIVE/" && echo "archived $OUT -> $ARCHIVE"
