#!/bin/bash
# Where are the backbones on Gadi? Run on a login node (no GPU, a few seconds):
#   bash scripts/gadi/find_models.sh
# Lists every HF cache entry of Qwen2.5-VL-7B, Qwen3-VL-8B and InternVL3.5-8B under your own dirs
# (and the group's /g/data/hn98), once each, with the number of weight shards in its snapshot.
# A complete model has shards and a config.json. If the entry sits in <dir>/hub/, pass <dir> as
# HF_HOME (qsub -v MODEL=...,HF_HOME=<dir>); env.sh already tries the usual places, so this is only
# needed when a job warns that the model is missing.
set -uo pipefail
PROJECT_CODE=${PROJECT_CODE:-hn98}
ROOTS=("/scratch/$PROJECT_CODE/$USER" "/scratch/jp09/$USER" "/g/data/$PROJECT_CODE")
NAMES=("models--Qwen--Qwen2.5-VL-7B-Instruct" "models--Qwen--Qwen3-VL-8B-Instruct"
       "models--OpenGVLab--InternVL3_5-8B-Instruct")

for name in "${NAMES[@]}"; do
  repo=${name#models--}; repo=${repo/--//}
  echo "== $repo"
  dirs=$(for root in "${ROOTS[@]}"; do
           [ -d "$root" ] && find "$root" -maxdepth 6 -type d -name "$name" -not -path "*/.locks/*" 2>/dev/null
         done | xargs -r -n1 realpath 2>/dev/null | sort -u)
  if [ -z "$dirs" ]; then
    echo "  not found: download on a login node, e.g." \
         "HF_HOME=/g/data/$PROJECT_CODE/$USER/models/hf_cache huggingface-cli download $repo"
    continue
  fi
  while IFS= read -r dir; do
    parent=$(dirname "$dir")
    if [ "$(basename "$parent")" = hub ]; then where="HF_HOME=$(dirname "$parent")"; else where="HF_HUB_CACHE=$parent (no hub/ level)"; fi
    snaps=0
    for snap in "$dir"/snapshots/*/; do
      [ -d "$snap" ] || continue
      snaps=1
      shards=$(find -L "$snap" -maxdepth 1 -name "*.safetensors" | wc -l)
      cfg=$([ -f "$snap/config.json" ] && echo yes || echo NO)
      echo "  $where  shards=$shards config=$cfg  ($(du -shL "$snap" 2>/dev/null | cut -f1))  $dir"
    done
    [ "$snaps" = 1 ] || echo "  $where  no snapshot (incomplete download)  $dir"
  done <<< "$dirs"
done
