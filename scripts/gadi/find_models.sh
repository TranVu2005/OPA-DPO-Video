#!/bin/bash
# Where are the backbones on Gadi? Run on a login node (no GPU, a few seconds):
#   bash scripts/gadi/find_models.sh
# Lists every HF cache entry of Qwen2.5-VL-7B, Qwen3-VL-8B and InternVL3.5-8B under your own dirs
# (and the group's /g/data/hn98), with the number of weight shards in each snapshot. A complete
# model has shards and a config.json. Pass the cache root (the dir that holds hub/) as HF_HOME:
#   qsub -v MODEL=qwen3,HF_HOME=/scratch/jp09/$USER/hf_cache_qwen3 ...
# env.sh already tries the usual places (see its HF_HOME loop), so this is only needed when a job
# warns that the model is missing.
set -uo pipefail
PROJECT_CODE=${PROJECT_CODE:-hn98}
ROOTS=("/scratch/$PROJECT_CODE/$USER" "/scratch/jp09/$USER" "/g/data/$PROJECT_CODE/$USER" "/g/data/$PROJECT_CODE")
NAMES=("models--Qwen--Qwen2.5-VL-7B-Instruct" "models--Qwen--Qwen3-VL-8B-Instruct"
       "models--OpenGVLab--InternVL3_5-8B-Instruct")

for name in "${NAMES[@]}"; do
  repo=${name#models--}; repo=${repo/--//}
  echo "== $repo"
  found=0
  for root in "${ROOTS[@]}"; do
    [ -d "$root" ] || continue
    while IFS= read -r dir; do
      found=1
      for snap in "$dir"/snapshots/*/; do
        [ -d "$snap" ] || continue
        shards=$(find -L "$snap" -maxdepth 1 -name "*.safetensors" | wc -l)
        cfg=$([ -f "$snap/config.json" ] && echo yes || echo NO)
        echo "  HF_HOME=$(dirname "$(dirname "$dir")")  shards=$shards config=$cfg  ($(du -shL "$snap" 2>/dev/null | cut -f1))"
      done
    done < <(find "$root" -maxdepth 6 -type d -name "$name" 2>/dev/null)
  done
  [ "$found" = 1 ] || echo "  not found: download on a login node, e.g." \
    "HF_HOME=/g/data/$PROJECT_CODE/$USER/models/hf_cache huggingface-cli download $repo"
done
