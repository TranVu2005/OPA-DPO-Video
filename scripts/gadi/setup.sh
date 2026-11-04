#!/bin/bash
# Run ONCE on a Gadi login node (has internet), AFTER sections 1.2-1.3 of Adaptive-SEED's run.md
# (shared venv with transformers 4.51.3 / peft 0.15.2, Qwen2.5-VL-7B in the HF cache).
#   cd /scratch/hn98/$USER && git clone <this repo> OPA-DPO-Video && cd OPA-DPO-Video
#   bash scripts/gadi/setup.sh
set -euo pipefail
S=/scratch/hn98/$USER
module load python3/3.11.7 cuda/12.3.2
source "$S/venv/bin/activate"
export HF_HOME=$S/hf_cache PIP_CACHE_DIR=$S/pip_cache

pip install openai av pyyaml decord          # only what Adaptive-SEED's venv lacks
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
if ! ls "$HF_HOME/hub" | grep -q "models--Qwen--Qwen2.5-VL-7B-Instruct"; then
  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
fi
echo "OK. Next: bash scripts/gadi/prepare.sh"
