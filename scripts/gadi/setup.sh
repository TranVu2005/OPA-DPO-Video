#!/bin/bash
# Run ONCE on a Gadi login node (has internet): venv + packages + model download.
#   PROJECT_CODE=ab12 bash scripts/gadi/setup.sh
set -euo pipefail
export PROJECT_CODE=${PROJECT_CODE:?set PROJECT_CODE}
VENV=${VENV:-/g/data/$PROJECT_CODE/$USER/venvs/opadpo-video}
HF_HOME=/g/data/$PROJECT_CODE/$USER/hf

module load python3/3.11.7 cuda/12.4.1 2>/dev/null || module load python3 cuda
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optional, faster attention on H200 (long build; skip if it fails):
# MAX_JOBS=8 pip install flash-attn --no-build-isolation

export HF_HOME
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir "$HF_HOME/models/Qwen2.5-VL-7B-Instruct"
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
