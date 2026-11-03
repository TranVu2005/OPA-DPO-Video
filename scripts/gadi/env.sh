# Sourced by every PBS script. Edit PROJECT / paths once here.
# Check module versions with `module avail python3 cuda` on Gadi.
export PROJECT_CODE=${PROJECT_CODE:-hn98}                       # your NCI project code
export ROOT=${ROOT:-/g/data/$PROJECT_CODE/$USER/OPA-DPO-Video}  # this repo
export VENV=${VENV:-/g/data/$PROJECT_CODE/$USER/venvs/opadpo-video}
export SCRATCH_DIR=/scratch/$PROJECT_CODE/$USER/opadpo-video

module load python3/3.11.7 cuda/12.4.1 2>/dev/null || module load python3 cuda

source "$VENV/bin/activate"
cd "$ROOT"

# Compute nodes have no internet: everything must already be downloaded.
export HF_HOME=/g/data/$PROJECT_CODE/$USER/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export TMPDIR=${PBS_JOBFS:-/tmp}
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT:$PYTHONPATH"

# Local snapshot of the base model, downloaded by setup.sh.
export BASE_MODEL=$HF_HOME/models/Qwen2.5-VL-7B-Instruct
export NGPUS=${PBS_NGPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}
