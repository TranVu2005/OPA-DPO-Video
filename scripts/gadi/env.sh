# Sourced by every script. Same layout as Adaptive-SEED's run.md: everything under
# /scratch/hn98/$USER, shared venv and HF cache, VideoHallucer read from project jp09.
export PROJECT_CODE=${PROJECT_CODE:-hn98}
export S=/scratch/$PROJECT_CODE/$USER
export ROOT=${ROOT:-$S/OPA-DPO-Video}                  # this repo
export VENV=${VENV:-$S/venv}                           # shared with Adaptive-SEED
export HF_HOME=${HF_HOME:-$S/hf_cache}                 # shared with Adaptive-SEED (Qwen already downloaded)
export OUT=${OUT:-$ROOT/outputs}                       # working dir on scratch (fast, temporary)
export ARCHIVE=${ARCHIVE:-/g/data/$PROJECT_CODE/$USER/OPA-DPO-Video/outputs}  # kept on gdata by archive.sh
export DATA=$OUT/data                                  # prompts.jsonl, eval_vh.jsonl, split.json
export WORK=$OUT/work                                  # rollouts, reports, sft/dpo jsonl, ref log-probs
export VH_ROOT=${VH_ROOT:-/scratch/jp09/dd9648/datasets_video_hallu/videohallucer}
export EH_ROOT=${EH_ROOT:-$S/runs/exp1/EventHallusion} # prepared by prep_eh in Adaptive-SEED's run.md
export BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct          # resolved from $HF_HOME (offline)

module load python3/3.11.7 cuda/12.3.2
source "$VENV/bin/activate"
cd "$ROOT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1         # compute nodes have no internet
export WANDB_MODE=offline
export TMPDIR=${PBS_JOBFS:-/tmp}
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1                       # no .pyc: saves inodes on scratch
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export NGPUS=${PBS_NGPUS:-1}
mkdir -p "$DATA" "$WORK" "$OUT/logs"

# Arguments every stage shares.
COMMON=(model.base_model="$BASE_MODEL" work_dir="$WORK")
