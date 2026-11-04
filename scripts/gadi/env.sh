# Sourced by every script. Same layout as Adaptive-SEED's run.md: everything under
# /scratch/hn98/$USER, shared venv and HF cache, VideoHallucer read from project jp09.
export PROJECT_CODE=${PROJECT_CODE:-hn98}
export S=/scratch/$PROJECT_CODE/$USER
export ROOT=${ROOT:-$S/OPA-DPO-Video}                  # this repo
export VENV=${VENV:-$S/venv}                           # shared with Adaptive-SEED
export HF_HOME=${HF_HOME:-$S/hf_cache}                 # shared with Adaptive-SEED (Qwen already downloaded)
export OUT=${OUT:-/g/data/$PROJECT_CODE/$USER/OPA-DPO-Video/outputs}  # all outputs on gdata (group rule: big data not on scratch)
export ARCHIVE=${ARCHIVE:-$OUT}                          # != $OUT only if you work elsewhere (e.g. OUT on scratch)
export DATA=$OUT/data                                  # prompts.jsonl, eval_vh.jsonl, split.json
export WORK=$OUT/work                                  # rollouts, reports, sft/dpo jsonl, ref log-probs
export VH_ROOT=${VH_ROOT:-/scratch/jp09/dd9648/datasets_video_hallu/videohallucer}
export EH_ROOT=${EH_ROOT:-$S/runs/exp1/EventHallusion} # prepared by prep_eh in Adaptive-SEED's run.md
export BASE_MODEL=Qwen/Qwen2.5-VL-7B-Instruct          # resolved from $HF_HOME (offline)

# Group rule (hn98): environments live as a conda-pack tarball in /g/data/hn98/envs and are unpacked
# into $PBS_JOBFS inside the job, so nobody keeps a full env (tens of thousands of files) on scratch.
# Falls back to the scratch venv while no tarball exists (login node, or before the env is packed).
ENV_NAME=${ENV_NAME:-opadpo-video}
ENV_TAR=${ENV_TAR:-/g/data/$PROJECT_CODE/envs/$ENV_NAME.tar.gz}
if [ -n "${PBS_JOBFS:-}" ] && [ -f "$ENV_TAR" ]; then
  module load cuda/12.3.2                              # no python3 module: the env brings its own Python
  ENV_DIR=$PBS_JOBFS/$ENV_NAME
  if [ ! -f "$ENV_DIR/.unpacked" ]; then               # unpack once per job (archive.sh sources this again)
    mkdir -p "$ENV_DIR"
    tar -xzf "$ENV_TAR" -C "$ENV_DIR"
  fi
  case $- in *u*) _U=1;; *) _U=;; esac
  set +u                                               # conda-pack's activate reads unset CONDA_PREFIX
  source "$ENV_DIR/bin/activate"
  if [ -n "$_U" ]; then set -u; fi
  if [ ! -f "$ENV_DIR/.unpacked" ]; then conda-unpack && touch "$ENV_DIR/.unpacked"; fi
else
  module load python3/3.11.7 cuda/12.3.2
  source "$VENV/bin/activate"
fi
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
