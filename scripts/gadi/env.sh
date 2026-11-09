# Sourced by every script. Same layout as Adaptive-SEED's run.md: everything under
# /scratch/hn98/$USER, shared venv and HF cache, VideoHallucer read from project jp09.
export PROJECT_CODE=${PROJECT_CODE:-hn98}
export S=/scratch/$PROJECT_CODE/$USER
export ROOT=${ROOT:-$S/OPA-DPO-Video}                  # this repo
export VENV=${VENV:-$S/venv}                           # shared with Adaptive-SEED
export OUT=${OUT:-/g/data/$PROJECT_CODE/$USER/OPA-DPO-Video/outputs}  # all outputs on gdata (group rule: big data not on scratch)
export ARCHIVE=${ARCHIVE:-$OUT}                          # != $OUT only if you work elsewhere (e.g. OUT on scratch)
export DATA=$OUT/data                                  # prompts.jsonl, eval_vh.jsonl, split.json
export WORK=$OUT/work                                  # rollouts, reports, sft/dpo jsonl, ref log-probs
export VH_ROOT=${VH_ROOT:-/scratch/jp09/dd9648/datasets_video_hallu/videohallucer}
export EH_ROOT=${EH_ROOT:-$S/runs/exp1/EventHallusion} # prepared by prep_eh in Adaptive-SEED's run.md

# Backbone (qsub -v MODEL=...). Same checkpoints as Adaptive-SEED (config.py / models/__init__.py).
# Qwen3-VL and InternVL3.5 need transformers >= 4.57, so they use a second env tarball.
# MODEL_TAG suffixes every output / reference dir, so the Qwen2.5 results keep their old names.
export MODEL=${MODEL:-qwen25}
case $MODEL in
  qwen25)   _MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct;       MODEL_TAG="";        _ENV=opadpo-video ;;
  qwen3)    _MODEL_ID=Qwen/Qwen3-VL-8B-Instruct;         MODEL_TAG=_qwen3;    _ENV=opadpo-video-tf457 ;;
  internvl) _MODEL_ID=OpenGVLab/InternVL3_5-8B-Instruct; MODEL_TAG=_internvl; _ENV=opadpo-video-tf457 ;;
  *) echo "env.sh: MODEL must be qwen25, qwen3 or internvl (got '$MODEL')" >&2; return 1 2>/dev/null || exit 1 ;;
esac
export BASE_MODEL=${MODEL_PATH:-$_MODEL_ID} MODEL_TAG     # MODEL_PATH=<local dir> overrides the hub id

# HF cache holding BASE_MODEL (compute nodes are offline). An HF_HOME that has the model wins
# (qsub -v HF_HOME=...); an HF_HOME without it, e.g. exported by ~/.bashrc for Adaptive-SEED, is
# ignored, and the first of these that has the model is used (gdata before scratch).
# Checked with find_models.sh on 2026-11-08:
#   Qwen2.5-VL-7B   /scratch/hn98/$USER/hf_cache
#   Qwen3-VL-8B     /scratch/jp09/$USER/hf_cache_qwen3
#   InternVL3.5-8B  /g/data/hn98/$USER/models/hf_cache_internvl35 (also /scratch/jp09/$USER/hf_cache_internvl35)
_HUB_DIR="models--${BASE_MODEL//\//--}"
if [ -n "${HF_HOME:-}" ] && [ ! -d "$HF_HOME/hub/$_HUB_DIR" ] && [ ! -d "$BASE_MODEL" ]; then
  echo "env.sh: HF_HOME=$HF_HOME has no $BASE_MODEL; searching the usual caches" >&2
  unset HF_HOME
fi
if [ -z "${HF_HOME:-}" ]; then
  _G=/g/data/$PROJECT_CODE/$USER/models
  for _d in "$S/hf_cache" \
            "$_G/hf_cache_qwen25" "$_G/hf_cache_qwen3" "$_G/hf_cache_internvl35" "$_G/hf_cache_internvl" "$_G/hf_cache" "$_G" \
            "/scratch/jp09/$USER/hf_cache_qwen3" "/scratch/jp09/$USER/hf_cache_internvl35" \
            "/scratch/jp09/$USER/hf_cache_internvl" "/scratch/jp09/$USER/hf_cache"; do
    if [ -d "$_d/hub/$_HUB_DIR" ]; then HF_HOME=$_d; break; fi
  done
fi
export HF_HOME=${HF_HOME:-$S/hf_cache}
# Older variables override HF_HOME inside transformers / huggingface_hub; pin them all to one place.
export HF_HUB_CACHE=$HF_HOME/hub HUGGINGFACE_HUB_CACHE=$HF_HOME/hub TRANSFORMERS_CACHE=$HF_HOME/hub
echo "env.sh: MODEL=$MODEL BASE_MODEL=$BASE_MODEL HF_HOME=$HF_HOME" >&2
if [ ! -d "$BASE_MODEL" ] && [ ! -d "$HF_HUB_CACHE/$_HUB_DIR" ]; then
  echo "env.sh: WARNING $BASE_MODEL is not in $HF_HUB_CACHE. Find it with scripts/gadi/find_models.sh," \
       "then qsub -v HF_HOME=<dir holding hub/$_HUB_DIR>; or download it on a login node." >&2
fi
# trust_remote_code (InternVL) copies modeling files here: keep them in jobfs, not in the HF cache.
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-${PBS_JOBFS:-${TMPDIR:-/tmp}}/hf_modules}

# Group rule (hn98): environments live as a conda-pack tarball in /g/data/hn98/envs and are unpacked
# into $PBS_JOBFS inside the job, so nobody keeps a full env (tens of thousands of files) on scratch.
# Falls back to the scratch venv while no tarball exists (login node, or before the env is packed).
ENV_NAME=${ENV_NAME:-$_ENV}
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
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}  # less fragmentation near the 140GB limit
export PYTHONDONTWRITEBYTECODE=1                       # no .pyc: saves inodes on scratch
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export NGPUS=${PBS_NGPUS:-1}
mkdir -p "$DATA" "$WORK" "$OUT/logs"

# Arguments every stage shares.
COMMON=(model.base_model="$BASE_MODEL" work_dir="$WORK")
