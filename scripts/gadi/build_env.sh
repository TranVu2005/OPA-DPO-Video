#!/bin/bash
# Build the env in $PBS_JOBFS (not scratch) and pack it to /g/data/hn98/envs. Run inside a copyq job:
#   qsub -I -P hn98 -q copyq -l ncpus=1,jobfs=40GB,mem=32GB,walltime=02:00:00,storage=scratch/hn98+gdata/hn98
#   cd /scratch/hn98/$USER/OPA-DPO-Video && bash scripts/gadi/build_env.sh
set -euo pipefail
: "${PBS_JOBFS:?run inside a PBS job (copyq)}"
NAME=${ENV_NAME:-opadpo-video}
# The group conda base is read-only: keep every cache in jobfs (not in it, not in the small $HOME).
export CONDA_PKGS_DIRS=$PBS_JOBFS/conda_pkgs PIP_CACHE_DIR=$PBS_JOBFS/pip_cache TMPDIR=$PBS_JOBFS/tmp
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"
source /g/data/hn98/mini3/etc/profile.d/conda.sh
conda create -y -p "$PBS_JOBFS/$NAME" --override-channels -c conda-forge python=3.11
conda activate "$PBS_JOBFS/$NAME"
pip install -r scripts/gadi/env_requirements.txt
pip install conda-pack
mkdir -p /g/data/hn98/envs
conda-pack -p "$PBS_JOBFS/$NAME" -o "/g/data/hn98/envs/$NAME.tar.gz"
python -c "import torch, transformers, peft, decord, av; print(torch.__version__, transformers.__version__, peft.__version__)"
ls -lh "/g/data/hn98/envs/$NAME.tar.gz"
