#!/bin/bash
#SBATCH -J FlowER_newData
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptim@mit.edu
#SBATCH -p mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=47:59:59
#SBATCH --output=/home/ptim/orcd/scratch/logs/%x_%j.out
#SBATCH --error=/home/ptim/orcd/scratch/logs/%x_%j.err
#SBATCH --requeue

set -euo pipefail

# Usage:
#   sbatch --export=ALL,CONDA_ENV=flower scripts/submit_engaging_newData.sh

REPO_DIR="/orcd/home/002/ptim/FlowER/FlowERrs"
cd "$REPO_DIR"
mkdir -p /home/ptim/orcd/scratch/logs

if [[ -f /etc/profile ]]; then
  set +u
  source /etc/profile
  set -u
fi


module load deprecated-modules
module load anaconda3/2022.05-x86_64

CONDA_ENV_NAME="${CONDA_ENV:-flower}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "conda command not found after module load; cannot activate ${CONDA_ENV_NAME}" >&2
  exit 1
fi

python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    raise SystemExit(f"torch import failed: {e}")
try:
    import rdkit
except Exception as e:
    raise SystemExit(f"rdkit import failed: {e}")
print(f"Python: {sys.executable}")
print(f"Torch: {torch.__version__}")
print(f"RDKit: {rdkit.__version__}")
PY

[ -f "$REPO_DIR/run_FlowER_large_newData.sh" ] || { echo "$REPO_DIR/run_FlowER_large_newData.sh not found"; exit 1; }
sh "$REPO_DIR/run_FlowER_large_newData.sh"
