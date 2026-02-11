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
# Optional:
#   sbatch --export=ALL,CONDA_ENV=flower,CONDA_MODULE_PREREQ=deprecated-modules,CONDA_MODULE=anaconda3/2022.05-x86_64 scripts/submit_engaging_newData.sh

REPO_DIR="/home/ptim/FlowER/FlowERrs"
cd "$REPO_DIR"

if [[ -f /etc/profile ]]; then
  set +u
  source /etc/profile
  set -u
fi


module load miniforge

conda activate flower

[ -f "$REPO_DIR/run_FlowER_large_newData.sh" ] || { echo "$REPO_DIR/run_FlowER_large_newData.sh not found"; exit 1; }
sh "$REPO_DIR/run_FlowER_large_newData.sh"
