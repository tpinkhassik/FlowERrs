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

cd "$(dirname "$0")/.."
mkdir -p /home/ptim/orcd/scratch/logs

if [[ -f /etc/profile ]]; then
  set +u
  source /etc/profile
  set -u
fi

if command -v module >/dev/null 2>&1; then
  module load "${MINIFORGE_MODULE:-miniforge/24.3.0-0}"
fi

if [[ -n "${CONDA_ENV:-}" ]]; then
  source activate "${CONDA_ENV}"
fi

sh run_FlowER_large_newData.sh
