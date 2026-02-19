#!/bin/bash
#SBATCH -J FlowER_newData_mit_normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptim@mit.edu
#SBATCH -p sched_mit_ccoley
#SBATCH -G 8
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --exclude=node2507,node2510
#SBATCH --mem=256G
#SBATCH --time=5:59:59
#SBATCH --output=/home/ptim/orcd/scratch/logs/%x_%j.out
#SBATCH --error=/home/ptim/orcd/scratch/logs/%x_%j.err


set -euo pipefail

# Usage:
#   sbatch --export=ALL,CONDA_ENV=flower scripts/submit_engaging_newData.sh

REPO_DIR="/home/ptim/FlowER/FlowERrs"
cd "$REPO_DIR"

module load miniforge

conda activate flower

[ -f "$REPO_DIR/run_FlowER_large_newData.sh" ] || { echo "$REPO_DIR/run_FlowER_large_newData.sh not found"; exit 1; }
sh "$REPO_DIR/run_FlowER_large_newData.sh"
