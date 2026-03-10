#!/bin/bash
#SBATCH -J FlowER_train_0228
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptim@mit.edu
#SBATCH -p mi3001x
#SBATCH -N 1
#SBATCH --time=4:00:00
#SBATCH --output=/work1/connorcoley/ptim/logs/%x_%j.out
#SBATCH --error=/work1/connorcoley/ptim/logs/%x_%j.err
#SBATCH --requeue

set -euo pipefail

# Usage:
#   sbatch scripts/submit_AMD_newData.sh

export REPO_DIR="/home1/ptim/FlowER/FlowERrs"
cd "$REPO_DIR"

source .flower/bin/activate

[ -f "$REPO_DIR/run_FlowER_large_newData.sh" ] || { echo "$REPO_DIR/run_FlowER_large_newData.sh not found"; exit 1; }
sh "$REPO_DIR/run_FlowER_large_newData.sh"
