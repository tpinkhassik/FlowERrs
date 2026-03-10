#!/bin/bash
#SBATCH -J FlowER_cas_beam
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptim@mit.edu
#SBATCH -p sched_engage_amd          # TODO: verify AMD partition name
#SBATCH -G mi210:1                   # TODO: verify GPU type (mi210, mi250x, mi300x)
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=256G
#SBATCH --time=5:59:59
#SBATCH --output=/home/ptim/orcd/scratch/logs/%x_%j.out
#SBATCH --error=/home/ptim/orcd/scratch/logs/%x_%j.err
#SBATCH --requeue

set -euo pipefail

# Usage:
#   Vanilla beam search:
#     sbatch scripts/submit_amd_cas_beam.sh
#
#   Diverse beam search:
#     sbatch --export=ALL,DIVERSE=1 scripts/submit_amd_cas_beam.sh

REPO_DIR="/home/ptim/FlowER/FlowERrs"
cd "$REPO_DIR"

module load miniforge

conda activate flower

# AMD ROCm — use HIP_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=0

# Override the orchestration script's GPU settings for ROCm
export NUM_GPUS_PER_NODE=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Point at CAS beam search data
export TEST_FILE="$REPO_DIR/data/cas/beam.txt"
[ -f "$TEST_FILE" ] || { echo "$TEST_FILE not found"; exit 1; }

# Model config (must match training)
export DATA_NAME="flower_new_dataset"
export EXP_NAME="mit_normal_gpu_chi_test"
export EMB_DIM=256
export RBF_HIGH=12
export RBF_GAP=0.1
export SIGMA=0.15
export MODEL_NAME=""  # TODO: fill in checkpoint filename

export MODEL_PATH="/home/ptim/orcd/scratch/FlowERrs_checkpoints/$DATA_NAME/$EXP_NAME/"
export SCALE=1  # full sample size (64) for beam search

# Choose vanilla or diverse beam search
if [ "${DIVERSE:-0}" = "1" ]; then
    export RESULT_PATH="/home/ptim/orcd/scratch/FlowERrs_results/$DATA_NAME/$EXP_NAME/cas_beam_diverse/"
    echo "Running DiverseFlow beam search"
    sh scripts/search_diverse_multiGPU.sh
else
    export RESULT_PATH="/home/ptim/orcd/scratch/FlowERrs_results/$DATA_NAME/$EXP_NAME/cas_beam/"
    echo "Running vanilla beam search"
    sh scripts/search_multiGPU.sh
fi
