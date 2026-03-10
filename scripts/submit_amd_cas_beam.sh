#!/bin/bash
#SBATCH -J FlowER_beam_diverse
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ptim@mit.edu
#SBATCH -p mi3258x
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=256G
#SBATCH --time=10:00:00
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
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Override the orchestration script's GPU settings for ROCm
export NUM_GPUS_PER_NODE=8
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Model config (must match training checkpoint)
export DATA_NAME="flower_new_dataset"
export EXP_NAME="mit_normal_gpu_chi_test"
export EMB_DIM=256
export RBF_HIGH=12
export RBF_GAP=0.1
export SIGMA=0.15
export USE_CHIRALITY=0
export MODEL_NAME="model.2880000_95.pt"

export MODEL_PATH="$WORK/checkpoints/"
export TEST_FILE="$WORK/data/beam.txt"
export SCALE=1
export TEST_BATCH_SIZE=1024
export NUM_WORKERS=4

export BEAM_SIZE=10
export NBEST=5
export MAX_DEPTH=10
export CHUNK_SIZE=50

export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=1235

[ -f "$TEST_FILE" ] || { echo "$TEST_FILE not found"; exit 1; }

# Choose vanilla or diverse beam search
if [ "${DIVERSE:-0}" = "1" ]; then
    export RESULT_PATH="$WORK/results/diverse"
    echo "Running DiverseFlow beam search"
    sh scripts/search_diverse_multiGPU.sh
else
    export RESULT_PATH="$WORK/results/vanilla"
    echo "Running vanilla beam search"
    sh scripts/search_multiGPU.sh
fi
