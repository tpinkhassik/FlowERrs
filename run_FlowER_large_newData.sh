#!/bin/sh


export DATA_NAME="flower_new_dataset"
export EXP_NAME="mit_normal_gpu_chi_test"
export EMB_DIM=256
export RBF_HIGH=12
export RBF_GAP=0.1
export SIGMA=0.15

export MODEL_NAME="" # your trained checkpoint here

export TRAIN_BATCH_SIZE=4096
export VAL_BATCH_SIZE=4096
export TEST_BATCH_SIZE=1024

export PARSE_WORKERS=4
export NUM_WORKERS=4
export CUDA_VISIBLE_DEVICES=0
#export HIP_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NUM_GPUS_PER_NODE=1

export NUM_NODES=1
export NODE_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=1235

export TRAIN_FILE=${TRAIN_FILE:-/home/ptim/orcd/scratch/data/FlowERrs_train.txt}
export VAL_FILE=${VAL_FILE:-/home/ptim/orcd/scratch/data/FlowERrs_val.txt}
#export TEST_FILE=${TEST_FILE:-/home/ptim/orcd/scratch/data/FlowERrs_test.txt}
export TEST_FILE=${TEST_FILE:-/home/ptim/orcd/scratch/data/FlowERrs_test_200.txt}
#export TEST_FILE=${TEST_FILE:-/home/ptim/orcd/scratch/data/FlowERrs_test_5k_random.txt}
# export TEST_FILE=${TEST_FILE:-/home/ptim/FlowER/FlowERrs/data/$DATA_NAME/beam.txt}


# export MODEL_PATH=$PWD/checkpoints/$DATA_NAME/$EXP_NAME/
export MODEL_PATH=/home/ptim/orcd/scratch/FlowERrs_checkpoints/$DATA_NAME/$EXP_NAME/
#export RESULT_PATH=/home/ptim/orcd/scratch/FlowERrs_results/$DATA_NAME/$EXP_NAME/mit_preemptable/

#export RESULT_PATH=/home/ptim/orcd/scratch/FlowERrs_results/$DATA_NAME/$EXP_NAME/mit_preemptable/quicktest/tmp/

 [ -f $TRAIN_FILE ] || { echo $TRAIN_FILE does not exist; exit; }
 [ -f $VAL_FILE ] || { echo $VAL_FILE does not exist; exit; }
 [ -f $TEST_FILE ] || { echo $TEST_FILE does not exist; exit; }


export SCALE=4 # smaller sample size during training validation
sh /home/ptim/FlowER/FlowERrs/scripts/train.sh

#export SCALE=1 # larger sample size during testing

#sh scripts/eval_multiGPU.sh
# sh scripts/search.sh
# sh scripts/search_multiGPU.sh
