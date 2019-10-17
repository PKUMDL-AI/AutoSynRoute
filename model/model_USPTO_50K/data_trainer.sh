#!/usr/bin/env bash
TRAIN_DATA_DIR=$1
__script_dir=$(cd `dirname $0`; pwd)
cd ${__script_dir}
USR_DIR=${__script_dir}/my_problem
PROBLEM=$2
DATA_DIR=${__script_dir}/$TRAIN_DATA_DIR
TMP_DIR=${__script_dir}/tmp/
TRAIN_DIR=${DATA_DIR}/train/
MODEL=transformer
HPARAMS_SET=transformer_base  #transformer_base_single_gpu
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR



t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS_SET \
  --output_dir=$TRAIN_DIR \
  --hparams='batch_size=4096' \
  --eval_steps=100 \
  --save_checkpoints_steps=2000 \
  --log_step_count_steps=200 \
  --train_steps=${3}
