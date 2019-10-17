#!/usr/bin/env bash
TRAIN_DATA_DIR=$1
__script_dir=$(cd `dirname $0`; pwd)
cd ${__script_dir}
USR_DIR=${__script_dir}/my_problem
PROBLEM=$2
DATA_DIR=${__script_dir}/$TRAIN_DATA_DIR
TMP_DIR=$3
TRAIN_DIR=${DATA_DIR}/train/
MODEL=transformer
HPARAMS_SET=transformer_base  #transformer_base_single_gpu
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

cp ${TMP_DIR}/vocab.token ${DATA_DIR}

t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM
