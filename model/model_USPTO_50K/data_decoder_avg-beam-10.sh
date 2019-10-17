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
TEST_SET=$4
OUT_FILE=$5
mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR


/lustre1/lhlai_pkuhpc/wangsw/software/anaconda3/bin/t2t-decoder \
    --t2t_usr_dir=$USR_DIR  \
        --problem=$PROBLEM \
            --data_dir=$DATA_DIR \
                --model=${MODEL} \
                    --hparams_set=${HPARAMS_SET} \
                        --output_dir=$TRAIN_DIR \
                            --decode_from_file=${TMP_DIR}/${TEST_SET} \
                                --decode_to_file=$TRAIN_DIR/${OUT_FILE} \
                                    --decode_hparams="beam_size=10,alpha=0.75,batch_size=${6},return_beams=True" \
                                        --checkpoint_path=${7}
