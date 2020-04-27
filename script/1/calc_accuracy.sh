#!/usr/bin/env bash

CUR_PATH=$(pwd)
NEWEST_MODEL=$(ls save_model/model_* -lt | head -n 1)
NEWEST_MODEL=${NEWEST_MODEL##*/}
SAVE_MODEL_PATH=${CUR_PATH}/save_model/${NEWEST_MODEL}

TEST_BASE_PATH="/home/nie/f/dataset"

YAML_FILE_PATH=${CUR_PATH}/fcos.yaml

SAVE_BASE_PATH="/home/nie/f/test/"

cd ../
python class_pr_cal.py \
        --model_path ${SAVE_MODEL_PATH} \
        --test_base_path ${TEST_BASE_PATH} \
        --save_base_path ${SAVE_BASE_PATH} \
        --config_file ${YAML_FILE_PATH} \
        --confidence_thresholds 0.3 0.4 0.5 0.6 0.8 0.2 0.1
