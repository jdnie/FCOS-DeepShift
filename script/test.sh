#!/usr/bin/env bash

CUR_PATH=$(pwd)
cd ../
CUDA_VISIBLE_DEVICES=3 python tools/test_net.py \
    --config-file configs/fcos/fcos_R_50_FPN_1x_voc.yaml \
    MODEL.WEIGHT ${CUR_PATH}/save_model/model_final.pth \
    TEST.IMS_PER_BATCH 4