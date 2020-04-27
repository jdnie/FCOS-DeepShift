#!/usr/bin/env bash

CUR_PATH=$(pwd)
cd ../
python draw_graph_from_log.py \
    --log_path ${CUR_PATH}/save_model/log.txt \
    --float_perfix "loss: " \
    --jpg_path ${CUR_PATH}/loss.jpg

python draw_graph_from_log.py \
    --log_path ${CUR_PATH}/save_model/log.txt \
    --float_perfix "loss_cls: " \
    --jpg_path ${CUR_PATH}/loss_cls.jpg

python draw_graph_from_log.py \
    --log_path ${CUR_PATH}/save_model/log.txt \
    --float_perfix "loss_reg: " \
    --jpg_path ${CUR_PATH}/loss_reg.jpg

python draw_graph_from_log.py \
    --log_path ${CUR_PATH}/save_model/log.txt \
    --float_perfix "loss_centerness: " \
    --jpg_path ${CUR_PATH}/loss_centerness.jpg
