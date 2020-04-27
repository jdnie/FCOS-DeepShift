#!/usr/bin/env bash

for i in 0 1 2 3 4 5 6 7
do
    mem=$(nvidia-smi -q -i ${i} -d MEMORY | grep -A 4 GPU | grep Used)
    mem=${mem#*:}
    mem=${mem%*MiB}
    echo "GPU "${i}" used "${mem}" MiB."
    if test $((${mem} < 2048)) -eq 1
    then
        echo "Free GPU "${i}
        export CUDA_VISIBLE_DEVICES=${i}
        break
    fi
done

CUR_PATH=$(pwd)
cd ../../
# export PYTHONPATH=${CUR_PATH}/../../:${PYTHONPATH}
python tools/train_deepshift_net.py \
    --skip-test \
    --iter-clear \
    --config-file ${CUR_PATH}/fcos.yaml \
    DATALOADER.NUM_WORKERS 1 \
    OUTPUT_DIR ${CUR_PATH}/save_model
    # --ignore-head \
