#!/usr/bin/env bash

if [ $# -eq 0 ]
then
    echo "Help: sh train_multi.sh [gup_nums:2(default)] [yam:fcos_efficientnet_b3_voc.yaml(default)]"
    echo "Yamls:"
    ls ../configs/fcos
    read -p "Press any key to continue" tmp_var
fi

if [ -z $1 ]
then
    GPU_NUMS=2
else
    GPU_NUMS=$1
fi

if [ -z $2 ]
then
    YAML=resnet/fcos_R_50_FPN_2x_voc.yaml
    # YAML=efficientnet/fcos_efficientnet_b3_voc.yaml
else
    YAML=$2
fi

echo "Query "${GPU_NUMS}" free gpus, yaml is "${YAML}

CUDA_DEVICE=
FREE_GPU_NUMS=0
# for i in 0 1 2 3 4 5 6 7
for i in 0 1 2 3 5 6 7
do
    mem=$(nvidia-smi -q -i ${i} -d MEMORY | grep -A 4 GPU | grep Used)
    mem=${mem#*:}
    mem=${mem%*MiB}
    echo "GPU "${i}" used "${mem}" MiB."
    if test $((${mem} < 2048)) -eq 1
    then
        echo "Free GPU "${i}
        FREE_GPU_NUMS=`expr ${FREE_GPU_NUMS} + 1`
        if [ ${FREE_GPU_NUMS} -eq 1 ]
        then
            CUDA_DEVICE=${i}
        else
            CUDA_DEVICE=${CUDA_DEVICE}","${i}
        fi

        if [ ${FREE_GPU_NUMS} -ge ${GPU_NUMS} ]
        then
            # export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}
            echo "export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}
            echo "free gpus "${FREE_GPU_NUMS}
            break
        fi
    fi
done


CUR_PATH=$(pwd)
cd ../
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE}  python -m torch.distributed.launch \
    --nproc_per_node=${FREE_GPU_NUMS} \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --skip-test \
    --iter-clear \
    --ignore-head \
    --config-file configs/fcos/${YAML} \
    DATALOADER.NUM_WORKERS 1 \
    OUTPUT_DIR ${CUR_PATH}/save_model
