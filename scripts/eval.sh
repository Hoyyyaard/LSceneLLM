#!/usr/bin/env bash

set -x

PORT=$2
ADDR=$1
NGPUS=$3

python -m torch.distributed.launch --master_addr ${ADDR} --master_port=${PORT} --nproc_per_node=${NGPUS} main_ALLM.py \
    --launcher pytorch \
    --config cfgs/XR-QA-Evaluation.yaml \
    --exp_name Release_Test \
    --ckpt ckpts/release.pth\
    --test 