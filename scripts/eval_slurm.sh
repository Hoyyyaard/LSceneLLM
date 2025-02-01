#!/usr/bin/env bash

set -x

export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export HOME=/gpfs/u/home/LMCG/LMCGljnn/scratch
RANDOM=$$
DIV=1000
OFFSET=24000
MASTER_PORT=$(($RANDOM%$DIV+$OFFSET))
NODE_RANK=${SLURM_PROCID}


NUM_GPUS_PER_NODE=$1

echo "ip: $ip"
echo "FLAG: $FLAG"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "NODE_RANK: $NODE_RANK"
echo "NUM_GPUS_PER_NODE: $NUM_GPUS_PER_NODE"
echo "MASTER_PORT: $MASTER_PORT"

NUM_NODES=${2:-1}
CMD="python -u -m torch.distributed.launch  --nnodes=$NUM_NODES --nproc_per_node=$NUM_GPUS_PER_NODE --master_addr=$ip --node_rank=$NODE_RANK --max_restarts 5"


$CMD  main_ALLM.py \
--launcher slurm \
--config cfgs/XR-QA-Evaluation.yaml \
--exp_name Release_Test \
--ckpt ckpts/release.pth\
--test 