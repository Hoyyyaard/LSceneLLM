NUM_GPUS_PER_NODE=${2:-6}
NUM_NODES=${3:-2}
JOB_ID=${4:-"allm"}
LOOP_COUNTER=0
SCRIPT=${1:-"scripts/train_slurm.sh"}

srun -J allm --gres=gpu:$NUM_GPUS_PER_NODE -N $NUM_NODES --mem=500G --time 06:00:00 --pty bash $SCRIPT $NUM_GPUS_PER_NODE $NUM_NODES $JOB_ID

