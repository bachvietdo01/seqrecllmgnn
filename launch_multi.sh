#!/bin/bash

#SBATCH --job-name=pixelrec_train
#SBATCH --partition=learnai  # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:8        # uncomment only if/as needed
#SBATCH --time=3-00:00:00    # run for one day
#SBATCH --cpus-per-task=32   # change as needed
#SBATCH --ntasks-per-node=1  # tasks reuested per node
#SBATCH --nodes=8
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/%u/pixelrec_train_%j.log

NODE_RANK=$SLURM_NODEID
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
NNODES=$SLURM_NNODES
NPROC_PER_NODE=8
echo "NODE_RANK="$NODE_RANK
echo "NNODES="$SLURM_NNODES
echo "NPROC_PER_NODE="$NPROC_PER_NODE
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
##WORLD_SIZE=$(($SLURM_NNODES *  $SLURM_NTASKS_PER_NODE))
##echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT

python code/mainmn.py --nproc_per_node="${NPROC_PER_NODE}" --nnodes="${NNODES}" --rdzv_id="$SLURM_JOB_ID" --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" --config_file IDNet/srgnnseq.yaml overall/ID.yaml
