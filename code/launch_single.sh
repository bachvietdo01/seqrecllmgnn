#!/bin/bash

#SBATCH --job-name=pixelrec_train
#SBATCH --partition=learnai  # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:8        # uncomment only if/as needed
#SBATCH --time=3-00:00:00    # run for one day
#SBATCH --cpus-per-task=64    # change as needed
#SBATCH --ntasks-per-node=1   # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/%u/pixel_rec_train-%j.log

export CUDA_VISIBLE_DEVICES=0
python main.py --device 0 --config_file IDNet/srgnnseqllm.yaml overall/ID.yaml
