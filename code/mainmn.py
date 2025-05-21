import os
import argparse
import socket

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--nproc_per_node", type=int)
    parser.add_argument("--nnodes", type=int)
    parser.add_argument("--rdzv_id", type=int)
    parser.add_argument("--rdzv_endpoint", type=str)
    parser.add_argument("--config_file",nargs='+')
    
    args = parser.parse_args()
    # device = args.device
    config_file = args.config_file 
    nproc_per_node = args.nproc_per_node
    nnodes = args.nnodes
    rdzv_id = args.rdzv_id
    rdzv_endpoint = args.rdzv_endpoint
    
    
    if len(config_file) ==2:
        run_yaml = f"srun torchrun --nproc_per_node={nproc_per_node} --nnodes={nnodes} --rdzv_id={rdzv_id} --rdzv_backend=c10d --rdzv_endpoint={rdzv_endpoint} run.py --config_file {config_file[0]} {config_file[1]}"
    elif len(config_file) ==1:
        run_yaml = f"srun torchrun --nproc_per_node={nproc_per_node} --nnodes={nnodes} --rdzv_id={rdzv_id} --rdzv_backend=c10d --rdzv_endpoint={rdzv_endpoint} run.py --config_file {config_file[0]}"

    os.system(run_yaml)


 

