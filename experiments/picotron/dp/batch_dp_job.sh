#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=36
#SBATCH --gres=gpu:3
#SBATCH --mem=108G
#SBATCH --output=picotron_dp_log.txt

NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL NCCL_DEBUG_FILE=dp_logs/picotron_nccl_logs.%h.%p torchrun --nproc_per_node 3 train.py --config dp/llama-1B/config.json 