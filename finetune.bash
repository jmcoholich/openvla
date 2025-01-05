#!/bin/bash
#SBATCH --job-name=ftopenvla_4gpu
#SBATCH --output=openvla.out
#SBATCH --error=openvla.err
#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate openvla

srun -u torchrun --standalone --nnodes 1 --nproc-per-node 4 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir datasets \
  --dataset_name syn_coke \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 8 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project openvla \
  --wandb_entity jwit3-georgia-institute-of-technology \
  --shuffle_buffer_size 8000 \
#   --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
#   --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
#   --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \