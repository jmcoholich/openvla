#!/bin/bash
#SBATCH --job-name=ftopenvla_8gpu
#SBATCH --output=openvla_syn_coke_10k.out
#SBATCH --error=openvla_syn_coke_10k.err
#SBATCH --partition="kira-lab"
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH --exclude="clippy"

USER=$(whoami)
source /coc/testnvme/$USER/.bashrc
conda activate openvla

srun -u torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir datasets \
  --dataset_name syn_coke10k \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 4 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project openvla \
  --wandb_entity jwit3-georgia-institute-of-technology \
  --shuffle_buffer_size 8000 \
  --run_id_note $1 \
#   --save_steps <NUMBER OF GRADIENT STEPS PER CHECKPOINT SAVE>
#   --run_root_dir <PATH TO LOG/CHECKPOINT DIR> \
#   --adapter_tmp_dir <PATH TO TEMPORARY DIR TO SAVE ADAPTER WEIGHTS> \
