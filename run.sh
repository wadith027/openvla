#!/usr/bin/env bash
#SBATCH --job-name=libero_shift_sweep
#SBATCH --partition=Quick
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/libero_shift_sweep_gamma_%j.out

set -e

ENV_NAME=openvla

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

srun -w GPU47 --gpus=1 -p Quick bash -c "
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate openvla
python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 10 \
  --shift_mode blur
" &

srun -w GPU46 --gpus=1 -p Quick bash -c "
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate openvla
python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 10 \
  --shift_mode noise
" &

python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 10 \
  --shift_mode gamma &

wait
