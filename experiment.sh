#!/bin/bash
#SBATCH -o std_out
#SBATCH -e std_err
#SBATCH -p Quick
#SBATCH --exclude=GPU41,GPU42
#SBATCH --mem=240G
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=4

source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla

srun --exclusive --ntasks=1 --job-name=texture --output=%x-%j.out \
python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5 \
  --shift_mode texture &

srun --exclusive --ntasks=1 --job-name=blur --output=%x-%j.out \
python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5 \
  --shift_mode blur &

srun --exclusive --ntasks=1  --job-name=gamma --output=%x-%j.out \
python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5 \
  --shift_mode gamma &

srun --exclusive --ntasks=1 --job-name=noise --output=%x-%j.out \
python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5 \
  --shift_mode noise &

wait
