#!/usr/bin/env bash
#SBATCH --job-name=smoke_weight_shift
#SBATCH --account=bgub-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --time=2:00:00
#SBATCH --chdir=/projects/bgub/openvla-tta/openvla
#SBATCH --output=/projects/bgub/openvla-tta/openvla/logs_kwadith/smoke_weight_shift_%j.out
#SBATCH --error=/projects/bgub/openvla-tta/openvla/logs_kwadith/smoke_weight_shift_%j.err

set -e

mkdir -p /projects/bgub/openvla-tta/openvla/logs_kwadith

# module load pytorch-conda/2.8
source /projects/bgub/openvla-tta/env.sh

export PYTHONPATH=.

/work/hdd/bgub/conda/envs/openvla/bin/python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1 \
  --shift_names '["physics"]' \
  --shift_mode object_weight \
  --sweep_severities '[1]' \
  --mode none
