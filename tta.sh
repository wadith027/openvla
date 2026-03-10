#!/bin/bash
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err
#SBATCH -p Quick
#SBATCH --exclude=GPU41,GPU42
#SBATCH --mem=40G
#SBATCH --gpus=1
#SBATCH --array=0-3

source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla

SHIFTS=("texture" "blur" "gamma" "noise")
SHIFT=${SHIFTS[$SLURM_ARRAY_TASK_ID]}

echo "Running shift mode: $SHIFT"

python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5 \
  --shift_mode $SHIFT