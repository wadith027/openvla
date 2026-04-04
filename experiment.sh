#!/bin/bash
#SBATCH -o std_out
#SBATCH -e std_err
#SBATCH -p Quick
#SBATCH --exclude=GPU41,GPU42
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --ntasks=1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla

CHECKPOINT="openvla/openvla-7b-finetuned-libero-spatial"
TASK_SUITE="libero_spatial"
N_TRIALS=5

# Run all shift modes sequentially — one at a time.
# Appearance shifts
for SHIFT_MODE in texture blur gamma noise; do
    echo "=========================================="
    echo "Running appearance shift: $SHIFT_MODE"
    echo "=========================================="
    python experiments/robot/libero/run_shift_sweep.py \
      --model_family openvla \
      --pretrained_checkpoint $CHECKPOINT \
      --task_suite_name $TASK_SUITE \
      --center_crop True \
      --num_trials_per_task $N_TRIALS \
      --shift_names '["appearance"]' \
      --shift_mode $SHIFT_MODE
done

# Physics shifts
echo "=========================================="
echo "Running physics shift: object_weight"
echo "=========================================="
python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint $CHECKPOINT \
  --task_suite_name $TASK_SUITE \
  --center_crop True \
  --num_trials_per_task $N_TRIALS \
  --shift_names '["physics"]' \
  --shift_mode object_weight \
  --sweep_severities '[1,2,3,4,5,6]'

echo "=========================================="
echo "Running physics shift: gripper_strength"
echo "=========================================="
python experiments/robot/libero/run_shift_sweep.py \
  --model_family openvla \
  --pretrained_checkpoint $CHECKPOINT \
  --task_suite_name $TASK_SUITE \
  --center_crop True \
  --num_trials_per_task $N_TRIALS \
  --shift_names '["physics"]' \
  --shift_mode gripper_strength \
  --sweep_severities '[1,2,3,4,5]'

echo "All shifts complete."
