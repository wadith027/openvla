#!/bin/bash
#SBATCH --account=bgub-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH -o smoke_test_control_%j.out
#SBATCH -e smoke_test_control_%j.err
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4

# Usage:
#   sbatch smoke_test_control.sh [shift_mode] [severity]
#
#   shift_mode: latency | freq_drop  (default: latency)
#   severity:   1-5                  (default: 5)
#
# Examples:
#   sbatch smoke_test_control.sh
#   sbatch smoke_test_control.sh freq_drop 3


echo $CUDA_VISIBLE_DEVICES
nvidia-smi


SHIFT_MODE=${1:-latency}
SEVERITY=${2:-5}


echo "Starting control smoke test: shift_mode=${SHIFT_MODE}, severity=${SEVERITY}, 1 trial per task..."

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 1 \
  --shift_name control \
  --shift_mode ${SHIFT_MODE} \
  --severity ${SEVERITY} \
  --mode none \
  --run_id_note control_${SHIFT_MODE}_s${SEVERITY}

echo "Control smoke test complete."

