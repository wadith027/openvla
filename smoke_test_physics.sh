#!/bin/bash
#SBATCH -o smoke_test_%j.out
#SBATCH -e smoke_test_%j.err
#SBATCH -p Quick
#SBATCH --exclude=GPU41,GPU42
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --ntasks=1

# Usage:
#   bash smoke_test_physics.sh <shift_mode> <physics_value>
#
#   shift_mode:    object_weight | gripper_strength  (default: object_weight)
#   physics_value: any float multiplier              (default: 10.0)
#
# Examples:
#   bash smoke_test_physics.sh                          # object_weight 10x
#   bash smoke_test_physics.sh object_weight 1000.0     # object_weight 1000x
#   bash smoke_test_physics.sh gripper_strength 0.1     # gripper at 10% strength
#   sbatch smoke_test_physics.sh gripper_strength 0.5

SHIFT_MODE=${1:-object_weight}
PHYSICS_VALUE=${2:-10.0}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=.

echo "Starting physics smoke test: shift_mode=${SHIFT_MODE}, physics_value=${PHYSICS_VALUE}, 1 trial per task..."

xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --shift_name physics \
  --shift_mode ${SHIFT_MODE} \
  --severity 1 \
  --physics_value_override ${PHYSICS_VALUE} \
  --num_trials_per_task 1 \
  --center_crop True

echo "Smoke test complete."
