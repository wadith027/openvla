#!/bin/bash
# Usage:
#   bash run_sweep.sh [task_suite]
#
#   task_suite: libero_spatial | libero_object | libero_goal | libero_10  (default: libero_spatial)
#
# Examples:
#   bash run_sweep.sh
#   bash run_sweep.sh libero_object

TASK_SUITE=${1:-libero_spatial}

case $TASK_SUITE in
    libero_spatial) CHECKPOINT=/projects/bgub/models/openvla/openvla-7b-finetuned-libero-spatial ;;
    libero_object)  CHECKPOINT=/projects/bgub/models/openvla/openvla-7b-finetuned-libero-object ;;
    libero_goal)    CHECKPOINT=/projects/bgub/models/openvla/openvla-7b-finetuned-libero-goal ;;
    libero_10)      CHECKPOINT=/projects/bgub/models/openvla/openvla-7b-finetuned-libero-10 ;;
    *) echo "Unknown task suite: $TASK_SUITE"; exit 1 ;;
esac

mkdir -p logs

# ── Appearance shifts ──────────────────────────────────────────────────────────
for shift_mode in gamma noise blur texture; do
    sbatch --account=bgub-delta-gpu \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --mem=60G \
           --ntasks=1 \
           --time=23:00:00 \
           --job-name="app_${shift_mode}_${TASK_SUITE}" \
           --output="logs/appearance_${TASK_SUITE}_${shift_mode}_%j.out" \
           --wrap="
source /projects/bgub/miniconda3/etc/profile.d/conda.sh
conda activate /work/hdd/bgub/conda/envs/openvla
cd /projects/bgub/openvla-tta/openvla
python experiments/robot/libero/run_shift_sweep.py \
  --pretrained_checkpoint ${CHECKPOINT} \
  --task_suite_name ${TASK_SUITE} \
  --center_crop True \
  --num_trials_per_task 10 \
  --shift_names appearance \
  --shift_mode ${shift_mode} \
  --sweep_severities 1 2 3 4 5 \
  --seeds 0 1 2 \
  --sweep_name appearance_${shift_mode}_${TASK_SUITE}
"
    echo "Submitted: ${TASK_SUITE} appearance ${shift_mode}"
done

# ── Physics shifts ─────────────────────────────────────────────────────────────
for shift_mode in object_weight gripper_strength; do
    sbatch --account=bgub-delta-gpu \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --mem=60G \
           --ntasks=1 \
           --time=23:00:00 \
           --job-name="phy_${shift_mode}_${TASK_SUITE}" \
           --output="logs/physics_${TASK_SUITE}_${shift_mode}_%j.out" \
           --wrap="
source /projects/bgub/miniconda3/etc/profile.d/conda.sh
conda activate /work/hdd/bgub/conda/envs/openvla
cd /projects/bgub/openvla-tta/openvla
python experiments/robot/libero/run_shift_sweep.py \
  --pretrained_checkpoint ${CHECKPOINT} \
  --task_suite_name ${TASK_SUITE} \
  --center_crop True \
  --num_trials_per_task 10 \
  --shift_names physics \
  --shift_mode ${shift_mode} \
  --sweep_severities 1 2 3 4 5 \
  --seeds 0 1 2 \
  --sweep_name physics_${shift_mode}_${TASK_SUITE}
"
    echo "Submitted: ${TASK_SUITE} physics ${shift_mode}"
done

# ── Control shifts ─────────────────────────────────────────────────────────────
for shift_mode in latency freq_drop; do
    sbatch --account=bgub-delta-gpu \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --mem=60G \
           --ntasks=1 \
           --time=23:00:00 \
           --job-name="ctrl_${shift_mode}_${TASK_SUITE}" \
           --output="logs/control_${TASK_SUITE}_${shift_mode}_%j.out" \
           --wrap="
source /projects/bgub/miniconda3/etc/profile.d/conda.sh
conda activate /work/hdd/bgub/conda/envs/openvla
cd /projects/bgub/openvla-tta/openvla
python experiments/robot/libero/run_shift_sweep.py \
  --pretrained_checkpoint ${CHECKPOINT} \
  --task_suite_name ${TASK_SUITE} \
  --center_crop True \
  --num_trials_per_task 10 \
  --shift_names control \
  --shift_mode ${shift_mode} \
  --sweep_severities 1 2 3 4 5 \
  --seeds 0 1 2 \
  --sweep_name control_${shift_mode}_${TASK_SUITE}
"
    echo "Submitted: ${TASK_SUITE} control ${shift_mode}"
done

echo "All shifts submitted for ${TASK_SUITE} (8 jobs total)."
