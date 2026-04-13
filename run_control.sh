#!/bin/bash
# Usage:
#   bash run_control.sh [task_suite]
#
#   task_suite: libero_spatial | libero_object | libero_goal | libero_10  (default: libero_spatial)
#
# Examples:
#   bash run_control.sh
#   bash run_control.sh libero_object

TASK_SUITE=${1:-libero_spatial}

# Map task suite to local checkpoint
case $TASK_SUITE in
    libero_spatial) CHECKPOINT=/projects/bgub/models/openvla/openvla-7b-finetuned-libero-spatial ;;
    libero_object)  CHECKPOINT=/projects/bgub/models/openvla/openvla-7b-finetuned-libero-object ;;
    libero_goal)    CHECKPOINT=/projects/bgub/models/openvla/openvla-7b-finetuned-libero-goal ;;
    libero_10)      CHECKPOINT=/projects/bgub/models/openvla/openvla-7b-finetuned-libero-10 ;;
    *) echo "Unknown task suite: $TASK_SUITE"; exit 1 ;;
esac

LATENCY_SEVS=(1 2 3 4 5)
FREQ_DROP_SEVS=(1 2 3 4 5)
RESULTS_DIR=/projects/bgub/${USER}/results/control_shifts/${TASK_SUITE}
mkdir -p ${RESULTS_DIR}/logs

for sev in "${LATENCY_SEVS[@]}"; do
    sbatch --account=bgub-delta-gpu \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --mem=60G \
           --ntasks=1 \
           --time=23:00:00 \
           --job-name="ctrl_lat_${TASK_SUITE}_s${sev}" \
           --output="${RESULTS_DIR}/logs/latency_s${sev}_%j.out" \
           --wrap="
source /projects/bgub/miniconda3/etc/profile.d/conda.sh
conda activate /work/hdd/bgub/conda/envs/openvla
cd /projects/bgub/openvla-tta/openvla
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ${CHECKPOINT} \
  --task_suite_name ${TASK_SUITE} \
  --center_crop True \
  --num_trials_per_task 10 \
  --shift_name control \
  --shift_mode latency \
  --severity ${sev} \
  --mode none \
  --run_id_note control_latency_s${sev}
"
    echo "Submitted: ${TASK_SUITE} latency severity=${sev}"
done

for sev in "${FREQ_DROP_SEVS[@]}"; do
    sbatch --account=bgub-delta-gpu \
           --partition=gpuA100x4 \
           --gres=gpu:1 \
           --mem=60G \
           --ntasks=1 \
           --time=23:00:00 \
           --job-name="ctrl_fd_${TASK_SUITE}_s${sev}" \
           --output="${RESULTS_DIR}/logs/freqdrop_s${sev}_%j.out" \
           --wrap="
source /projects/bgub/miniconda3/etc/profile.d/conda.sh
conda activate /work/hdd/bgub/conda/envs/openvla
cd /projects/bgub/openvla-tta/openvla
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint ${CHECKPOINT} \
  --task_suite_name ${TASK_SUITE} \
  --center_crop True \
  --num_trials_per_task 10 \
  --shift_name control \
  --shift_mode freq_drop \
  --severity ${sev} \
  --mode none \
  --run_id_note control_freqdrop_s${sev}
"
    echo "Submitted: ${TASK_SUITE} freq_drop severity=${sev}"
done

echo "All control shift jobs submitted for ${TASK_SUITE}."
