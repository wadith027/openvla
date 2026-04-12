#!/bin/bash
#SBATCH -o logs/ttvla_sweep-%A-%a.out
#SBATCH -e logs/ttvla_sweep-%A-%a.err
#SBATCH --account=bgub-delta-gpu \
#SBATCH --partition=gpuA100x4-interactive \
#SBATCH --mem=60G
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=2 
#SBATCH --gpus-per-node=2 
#SBATCH --gpus=2
#SBATCH --array=0-29

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
source "./env.sh"

mkdir -p logs

SHIFT_NAMES=()
SHIFT_MODES=()
SEVERITIES=()

for sev in 1 2 3 4 5;   do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(gamma);            SEVERITIES+=($sev); done
for sev in 1 2 3 4 5;   do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(noise);            SEVERITIES+=($sev); done
for sev in 1 2 3 4 5;   do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(blur);             SEVERITIES+=($sev); done
for sev in 1 2 3 4;     do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(texture);          SEVERITIES+=($sev); done
for sev in 1 2 3 4 5 6; do SHIFT_NAMES+=(physics);    SHIFT_MODES+=(object_weight);    SEVERITIES+=($sev); done
for sev in 1 2 3 4 5;   do SHIFT_NAMES+=(physics);    SHIFT_MODES+=(gripper_strength); SEVERITIES+=($sev); done

SHIFT_NAME=${SHIFT_NAMES[$SLURM_ARRAY_TASK_ID]}
SHIFT_MODE=${SHIFT_MODES[$SLURM_ARRAY_TASK_ID]}
SEVERITY=${SEVERITIES[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: shift_name=$SHIFT_NAME shift_mode=$SHIFT_MODE severity=$SEVERITY"


# Override socket path
export TTA_SOCKET_PATH="${DATA_DIR}/tmp/redis_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sock"

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT="openvla/openvla-7b-finetuned-libero-spatial"
TASK_SUITE="libero_spatial"
N_TRIALS=5
TTA_STEP=5

conda activate openvla
cd "$VLA_DIR/openvla"

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "$CHECKPOINT" \
  --task_suite_name "$TASK_SUITE" \
  --center_crop True \
  --num_trials_per_task $N_TRIALS \
  --shift_name "$SHIFT_NAME" \
  --shift_mode "$SHIFT_MODE" \
  --severity $SEVERITY \
  --sweep_severity $SEVERITY \
  --mode ttvla \
  --tta_step $TTA_STEP \
  --run_id_note "ttvla_sweep__${SHIFT_NAME}_${SHIFT_MODE}_s${SEVERITY}" \
  --transfer_dir "./transfer_images/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
