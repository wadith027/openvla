#!/bin/bash
#SBATCH -o logs/ttvla_sweep-%A-%a.out
#SBATCH -e logs/ttvla_sweep-%A-%a.err
#SBATCH --account=bgub-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4 
#SBATCH --gpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --array=0-73

set -e

source "./env.sh"
mkdir -p logs
# kill conda compiler override
unset CC
unset CXX

# force system compiler (your cluster toolchain)
export CC=/opt/rh/gcc-toolset-13/root/usr/bin/gcc
export CXX=/opt/rh/gcc-toolset-13/root/usr/bin/g++

SHIFT_NAMES=()
SHIFT_MODES=()
SEVERITIES=()
TASK_STARTS=()
TASK_ENDS=()

# Full sweep: 74 jobs = 37 conditions × 2 task-range halves (0–4, 5–9)
#   appearance: gamma(5) + noise(4) + blur(4) + texture(3) = 19 conditions
#   physics:    object_weight(6) + gripper_strength(5)     = 11 conditions
#   control:    latency(5) + freq_drop(5)                  = 10 conditions
for sev in 1 2 3 4 5;   do for half in "0 4" "5 9"; do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(gamma);            SEVERITIES+=($sev); TASK_STARTS+=(${half% *}); TASK_ENDS+=(${half#* }); done; done
for sev in 2 3 4 5;   do for half in "0 4" "5 9"; do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(noise);            SEVERITIES+=($sev); TASK_STARTS+=(${half% *}); TASK_ENDS+=(${half#* }); done; done
for sev in 2 3 4 5;   do for half in "0 4" "5 9"; do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(blur);             SEVERITIES+=($sev); TASK_STARTS+=(${half% *}); TASK_ENDS+=(${half#* }); done; done
for sev in 2 3 4;     do for half in "0 4" "5 9"; do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(texture);          SEVERITIES+=($sev); TASK_STARTS+=(${half% *}); TASK_ENDS+=(${half#* }); done; done
for sev in 1 2 3 4 5 6; do for half in "0 4" "5 9"; do SHIFT_NAMES+=(physics);    SHIFT_MODES+=(object_weight);    SEVERITIES+=($sev); TASK_STARTS+=(${half% *}); TASK_ENDS+=(${half#* }); done; done
for sev in 1 2 3 4 5;   do for half in "0 4" "5 9"; do SHIFT_NAMES+=(physics);    SHIFT_MODES+=(gripper_strength); SEVERITIES+=($sev); TASK_STARTS+=(${half% *}); TASK_ENDS+=(${half#* }); done; done
for sev in 1 2 3 4 5;   do for half in "0 4" "5 9"; do SHIFT_NAMES+=(control);    SHIFT_MODES+=(latency);          SEVERITIES+=($sev); TASK_STARTS+=(${half% *}); TASK_ENDS+=(${half#* }); done; done
for sev in 1 2 3 4 5;   do for half in "0 4" "5 9"; do SHIFT_NAMES+=(control);    SHIFT_MODES+=(freq_drop);        SEVERITIES+=($sev); TASK_STARTS+=(${half% *}); TASK_ENDS+=(${half#* }); done; done

SHIFT_NAME=${SHIFT_NAMES[$SLURM_ARRAY_TASK_ID]}
SHIFT_MODE=${SHIFT_MODES[$SLURM_ARRAY_TASK_ID]}
SEVERITY=${SEVERITIES[$SLURM_ARRAY_TASK_ID]}
TASK_START=${TASK_STARTS[$SLURM_ARRAY_TASK_ID]}
TASK_END=${TASK_ENDS[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: shift_name=$SHIFT_NAME shift_mode=$SHIFT_MODE severity=$SEVERITY tasks=$TASK_START-$TASK_END"


# Override socket path
export TTA_SOCKET_PATH="${DATA_DIR}/tmp/redis_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.sock"

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT="openvla/openvla-7b-finetuned-libero-spatial"
TASK_SUITE="libero_spatial"
N_TRIALS=10
TTA_STEP=5

cd "$VLA_DIR/openvla"

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "$CHECKPOINT" \
  --task_suite_name "$TASK_SUITE" \
  --center_crop True \
  --num_trials_per_task $N_TRIALS \
  --task_start $TASK_START \
  --task_end $TASK_END \
  --shift_name "$SHIFT_NAME" \
  --shift_mode "$SHIFT_MODE" \
  --severity $SEVERITY \
  --sweep_severity $SEVERITY \
  --mode ttvla \
  --tta_step $TTA_STEP \
  --run_id_note "ttvla_sweep__${SHIFT_NAME}_${SHIFT_MODE}_s${SEVERITY}_t${TASK_START}-${TASK_END}" \
  --transfer_dir "./transfer_images/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
