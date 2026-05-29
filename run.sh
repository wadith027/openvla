#!/usr/bin/env bash
# run.sh — evaluate one condition from the paper (Table 3)
#
# Usage:
#   bash run.sh [mode] [shift_mode] [severity]
#
# Arguments:
#   mode       : none | ttvla | robomonkey          (default: none)
#   shift_mode : gamma | noise | blur | texture |
#                object_weight | gripper_strength |
#                latency | freq_drop                (default: object_weight)
#   severity   : 1–5                                (default: 5)
#
# Examples:
#   bash run.sh                              # frozen baseline, object_weight sev 5
#   bash run.sh none gamma 3                 # frozen, gamma sev 3
#   bash run.sh ttvla object_weight 5        # TT-VLA + verification gate, object_weight sev 5
#   bash run.sh robomonkey latency 4         # RoboMonkey, latency sev 4
#
# Environment variables (override defaults):
#   CHECKPOINT  — path to pretrained checkpoint
#   CONDA_ENV   — conda environment name  (default: openvla)
#   NUM_TRIALS  — rollouts per task       (default: 50, paper uses 50)

set -euo pipefail

MODE=${1:-none}
SHIFT_MODE=${2:-object_weight}
SEVERITY=${3:-5}

CHECKPOINT=${CHECKPOINT:-checkpoints/openvla-7b-finetuned-libero-spatial}
CONDA_ENV=${CONDA_ENV:-openvla}
NUM_TRIALS=${NUM_TRIALS:-50}

# ── Activate environment ───────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Headless rendering ─────────────────────────────────────────────────────────
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=.

# ── Resolve shift name from shift mode ────────────────────────────────────────
case "$SHIFT_MODE" in
    gamma|noise|blur|texture)       SHIFT_NAME=appearance ;;
    object_weight|gripper_strength) SHIFT_NAME=physics ;;
    latency|freq_drop)              SHIFT_NAME=control ;;
    *) echo "Unknown shift_mode: $SHIFT_MODE"; exit 1 ;;
esac

# ── Verification gate (ttvla only, paper §3.4) ────────────────────────────────
# Default: verification ON for ttvla, OFF for everything else.
# Override: ENABLE_VERIFY=False bash run.sh ttvla ...  → unconditional TT-VLA
if [ "$MODE" = "ttvla" ] && [ "${ENABLE_VERIFY:-True}" != "False" ]; then
    VERIFY="--enable_verification_signals True --verify_cmd_mismatch_threshold 0.25"
else
    VERIFY="--enable_verification_signals False"
fi

echo "=================================================="
echo " mode        : $MODE"
echo " shift       : $SHIFT_NAME / $SHIFT_MODE  sev $SEVERITY"
echo " trials/task : $NUM_TRIALS  (10 tasks = $((NUM_TRIALS * 10)) total)"
echo " checkpoint  : $CHECKPOINT"
echo "=================================================="

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint "$CHECKPOINT" \
  --task_suite_name libero_spatial \
  --center_crop True \
  --shift_name "$SHIFT_NAME" \
  --shift_mode "$SHIFT_MODE" \
  --severity "$SEVERITY" \
  --num_trials_per_task "$NUM_TRIALS" \
  --mode "$MODE" \
  $VERIFY
