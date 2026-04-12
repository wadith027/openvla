#!/bin/bash
#SBATCH -o verify_signals_%j.out
#SBATCH -e verify_signals_%j.err
#SBATCH -p Quick
#SBATCH --exclude=GPU41,GPU42
#SBATCH --mem=60G
#SBATCH --gpus=1
#SBATCH --ntasks=1

# Smoke test for verification signals across all 8 shift types.
# Runs 1 trial per task at a chosen severity so you can see:
#   - The formatted signal summary block every 20 steps in the log
#   - The .signals.jsonl file with per-timestep signal values
#   - The verification_signals section in .metrics.json
#
# Usage:
#   bash smoke_test_verify_signals.sh [shift_name] [shift_mode] [severity]
#
#   shift_name : none | appearance | physics | control  (default: appearance)
#   shift_mode : gamma | noise | blur | texture | object_weight | gripper_strength | latency | freq_drop
#                (default: gamma)
#   severity   : 1-5  (default: 4 — high enough to trigger the gate)
#
# Examples — test each shift type:
#   bash smoke_test_verify_signals.sh appearance gamma          4
#   bash smoke_test_verify_signals.sh appearance noise          4
#   bash smoke_test_verify_signals.sh appearance blur           4
#   bash smoke_test_verify_signals.sh appearance texture        3
#   bash smoke_test_verify_signals.sh physics     object_weight 4
#   bash smoke_test_verify_signals.sh physics     gripper_strength 4
#   bash smoke_test_verify_signals.sh control     latency       4
#   bash smoke_test_verify_signals.sh control     freq_drop     4
#
# After the run, inspect results:
#   grep -A10 "VerifySignals" verify_signals_<JOB_ID>.out
#   cat experiments/log/<shift_mode>/EVAL-*.signals.jsonl | python -m json.tool | head -80
#   python -c "import json,pprint; pprint.pprint(json.load(open('experiments/log/<shift_mode>/EVAL-*.metrics.json'))['verification_signals'])"

SHIFT_NAME=${1:-appearance}
SHIFT_MODE=${2:-gamma}
SEVERITY=${3:-4}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH=.

echo "============================================================"
echo " Verification Signals Smoke Test"
echo "  shift_name : ${SHIFT_NAME}"
echo "  shift_mode : ${SHIFT_MODE}"
echo "  severity   : ${SEVERITY}"
echo "  trials     : 1 per task (fast smoke)"
echo "  mode       : none  (signals computed but TTA gate is advisory only)"
echo "============================================================"

xvfb-run --auto-servernum -s "-screen 0 640x480x24" \
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --shift_name ${SHIFT_NAME} \
  --shift_mode ${SHIFT_MODE} \
  --severity ${SEVERITY} \
  --num_trials_per_task 1 \
  --mode none \
  --enable_verification_signals True \
  --verify_severity_threshold 0.65 \
  --verify_entropy_threshold 3.5

EXIT_CODE=$?

echo ""
echo "============================================================"
echo " Run finished (exit=${EXIT_CODE}). Checking outputs..."
echo "============================================================"

# Print the latest .signals.jsonl (first 5 lines as sample)
JSONL=$(ls -t experiments/log/${SHIFT_MODE}/EVAL-*.signals.jsonl 2>/dev/null | head -1)
if [ -n "$JSONL" ]; then
    echo ""
    echo "--- .signals.jsonl sample (first 5 timesteps) ---"
    head -5 "$JSONL" | python -c "import sys,json; [print(json.dumps(json.loads(l), indent=2)) for l in sys.stdin]"
    echo "(full file: $JSONL)"
else
    echo "WARNING: no .signals.jsonl found under experiments/log/${SHIFT_MODE}/"
fi

# Print verification_signals block from .metrics.json
METRICS=$(ls -t experiments/log/${SHIFT_MODE}/EVAL-*.metrics.json 2>/dev/null | head -1)
if [ -n "$METRICS" ]; then
    echo ""
    echo "--- verification_signals in .metrics.json ---"
    python -c "
import json, pprint
d = json.load(open('${METRICS}'))
vs = d.get('verification_signals', {})
# Print thresholds + skip rate only (per_episode_summaries can be long)
print('thresholds       :', vs.get('thresholds'))
print('skip_rate        :', vs.get('adaptation_skip_rate'))
print('total_opportun.  :', vs.get('total_tta_opportunities'))
print('total_skipped    :', vs.get('total_tta_skipped'))
print('episode summaries:')
for ep in vs.get('per_episode_summaries', [])[:3]:
    pprint.pprint(ep)
    print()
"
    echo "(full metrics: $METRICS)"
else
    echo "WARNING: no .metrics.json found under experiments/log/${SHIFT_MODE}/"
fi

echo ""
echo "To see signal summary blocks from the log:"
echo "  grep -A12 'VerifySignals' verify_signals_${SLURM_JOB_ID:-\$JOBID}.out | head -80"
