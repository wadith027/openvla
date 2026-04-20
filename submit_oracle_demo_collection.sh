#!/bin/bash
#SBATCH --account=bgub-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --mem=60G
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=8:00:00
#SBATCH --array=0-36
#SBATCH --output=/projects/bgub/hkurdi/logs/oracle_demo_%A_%a.out
#SBATCH --error=/projects/bgub/hkurdi/logs/oracle_demo_%A_%a.err

source /projects/bgub/miniconda3/etc/profile.d/conda.sh
conda activate /work/hdd/bgub/conda/envs/openvla-oft
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=/projects/bgub/openvla-tta/openvla:/projects/bgub/openvla-tta/LIBERO:$PYTHONPATH

cd /projects/bgub/openvla-tta/openvla

SHIFT_NAMES=()
SHIFT_MODES=()
SEVERITIES=()

# appearance: gamma sev 1-5
for sev in 1 2 3 4 5; do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(gamma);            SEVERITIES+=($sev); done
# appearance: noise sev 2-5
for sev in 2 3 4 5;   do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(noise);            SEVERITIES+=($sev); done
# appearance: blur sev 2-5
for sev in 2 3 4 5;   do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(blur);             SEVERITIES+=($sev); done
# appearance: texture sev 2-4
for sev in 2 3 4;     do SHIFT_NAMES+=(appearance); SHIFT_MODES+=(texture);          SEVERITIES+=($sev); done
# physics: object_weight sev 1-6
for sev in 1 2 3 4 5 6; do SHIFT_NAMES+=(physics);  SHIFT_MODES+=(object_weight);   SEVERITIES+=($sev); done
# physics: gripper_strength sev 1-5
for sev in 1 2 3 4 5; do SHIFT_NAMES+=(physics);    SHIFT_MODES+=(gripper_strength); SEVERITIES+=($sev); done
# control: latency sev 1-5
for sev in 1 2 3 4 5; do SHIFT_NAMES+=(control);    SHIFT_MODES+=(latency);          SEVERITIES+=($sev); done
# control: freq_drop sev 1-5
for sev in 1 2 3 4 5; do SHIFT_NAMES+=(control);    SHIFT_MODES+=(freq_drop);        SEVERITIES+=($sev); done

SHIFT_NAME=${SHIFT_NAMES[$SLURM_ARRAY_TASK_ID]}
SHIFT_MODE=${SHIFT_MODES[$SLURM_ARRAY_TASK_ID]}
SEVERITY=${SEVERITIES[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: shift_name=$SHIFT_NAME shift_mode=$SHIFT_MODE severity=$SEVERITY"

CHECKPOINT="/projects/bgub/models/openvla/openvla-7b-finetuned-libero-spatial"
DEMO_DIR="/projects/bgub/openvla-tta/openvla/experiments/shifted_demos"

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint $CHECKPOINT \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5 \
  --shift_name $SHIFT_NAME \
  --shift_mode $SHIFT_MODE \
  --severity $SEVERITY \
  --sweep_severity $SEVERITY \
  --mode none \
  --save_demos True \
  --demo_output_dir $DEMO_DIR \
  --run_id_note "oracle_demo__${SHIFT_NAME}_${SHIFT_MODE}_s${SEVERITY}"