#!/bin/bash
# submit_oracle_eval.sh
# Submits Oracle evaluation jobs for all shift modes and severities

BASE_CHECKPOINT="/projects/bgub/openvla-tta/openvla/experiments/lora_oracle"
BASE_RESULTS="/projects/bgub/openvla-tta/openvla/experiments/lora_oracle_results"

submit_eval() {
    local SHIFT_MODE=$1
    local SEV=$2
    local SHIFT_NAME=$3
    local CHECKPOINT="${BASE_CHECKPOINT}/${SHIFT_MODE}/sev_${SEV}"
    local RESULTS_DIR="${BASE_RESULTS}/${SHIFT_MODE}/sev_${SEV}"

    mkdir -p $RESULTS_DIR

    sbatch --account=bgub-delta-gpu --partition=gpuA100x4 \
        --gres=gpu:1 --mem=60G --ntasks=1 --time=12:00:00 \
        --output=/projects/bgub/hkurdi/logs/oracle_eval_${SHIFT_MODE}_sev${SEV}_%j.out \
        --wrap="source /projects/bgub/miniconda3/etc/profile.d/conda.sh && \
conda activate /work/hdd/bgub/conda/envs/openvla-oft && \
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 && \
export PYTHONPATH=/projects/bgub/openvla-tta/openvla:/projects/bgub/openvla-tta/LIBERO:\$PYTHONPATH && \
cd /projects/bgub/openvla-tta/openvla && \
python experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint ${CHECKPOINT} \
    --shift_name ${SHIFT_NAME} \
    --shift_mode ${SHIFT_MODE} \
    --severity ${SEV} \
    --num_trials_per_task 5 \
    --mode none \
    --metrics_output_path ${RESULTS_DIR}/metrics.json"

    echo "Submitted Oracle eval: ${SHIFT_MODE} sev=${SEV}"
}

# Control shifts
for SEV in 1 2 3 4 5; do submit_eval latency $SEV control; done
for SEV in 1 2 3 4 5; do submit_eval freq_drop $SEV control; done

# Appearance shifts
for SEV in 1 2 3 4 5; do submit_eval gamma $SEV appearance; done
for SEV in 2 3 4 5;   do submit_eval noise $SEV appearance; done
for SEV in 2 3 4 5;   do submit_eval blur $SEV appearance; done
for SEV in 2 3 4;     do submit_eval texture $SEV appearance; done

# Physics shifts
for SEV in 1 2 3 4 5 6; do submit_eval object_weight $SEV physics; done
for SEV in 1 2 3 4 5;   do submit_eval gripper_strength $SEV physics; done