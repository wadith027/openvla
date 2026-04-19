#!/bin/bash
# submit_lora_finetune.sh
# Submits LoRA fine-tuning jobs for all shift modes and severities

for SHIFT_MODE in latency freq_drop; do
    if [ "$SHIFT_MODE" == "latency" ]; then
        SEVS="1 2 4 8"
    else
        SEVS="2 4 8"
    fi
    for SEV in $SEVS; do
        DEMO_DIR="/projects/bgub/openvla-tta/openvla/experiments/shifted_demos/${SHIFT_MODE}/sev_${SEV}"
        OUT_DIR="/projects/bgub/openvla-tta/openvla/experiments/lora_oracle/${SHIFT_MODE}/sev_${SEV}"

        sbatch --account=bgub-delta-gpu --partition=gpuA100x4 \
            --gres=gpu:1 --mem=60G --ntasks=1 --time=12:00:00 \
            --output=/projects/bgub/hkurdi/logs/lora_finetune_${SHIFT_MODE}_sev${SEV}_%j.out \
            --wrap="source /projects/bgub/miniconda3/etc/profile.d/conda.sh && \
conda activate /work/hdd/bgub/conda/envs/openvla-oft && \
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 && \
export PYTHONPATH=/projects/bgub/openvla-tta/openvla:/projects/bgub/openvla-tta/LIBERO:\$PYTHONPATH && \
cd /projects/bgub/openvla-tta/openvla && \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune_hdf5.py \
    --vla_path /projects/bgub/models/openvla/openvla-7b-finetuned-libero-spatial \
    --demo_root_dir ${DEMO_DIR} \
    --dataset_name ${SHIFT_MODE}_sev${SEV} \
    --run_root_dir ${OUT_DIR} \
    --max_steps 5000 \
    --batch_size 8 \
    --use_lora True \
    --lora_rank 32 \
    --save_steps 1000 \
    --use_wandb False"

        echo "Submitted finetune job: ${SHIFT_MODE} sev=${SEV}"
    done
done