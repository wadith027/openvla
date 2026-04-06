#!/bin/bash
#SBATCH -o %x-%A-%a.out
#SBATCH -e %x-%A-%a.err
#SBATCH -p Quick
#SBATCH --exclude=GPU41,GPU42
#SBATCH --mem=40G
#SBATCH --gpus=2
#SBATCH --array=0-3

source $(conda info --base)/etc/profile.d/conda.sh

SHIFTS=("texture" "blur" "gamma" "noise")
SHIFT=${SHIFTS[$SLURM_ARRAY_TASK_ID]}

# Per-job unique ports to avoid conflicts across array tasks
REWARD_PORT=$((3100 + SLURM_ARRAY_TASK_ID))
ACTION_PORT=$((3200 + SLURM_ARRAY_TASK_ID))
SGLANG_PORT=$((30000 + SLURM_ARRAY_TASK_ID))

MODEL_PATH="openvla/openvla-7b-finetuned-libero-spatial"

echo "Running shift mode: $SHIFT | reward=$REWARD_PORT action=$ACTION_PORT sglang=$SGLANG_PORT"

# --- Ensure log directory exists ---
mkdir -p /general/dayneguy/tmp

# --- Clear any stale log files ---
rm -f /general/dayneguy/tmp/reward_${REWARD_PORT}.log /general/dayneguy/tmp/action_${ACTION_PORT}.log

# --- Start reward server (monkey-verifier) on GPU 0 ---
REWARD_LOG=/general/dayneguy/tmp/reward_${REWARD_PORT}.log
conda activate monkey-verifier
cd /data/dayneguy/vla/RoboMonkey/monkey-verifier/src
CUDA_VISIBLE_DEVICES=0 python infer_server.py --port $REWARD_PORT > $REWARD_LOG 2>&1 &
REWARD_PID=$!

# --- Start VLA action server (sglang-vla) on GPU 1 ---
ACTION_LOG=/general/dayneguy/tmp/action_${ACTION_PORT}.log
conda activate sglang-vla
cd /data/dayneguy/vla/RoboMonkey/sglang-vla
CUDA_VISIBLE_DEVICES=1 python -u openvla_server.py \
    --port $ACTION_PORT \
    --sglang_port $SGLANG_PORT \
    --model_path $MODEL_PATH > $ACTION_LOG 2>&1 &
ACTION_PID=$!

# --- Wait for reward server ---
until grep -q "Application startup complete" $REWARD_LOG 2>/dev/null; do
    sleep 5
done
echo "Reward server ready on port $REWARD_PORT"

# --- Wait for action server (sglang takes longer to load) ---
until grep -q "Application startup complete" $ACTION_LOG 2>/dev/null; do
    sleep 5
done
echo "Action server ready on port $ACTION_PORT"

# --- Run eval ---
conda activate openvla
cd /data/dayneguy/vla/openvla

python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint $MODEL_PATH \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 5 \
  --shift_mode $SHIFT \
  --reward_server_port $REWARD_PORT \
  --action_server_port $ACTION_PORT \
  --task_id $SLURM_ARRAY_TASK_ID \
  --transfer_dir ./transfer_images/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
  --mode robomonkey

kill $REWARD_PID $ACTION_PID