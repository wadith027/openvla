#!/bin/bash
#SBATCH -o std_out
#SBATCH -e std_err 
#SBATCH -p Quick 
#SBATCH -w GPU45,GPU46,GPU47
#SBATCH --cpus-per-task=32 ### 32 CPUs per task 
#SBATCH --mem=40GB ### 100GB per task 
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1 ### 8 GPUs per task
#SBATCH --time=1-0
source $(conda info --base)/etc/profile.d/conda.sh
conda activate openvla
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True 
