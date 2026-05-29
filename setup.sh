#!/usr/bin/env bash
# setup.sh — one-shot environment setup for tta-worthiness
#
# Usage (from the repo root, i.e. inside tta-worthiness/openvla/):
#   bash setup.sh
#
# What it does:
#   1. Creates conda env "openvla" with Python 3.10
#   2. Installs PyTorch 2.2.0 (CUDA 12.1)
#   3. Installs this package (OpenVLA) in editable mode
#   4. Builds and installs FlashAttention 2.5.5
#   5. Installs LIBERO (expects ../LIBERO to exist as a submodule)
#   6. Installs LIBERO simulation dependencies
#   7. Downloads LIBERO-Spatial dataset
#   8. Downloads the OpenVLA-7B LIBERO-Spatial checkpoint from HuggingFace
#
# Requirements:
#   - conda must be installed and on PATH
#   - CUDA 12.1-compatible GPU
#   - ~30 GB free disk space for checkpoint + dataset

set -euo pipefail

# ── Configurable ──────────────────────────────────────────────────────────────
ENV_NAME="${ENV_NAME:-openvla}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
PYTHON_VERSION="3.10"
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[setup] $*"; }

# ── Step 1: conda env ──────────────────────────────────────────────────────────
log "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# ── Step 2: PyTorch ────────────────────────────────────────────────────────────
log "Installing PyTorch 2.2.0 (CUDA 12.1)..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121

# ── Step 3: OpenVLA package ────────────────────────────────────────────────────
log "Installing OpenVLA (editable)..."
cd "${REPO_ROOT}"
pip install -e .

# ── Step 4: FlashAttention ─────────────────────────────────────────────────────
log "Installing CUDA toolkit for FlashAttention build..."
conda install -c nvidia cuda-toolkit=12.1 -y
conda install nvidia::cuda-nvcc=12.1 -y
export PATH="${CONDA_PREFIX}/bin:${PATH}"
pip install packaging ninja
log "Building FlashAttention 2.5.5 (this takes a few minutes)..."
pip install flash-attn==2.5.5 --no-build-isolation --no-cache-dir

# ── Step 5: LIBERO ─────────────────────────────────────────────────────────────
LIBERO_DIR="${REPO_ROOT}/../LIBERO"
if [ ! -d "${LIBERO_DIR}" ]; then
    log "LIBERO submodule not found at ${LIBERO_DIR}."
    log "Run: git submodule update --init --recursive"
    exit 1
fi
log "Installing LIBERO..."
cd "${LIBERO_DIR}"
touch libero/__init__.py
pip install -e .

# ── Step 6: Simulation dependencies ───────────────────────────────────────────
log "Installing LIBERO simulation requirements..."
cd "${REPO_ROOT}"
pip install -r experiments/robot/libero/libero_requirements.txt
pip install numpy==1.26.4 tensorflow==2.15.0

# ── Step 7: Download LIBERO-Spatial dataset ────────────────────────────────────
log "Downloading LIBERO-Spatial dataset..."
cd "${LIBERO_DIR}"
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial
cd "${REPO_ROOT}"

# ── Step 8: Download checkpoint ────────────────────────────────────────────────
log "Downloading OpenVLA-7B LIBERO-Spatial checkpoint..."
mkdir -p "${CHECKPOINT_DIR}"
python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    "openvla/openvla-7b-finetuned-libero-spatial",
    local_dir="${CHECKPOINT_DIR}/openvla-7b-finetuned-libero-spatial",
)
print("Checkpoint saved to: ${CHECKPOINT_DIR}/openvla-7b-finetuned-libero-spatial")
EOF

# ── Done ───────────────────────────────────────────────────────────────────────
log ""
log "Setup complete. Activate the environment with:"
log "    conda activate ${ENV_NAME}"
log ""
log "Run a quick test (frozen baseline, object_weight sev 1):"
log "    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa PYTHONPATH=."
log "    python experiments/robot/libero/run_libero_eval.py \\"
log "      --pretrained_checkpoint ${CHECKPOINT_DIR}/openvla-7b-finetuned-libero-spatial \\"
log "      --task_suite_name libero_spatial --center_crop True \\"
log "      --shift_name physics --shift_mode object_weight --severity 1 \\"
log "      --num_trials_per_task 2 --mode none"
