#!/usr/bin/env bash

case "$(hostname)" in
    # dayneguy's machine
    gaivi*)
        export DATA_DIR=/data/dayneguy
        export VLA_DIR=/data/dayneguy/vla
        ;;
    # bgub's machine
    *delta*)
        export DATA_DIR=/work/hdd/bgub
        export VLA_DIR=/projects/bgub/openvla-tta
        export CONDA_ENVS_DIR=/work/hdd/bgub/conda/envs
        ;;
    *)
        echo "[env.sh] WARNING: unknown host $(hostname), set DATA_DIR and VLA_DIR manually" >&2
        ;;
esac

# ── Derived paths (do not edit) ───────────────────────────────────────────────
export TTA_SOCKET_PATH="${DATA_DIR}/tmp/redis.sock"
export TTA_SCRIPT="${VLA_DIR}/ttvla/tta.py"

mkdir -p "${DATA_DIR}/tmp"

# Tell conda where to find envs if non-default
if [ -n "${CONDA_ENVS_DIR}" ]; then
    export CONDA_ENVS_PATH="${CONDA_ENVS_DIR}"
fi
