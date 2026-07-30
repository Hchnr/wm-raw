#!/bin/bash
# wm-raw: 2-node × 8-GPU HSDP training launch script
# Mirrors the wm-training torchrun pattern for multi-node jobs.
#
# Usage: run this script on EACH node (via job scheduler).
#   Environment variables expected from scheduler:
#     MASTER_ADDR — hostname/IP of the master node
#     MASTER_PORT — port for rendezvous
#     RANK — node rank (0 or 1)

set -euo pipefail

cd /share/project/eai_pwm/home/hcr/repos/wm-raw

# --- Job identification ---
JOB_ID=$(grep -oP 'scheduling.k8s.io/group-name="\K[^"]+' /etc/downwardapi/annotations 2>/dev/null || echo "local")
RUN_PREFIX="${RUN_PREFIX:-gpic_hsdp_2x8}"
RUN_ID="${RUN_PREFIX}_${JOB_ID}"
LOG_DIR="/share/project/eai_pwm/home/hcr/repos/wm-raw/logs/${RUN_ID}"
mkdir -p "${LOG_DIR}"

# --- Environment ---
export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/wm-raw/src:${PYTHONPATH:-}
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Proxy (if needed for wandb)
export https_proxy="http://crhe:Aa123123@10.8.36.1:3128"
export http_proxy="http://crhe:Aa123123@10.8.36.1:3128"
export no_proxy="localhost,127.0.0.1,10.1.1.16,.local"

# Log environment
env | grep -E "http|RANK|MASTER|NCCL|CUDA" >> "${LOG_DIR}/node${RANK:-0}-envs.log" 2>/dev/null || true

# --- Launch ---
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  --rdzv-conf "join_timeout=1800,last_call_timeout=60" \
  -m wm_raw.train \
  --config configs/gpic_512_aligned_hsdp.yaml \
  2>&1 | tee "${LOG_DIR}/node${RANK:-0}.log"

sleep 999d
