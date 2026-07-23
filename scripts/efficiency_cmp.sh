#!/usr/bin/env bash
# Efficiency comparison: 3 experiments on single-node, 2 GPUs each
# All 3 experiments run in background (non-blocking) on different GPU pairs
# Requires 6 GPUs total (GPU 0-1, 2-3, 4-5)
#
# Usage: bash scripts/efficiency_cmp.sh
# Kill all: pkill -f "wm_raw.train.*--wandb-name new_repo_exp"

set -euo pipefail

CONFIG="configs/gpic_image_diffusion.yaml"
NPROC=2
MAX_STEPS=1000

echo "=== Launching 3 efficiency experiments (1000 steps each) ==="

# Exp1: No compile, No CUDAGraph
echo "[Exp1] No compile, No CUDAGraph (GPU 0,1)"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=$NPROC --master_port=29500 -m wm_raw.train \
    --config $CONFIG \
    --max-steps $MAX_STEPS \
    --compile false \
    --compile-mode default \
    --adam-fused false \
    --output-dir outputs/exp1_no_compile_no_cudagraph \
    --log-dir logs/exp1_no_compile_no_cudagraph \
    --wandb-project wm-training \
    --wandb-name new_repo_exp1-no-compile-no-cudagraph \
    > logs/exp1_no_compile_no_cudagraph.log 2>&1 &
PID1=$!

# Exp2: Compile, No CUDAGraph
echo "[Exp2] Compile, No CUDAGraph (GPU 2,3)"
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=$NPROC --master_port=29501 -m wm_raw.train \
    --config $CONFIG \
    --max-steps $MAX_STEPS \
    --compile true \
    --compile-mode default \
    --adam-fused false \
    --output-dir outputs/exp2_compile_no_cudagraph \
    --log-dir logs/exp2_compile_no_cudagraph \
    --wandb-project wm-training \
    --wandb-name new_repo_exp2-compile-no-cudagraph \
    > logs/exp2_compile_no_cudagraph.log 2>&1 &
PID2=$!

# Exp3: Compile + Fused Optimizer
echo "[Exp3] Compile + Fused Optimizer (GPU 4,5)"
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=$NPROC --master_port=29502 -m wm_raw.train \
    --config $CONFIG \
    --max-steps $MAX_STEPS \
    --compile true \
    --compile-mode default \
    --adam-fused true \
    --output-dir outputs/exp3_compile_fused \
    --log-dir logs/exp3_compile_fused \
    --wandb-project wm-training \
    --wandb-name new_repo_exp3-compile-fused \
    > logs/exp3_compile_fused.log 2>&1 &
PID3=$!

echo ""
echo "All experiments launched:"
echo "  Exp1 PID=$PID1 (GPU 0,1)"
echo "  Exp2 PID=$PID2 (GPU 2,3)"
echo "  Exp3 PID=$PID3 (GPU 4,5)"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/exp{1,2,3}_*.log"
echo ""
echo "Wait for all to finish:"
echo "  wait $PID1 $PID2 $PID3"
