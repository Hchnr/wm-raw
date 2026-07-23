#!/usr/bin/env bash
# Efficiency comparison: 4 experiments on single-node, 2 GPUs each
# All 4 experiments run in background (non-blocking) on different GPU pairs
# Requires 8 GPUs total (GPU 0-1, 2-3, 4-5, 6-7)
#
# Usage: bash scripts/efficiency_cmp.sh

set -euo pipefail

CONFIG="configs/gpic_image_diffusion.yaml"
NPROC=2
MAX_STEPS=1000

echo "=== Launching 4 efficiency experiments (1000 steps each) ==="

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
    --wandb-entity ybainlp \
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
    --wandb-entity ybainlp \
    --wandb-name new_repo_exp2-compile-no-cudagraph \
    > logs/exp2_compile_no_cudagraph.log 2>&1 &
PID2=$!

# Exp3: Compile + CUDAGraph
echo "[Exp3] Compile + CUDAGraph (GPU 4,5)"
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=$NPROC --master_port=29502 -m wm_raw.train \
    --config $CONFIG \
    --max-steps $MAX_STEPS \
    --compile true \
    --compile-mode reduce-overhead \
    --adam-fused false \
    --output-dir outputs/exp3_compile_cudagraph \
    --log-dir logs/exp3_compile_cudagraph \
    --wandb-project wm-raw-ablation \
    --wandb-entity ybainlp \
    --wandb-name new_repo_exp3-compile-cudagraph \
    > logs/exp3_compile_cudagraph.log 2>&1 &
PID3=$!

# Exp4: Compile + CUDAGraph + Fused Optimizer
echo "[Exp4] Compile + CUDAGraph + Fused (GPU 6,7)"
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=$NPROC --master_port=29503 -m wm_raw.train \
    --config $CONFIG \
    --max-steps $MAX_STEPS \
    --compile true \
    --compile-mode reduce-overhead \
    --adam-fused true \
    --output-dir outputs/exp4_compile_cudagraph_fused \
    --log-dir logs/exp4_compile_cudagraph_fused \
    --wandb-project wm-raw-ablation \
    --wandb-entity ybainlp \
    --wandb-name new_repo_exp4-compile-cudagraph-fused \
    > logs/exp4_compile_cudagraph_fused.log 2>&1 &
PID4=$!

echo ""
echo "All experiments launched:"
echo "  Exp1 PID=$PID1 (GPU 0,1)"
echo "  Exp2 PID=$PID2 (GPU 2,3)"
echo "  Exp3 PID=$PID3 (GPU 4,5)"
echo "  Exp4 PID=$PID4 (GPU 6,7)"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/exp{1,2,3,4}_*.log"
echo ""
echo "Wait for all to finish:"
echo "  wait $PID1 $PID2 $PID3 $PID4"
