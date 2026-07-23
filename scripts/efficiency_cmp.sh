#!/usr/bin/env bash
# Efficiency comparison: 4 experiments on single-node 2-GPU FSDP
# Usage: bash scripts/efficiency_cmp.sh <exp_number>
#   exp_number: 1|2|3|4|all

set -euo pipefail

CONFIG="configs/gpic_image_diffusion.yaml"
NPROC=2

run_exp1() {
    echo "=== Exp1: No compile, No CUDAGraph ==="
    torchrun --nproc_per_node=$NPROC -m wm_raw.train \
        --config $CONFIG \
        --compile false \
        --compile-mode default \
        --adam-fused false \
        --output-dir outputs/exp1_no_compile_no_cudagraph \
        --log-dir logs/exp1_no_compile_no_cudagraph \
        --wandb-project wm-raw-ablation \
        --wandb-name exp1-no-compile-no-cudagraph
}

run_exp2() {
    echo "=== Exp2: Compile, No CUDAGraph ==="
    torchrun --nproc_per_node=$NPROC -m wm_raw.train \
        --config $CONFIG \
        --compile true \
        --compile-mode default \
        --adam-fused false \
        --output-dir outputs/exp2_compile_no_cudagraph \
        --log-dir logs/exp2_compile_no_cudagraph \
        --wandb-project wm-raw-ablation \
        --wandb-name exp2-compile-no-cudagraph
}

run_exp3() {
    echo "=== Exp3: Compile + CUDAGraph ==="
    torchrun --nproc_per_node=$NPROC -m wm_raw.train \
        --config $CONFIG \
        --compile true \
        --compile-mode reduce-overhead \
        --adam-fused false \
        --output-dir outputs/exp3_compile_cudagraph \
        --log-dir logs/exp3_compile_cudagraph \
        --wandb-project wm-raw-ablation \
        --wandb-name exp3-compile-cudagraph
}

run_exp4() {
    echo "=== Exp4: Compile + CUDAGraph + Fused Optimizer ==="
    torchrun --nproc_per_node=$NPROC -m wm_raw.train \
        --config $CONFIG \
        --compile true \
        --compile-mode reduce-overhead \
        --adam-fused true \
        --output-dir outputs/exp4_compile_cudagraph_fused \
        --log-dir logs/exp4_compile_cudagraph_fused \
        --wandb-project wm-raw-ablation \
        --wandb-name exp4-compile-cudagraph-fused
}

case "${1:-all}" in
    1) run_exp1 ;;
    2) run_exp2 ;;
    3) run_exp3 ;;
    4) run_exp4 ;;
    all)
        run_exp1
        run_exp2
        run_exp3
        run_exp4
        ;;
    *)
        echo "Usage: $0 {1|2|3|4|all}"
        exit 1
        ;;
esac
