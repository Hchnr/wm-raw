#!/usr/bin/env bash
# Efficiency comparison: 3 experiments on single-node, 2 GPUs each
# All 3 experiments run in background (non-blocking) on different GPU pairs
# Requires 6 GPUs total (GPU 0-1, 2-3, 4-5)
#
# Usage:
#   bash scripts/efficiency_cmp.sh
#   BS=8 NW=4 bash scripts/efficiency_cmp.sh
#
# Kill all: pkill -f "wm_raw.train"

set -euo pipefail

CONFIG="configs/gpic_image_diffusion.yaml"
NPROC=2
MAX_STEPS=1000

BS=${BS:-4}   # batch_size for all experiments
NW=${NW:-4}   # num_workers for all experiments

echo "=== Launching 3 efficiency experiments (${MAX_STEPS} steps each, bs=${BS} nw=${NW}) ==="
echo ""

mkdir -p logs

# Exp1: Baseline (no compile, no fused)
TAG1="exp1_baseline_bs${BS}_nw${NW}"
echo "[Exp1] Baseline (GPU 0,1)"
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=$NPROC --master_port=29500 -m wm_raw.train \
    --config $CONFIG \
    --max-steps $MAX_STEPS \
    --compile false \
    --compile-mode default \
    --adam-fused false \
    --batch-size $BS \
    --num-workers $NW \
    --output-dir outputs/${TAG1} \
    --log-dir logs/${TAG1} \
    --wandb-project wm-training \
    --wandb-name ${TAG1} \
    > logs/${TAG1}.log 2>&1 &
PID1=$!

# Exp2: Compile
TAG2="exp2_compile_bs${BS}_nw${NW}"
echo "[Exp2] Compile (GPU 2,3)"
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=$NPROC --master_port=29501 -m wm_raw.train \
    --config $CONFIG \
    --max-steps $MAX_STEPS \
    --compile true \
    --compile-mode default \
    --adam-fused false \
    --batch-size $BS \
    --num-workers $NW \
    --output-dir outputs/${TAG2} \
    --log-dir logs/${TAG2} \
    --wandb-project wm-training \
    --wandb-name ${TAG2} \
    > logs/${TAG2}.log 2>&1 &
PID2=$!

# Exp3: Compile + Fused AdamW
TAG3="exp3_compile_fused_adamw_bs${BS}_nw${NW}"
echo "[Exp3] Compile + Fused AdamW (GPU 4,5)"
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=$NPROC --master_port=29502 -m wm_raw.train \
    --config $CONFIG \
    --max-steps $MAX_STEPS \
    --compile true \
    --compile-mode default \
    --adam-fused true \
    --batch-size $BS \
    --num-workers $NW \
    --output-dir outputs/${TAG3} \
    --log-dir logs/${TAG3} \
    --wandb-project wm-training \
    --wandb-name ${TAG3} \
    > logs/${TAG3}.log 2>&1 &
PID3=$!

echo ""
echo "All experiments launched:"
echo "  Exp1 (baseline)             PID=$PID1  (GPU 0,1)  -> outputs/${TAG1}"
echo "  Exp2 (compile)              PID=$PID2  (GPU 2,3)  -> outputs/${TAG2}"
echo "  Exp3 (compile-fused-adamw)  PID=$PID3  (GPU 4,5)  -> outputs/${TAG3}"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/${TAG1}.log"
echo "  tail -f logs/${TAG2}.log"
echo "  tail -f logs/${TAG3}.log"
echo ""
echo "Wait for all to finish:"
echo "  wait $PID1 $PID2 $PID3"
