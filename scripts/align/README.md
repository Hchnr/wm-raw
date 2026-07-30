# 数值对齐工具

分为推理对齐和训练对齐两个子目录。

## 目录结构

```
scripts/align/
├── README.md                              # 本文件
├── inference/                             # 推理对齐（已完成 ✅）
│   ├── compare_inference.py               # 逐层数值对比 (wm-raw vs wm-training)
│   ├── generate_image.py                  # 推理生成脚本（支持 .pt/.dcp, batch eval）
│   ├── align_check.py                     # wm-raw 激活捕获 + 对比
│   └── dump_online_activations.py         # wm-training 激活 dump
└── training/                              # 训练对齐（进行中）
    └── compare_train_step.py              # 单步 forward+backward 数值对比
```

## 推理对齐（已完成 ✅）

### 验证方式

相同 checkpoint + 相同 prompt + 相同 seed → 逐层 hidden states 对比 + 生成图片视觉对比。

### 工具说明

| 脚本 | 用途 |
|------|------|
| `compare_inference.py` | 加载两个模型，逐层对比 VLM hidden / diffusion hidden / prediction |
| `generate_image.py` | 单图/批量推理（支持 .pt 和 .dcp checkpoint，9 级 eval prompts）|
| `align_check.py` | 捕获 wm-raw 中间激活并保存 |
| `dump_online_activations.py` | 捕获 wm-training 中间激活并保存 |

### 使用示例

```bash
# 逐层对比
CUDA_VISIBLE_DEVICES=0 python scripts/align/inference/compare_inference.py \
    --checkpoint /path/to/step_295000_training.pt \
    --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
    --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
    --seed 1234

# 批量推理
CUDA_VISIBLE_DEVICES=0 python scripts/align/inference/generate_image.py \
    --checkpoint /path/to/step_295000_training.pt \
    --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
    --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
    --batch --output eval_results/ --cfg-scale 5.0 --num-steps 50
```

### 结果

- MRoPE cos/sin diff: **0.000000**
- Diffusion 28 层逐层 diff: **0.000000**（相同输入）
- 生成图片 block_ratio: **1.00**（smooth，无色块）

---

## 训练对齐（进行中）

### 验证方式

1. 加载相同 checkpoint（两侧模型）
2. 用一条相同的 GPIC 数据（固定 sample index）
3. 固定 seed（确定性 timestep + noise）
4. 跑 1 步 forward + backward
5. 对比：loss 值、prediction、per-parameter 梯度

### 验收标准

| 级别 | 标准 | 说明 |
|------|------|------|
| L1 | loss diff < 1e-5 | 同输入同 seed → loss 完全一致 |
| L2 | gradient max_diff < 1e-3 | 每个参数的梯度对齐 |
| L3 | weight diff < 1e-5 (after 1 step) | optimizer step 后权重一致 |
| L4 | 100 步 loss curve relative diff < 1% | 多步累积不发散 |

### 使用示例

```bash
export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/wm-training/src:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python scripts/align/training/compare_train_step.py \
    --checkpoint /path/to/step_295000_training.pt \
    --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
    --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
    --prepared-root /share/project/eai_pwm/prepared_datasets/gpic_train_v1_94_incremental \
    --seed 42
```

### 已知差异（需修复）

| 维度 | wm-training | wm-raw 当前 |
|------|-------------|-------------|
| VLM condition mask | 传 1D `[B,S]`，HF 内部建 mask | 自建 4D causal mask |
| state_target patchify | 在 model 外部做 patchify | model 内部做 patchify |
| latent_h/latent_w | 通过 latent_shape 隐式确定 | 需显式传入 |
| state_position_ids | 外部构建 2D grid `[3,B,N]` | model 内部构建（OK）|
| state_loss_mask | `make_image_feature_loss_mask`（前 16 dim） | 未传（全部有效）|
