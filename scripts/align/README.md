# 数值对齐工具

分为推理对齐和训练对齐两个子目录。

## 目录结构

```
scripts/align/
├── README.md                          # 本文件
├── inference/                         # 推理对齐（已完成 ✅）
│   ├── align_check.py                 # wm-raw 激活捕获 + 对比
│   └── dump_online_activations.py     # wm-training 激活 dump
├── training/                          # 训练对齐（进行中）
│   └── compare_train_step.py          # 单步 forward+backward 数值对比
└── (../compare_inference.py)          # 推理逐层对比（在 scripts/ 下）
```

## 推理对齐（已完成）

验证方式：相同 checkpoint + 相同 prompt + 相同 seed → 逐层 hidden states 对比。

关键修复：
- MRoPE interleave + 2D grid position IDs
- RMSNorm dtype 转换顺序
- Cross-attention nn.RMSNorm vs Qwen RMSNorm
- Timestep conditioning mode (adaln_zero)
- Attention mask format

结果：生成图片视觉一致，block_ratio 从 7.58 → 1.00。

## 训练对齐（进行中）

### 验证方式

1. 加载相同 checkpoint
2. 用一条相同的 GPIC 数据
3. 固定 seed（timestep, noise 确定性）
4. 跑 1 步 forward + backward
5. 对比：loss 值、prediction、梯度

### 验收标准

| 级别 | 标准 |
|------|------|
| L1 | 同输入 → loss diff < 1e-5 |
| L2 | per-param gradient max_diff < 1e-3 |
| L3 | 1 step 后 weight diff < 1e-5 |
| L4 | 100 步 loss 曲线 relative diff < 1% |

### 使用方法

```bash
export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/wm-training/src:$PYTHONPATH

python scripts/align/training/compare_train_step.py \
    --checkpoint /path/to/step_295000_training.pt \
    --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
    --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
    --prepared-root /share/project/eai_pwm/prepared_datasets/gpic_train_v1_94_incremental \
    --seed 42
```
