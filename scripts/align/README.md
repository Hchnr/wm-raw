# 数值对齐工具

逐层对比 wm-raw 与 wm-training（线上模型）的中间激活，定位 diverge 的层。

## 原理

1. 在 wm-training 环境跑一次 forward（真实 GPIC 数据，固定 seed/timestep/noise），用 hook 抓取所有中间 tensor，保存为 `alignment_fixture.pt`
2. 在 wm-raw 环境用相同 checkpoint + 相同输入回放 forward，同样 hook 抓取
3. 逐层对比 max_abs_err / mean_abs_err / max_rel_err

## 快速开始

```bash
# 确保环境
export PYTHONPATH=/share/project/eai_pwm/home/hcr/repos/wm-training/src:$PYTHONPATH

# Step 1: dump 线上模型激活（单卡，无 FSDP）
python scripts/align/dump_online_activations.py \
    --checkpoint /share/project/eai_pwm/repos/wm-training/outputs/qwen3vl_gpic_patchlatent_2dpos_adaln_fm_logitnormal_buckets_512_stage2_from_step145000_fsdp/checkpoints/step_275000.dcp \
    --vlm-path /share/project/eai_pwm/models/Qwen3-VL-4B-Instruct \
    --diffusion-path /share/project/eai_pwm/models/Qwen3-VL-2B-Instruct \
    --vae-path /share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors \
    --prepared-root /share/project/eai_pwm/prepared_datasets/gpic_train_v1_94_incremental \
    --output alignment_fixture.pt

# Step 2: 在 wm-raw 中回放并对比
python scripts/align/align_check.py replay \
    --reference alignment_fixture.pt \
    --checkpoint /share/project/eai_pwm/repos/wm-training/outputs/.../step_275000.dcp \
    --atol 1e-3
```

## 对比点

| 组件 | Hook 位置 | 数量 |
|------|-----------|------|
| VLM decoder layers | 每层输出 hidden_states | 36 |
| Cross-attention K/V | 每层 adapter 的 k_proj/v_proj 输出 | 28 |
| Diffusion input_proj | projection 后的 hidden | 1 |
| Diffusion time_embedder | sinusoidal embedding 输出 | 1 |
| Diffusion time_conditioner | Linear 输出 | 1 |
| Diffusion AdaLN | 每层 6 个 modulation 向量 | 28 |
| Diffusion decoder layers | 每层输出 hidden_states | 28 |
| Diffusion final_norm | RMSNorm 输出 | 1 |
| Diffusion output_head | velocity prediction | 1 |
| Loss | MSE scalar | 1 |

## 输出格式

```
  [vlm.layer*]
    ✓ vlm.layer.0: shape=[1, 128, 2560], max_abs=1.2e-5, mean_abs=3.1e-6, max_rel=8.4e-4
    ✓ vlm.layer.1: shape=[1, 128, 2560], max_abs=3.4e-5, mean_abs=7.2e-6, max_rel=1.1e-3

  [diffusion.layer*]
    ✓ diffusion.layer.0: shape=[1, 1024, 2048], max_abs=2.1e-4, ...
    ✗ diffusion.layer.5: shape=[1, 1024, 2048], max_abs=5.2e-2, ...  ← diverge 定位

Total: 126 comparisons, 123 passed, 3 failed
```

## 子命令

| 命令 | 用途 |
|------|------|
| `dump_online_activations.py` | 加载 wm-training 模型，forward 一次，保存全部中间激活 + fixture batch |
| `align_check.py smoke` | wm-raw 单卡 forward 冒烟测试（合成数据） |
| `align_check.py replay --reference <file>` | 加载 fixture 在 wm-raw 回放，逐层对比 |
| `align_check.py compare --reference <a> --activations <b>` | 离线对比两个 activation dump |

## 环境要求

- 单卡 GPU（建议 A100 80GB 或 4090 24GB）
- bf16 精度
- `torch.backends.cudnn.deterministic = True`
- 固定 seed=42
- 不开 FSDP / torch.compile

## 精度预期

- bf16 运算误差：单层 ~1e-4, 累积 28 层后 ~1e-3
- `atol=1e-3` 是合理阈值（超过说明逻辑不一致而非精度累积）
- 如果某层突然从 1e-4 跳到 1e-2+，基本是该层实现有 bug

## 文件说明

```
scripts/align/
├── README.md                      # 本文件
├── dump_online_activations.py     # wm-training 激活 dump 工具
└── align_check.py                 # wm-raw 回放 + 对比工具
```
