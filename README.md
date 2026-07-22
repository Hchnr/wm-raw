# wm-raw

静态、可读、CUDAGraph-ready 的 World Model 训练 codebase。

模型结构对齐 `wm-training` 中的 `Qwen3VLBagelModel`，当前支持 GPIC 数据 + VLM/State Diffusion 分支。

## 模型架构 (7.28B)

```
WorldModel
├── vlm (VLMBranch, ~4B)              ← Qwen3-VL-4B-Instruct
│   ├── embed_tokens
│   ├── vision_encoder (24 blocks)
│   ├── layers (36 DecoderLayers)
│   ├── norm
│   └── lm_head
├── state_diffusion (StateDiffusionBranch, ~2B)  ← Qwen3-VL-2B-Instruct
│   ├── input_proj
│   ├── time_embedder + time_conditioner
│   ├── adaln_layers (28 AdaLN modules)
│   ├── layers (28 DiffusionDecoderLayer)
│   ├── position_embedding
│   ├── final_norm
│   └── output_head
└── cross_attention (CrossAttentionStack)         ← random init
    └── 28 CrossAttentionAdapter layers
```

## 安装

```bash
pip install -e .

# 训练依赖（transformers 用于 tokenizer）
pip install -e ".[train]"
```

## 训练

### 配置文件

主配置：`configs/gpic_image_diffusion.yaml`

已对齐线上正式实验 (`qwen3vl_gpic_patchlatent_2dpos_adaln_fm_shift2_200k_fsdp.yaml`)。

### 单卡训练

```bash
python -m wm_raw.train --config configs/gpic_image_diffusion.yaml
```

### 多卡训练（FSDP2）

```bash
# 2 卡
torchrun --nproc_per_node=2 -m wm_raw.train --config configs/gpic_image_diffusion.yaml

# 8 卡
torchrun --nproc_per_node=8 -m wm_raw.train --config configs/gpic_image_diffusion.yaml
```

### 从 checkpoint 恢复

```bash
torchrun --nproc_per_node=8 -m wm_raw.train \
  --config configs/gpic_image_diffusion.yaml \
  --resume outputs/gpic_image_diffusion/checkpoints/step-005000
```

### 快速验证（小数据）

在配置文件中加 `max_samples` 限制数据量：

```yaml
data:
  max_samples: 100
  batch_size: 1
```

## 配置说明

| 字段 | 说明 | 线上默认值 |
|------|------|-----------|
| `model.vlm_path` | VLM 权重路径 | `/share/project/eai_pwm/models/Qwen3-VL-4B-Instruct` |
| `model.diffusion_path` | Diffusion backbone 权重 | `/share/project/eai_pwm/models/Qwen3-VL-2B-Instruct` |
| `vae.model_path` | VAE 权重 | `/share/project/eai_pwm/models/BAGEL-7B-MoT/ae.safetensors` |
| `optimizer.adapter_learning_rate` | Cross-attn 等新参数 LR | 1e-4 |
| `optimizer.diffusion_learning_rate` | Diffusion backbone LR | 5e-5 |
| `optimizer.vlm_learning_rate` | VLM LR（0 = 冻结） | 0.0 |
| `scheduler.type` | `constant` 或 `cosine` | constant |
| `scheduler.warmup_steps` | 线性 warmup 步数 | 5000 |
| `ema.enabled` | 是否开 EMA | true |
| `ema.decay` | EMA 衰减系数 | 0.9995 |
| `data.text_condition_dropout_prob` | CFG text dropout 概率 | 0.1 |
| `training.max_steps` | 总训练步数 | 200000 |
| `logging.wandb_enabled` | 是否开 WandB | true |

## 数据格式

`data.train_manifest` 指向 JSONL 文件，每行：

```json
{"image_path": "relative/path/to/image.png", "caption": "A cat sitting on a mat"}
```

`image_path` 相对于 `data.image_root` 解析。

## 测试

```bash
# CPU forward 测试（无需 GPU）
python tests/test_forward.py

# 单卡 GPU + torch.compile 冒烟测试
python tests/test_gpu.py

# 全量权重加载测试（需要 GPU + 模型文件）
python tests/test_full_model_load.py

# FSDP2 多卡训练测试（需要 ≥2 GPU）
torchrun --nproc_per_node=2 tests/test_fsdp2_training.py
```

## 项目结构

```
src/wm_raw/
├── config.py                  # 强类型配置 dataclass
├── diffusion.py               # Flow matching 噪声/损失
├── checkpoint.py              # HF→wm-raw 权重映射
├── training.py                # FSDP2 训练循环
├── train.py                   # CLI 入口 (torchrun)
├── models/
│   ├── rope.py                # MRoPE (静态)
│   ├── qwen3vl_backbone.py    # Qwen3-VL decoder layer
│   ├── vision_encoder.py      # ViT + PatchMerger
│   ├── vlm.py                 # VLM 分支
│   ├── diffusion_branch.py    # State diffusion 分支
│   ├── cross_attention.py     # Cross-attention stack
│   ├── adaln.py               # AdaLN-Zero timestep conditioning
│   ├── embeddings.py          # Latent patchify + position embed
│   └── model.py               # 顶层 WorldModel
├── data/
│   ├── dataset.py             # JSONL manifest dataset
│   └── collator.py            # VLM/Diffusion collator
├── vae/
│   └── frozen_codec.py        # 冻结 VAE encode/decode
└── utils/
    ├── diagnostics.py         # Tensor/gradient 诊断工具
    └── ema.py                 # EMA 参数平均
```
