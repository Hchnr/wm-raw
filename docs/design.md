# wm-raw 设计文档

> 目标：构建一个可读、compile-ready 的 World Model 训练 codebase，
> 模型结构对齐 wm-training 中 `Qwen3VLBagelModel`，优先支持 GPIC 数据 + State Diffusion 分支。
> 训练栈：PyTorch FSDP2 + DCP + torchrun + bf16 + torch.compile。

---

## 1. 整体目标与约束

| 维度 | 说明 |
|------|------|
| **数值对齐** | 给定相同 checkpoint 和输入，forward/loss 数值 bit-exact（bf16 精度内）|
| **compile 友好** | torch.compile(mode="default") 可覆盖完整 forward+backward |
| **可读性** | 单文件 < 500 行，模块职责单一，关键 tensor 操作有 shape 注释 |
| **最小范围** | 本期只实现 GPIC 数据的 State Diffusion 路径（VLM 作为 condition encoder） |
| **生产对齐** | 对标线上 config：`cross_kv_concat` + `adaln_zero` + resolution buckets + logit-normal |

---

## 2. 线上配置摘要（对齐目标）

当前线上使用的配置关键参数：

```yaml
model:
  vlm_path: Qwen3-VL-4B-Instruct
  diffusion_path: Qwen3-VL-2B-Instruct
  communication_policy: cross_kv_concat        # VLM KV concat 到 self-attn
  layer_mapping_policy: middle_n
  hidden_state_layer_offset: 1
  cross_attention_gate_init: 0.01
  latent:
    objective: flow_matching
    prediction_type: flow
    timestep_shift: 1.0
    timestep_sampling:
      type: logit_normal                       # NOT uniform
      mean: 0.0
      std: 1.0
    timestep_conditioning: adaln_zero          # AdaLN-Zero modulation
    tokenization: patchified
    patch_size: 2
    position_embedding: bagel_2d_sincos
    max_position_size: 64

data:
  resolution_buckets:
    sizes: [[512,512], [448,608], [608,448], [416,640], ...]
  tasks:
    - cfg_dropout_mode: sentinel_only
      text_condition_dropout_prob: 0.1

multitask:
  vlm_microbatches_per_step: 0                 # VLM only as condition encoder
  diffusion_microbatches_per_step: 1
  trainable_mode: diffusion
  train_diffusion_backbone: true

optimizer:
  adapter_learning_rate: 1.0e-4
  diffusion_learning_rate: 1.0e-4              # same as adapter
  vlm_learning_rate: 0.0                       # VLM frozen
  fused: true

training:
  torch_compile: {enabled: true, mode: default, dynamic: false}
```

---

## 3. Codebase 目录结构

```
wm-raw/
├── configs/
│   └── gpic_vlm_diffusion.yaml
├── docs/
│   └── design.md                         # 本文档
├── src/
│   └── wm_raw/
│       ├── __init__.py
│       ├── config.py                     # 全局配置 dataclass（强类型）
│       ├── diffusion.py                  # Flow matching + timestep sampling
│       ├── checkpoint.py                 # 权重加载（HF / online DCP 映射）
│       ├── training.py                   # FSDP2 训练循环
│       ├── train.py                      # CLI 入口
│       ├── models/
│       │   ├── model.py                  # WorldModel 顶层组装
│       │   ├── vlm.py                    # VLM 分支（condition encoder）
│       │   ├── diffusion_branch.py       # StateDiffusionBranch + DiffusionDecoderLayer
│       │   ├── cross_attention.py        # CrossAttentionStack（cross_kv_concat）
│       │   ├── adaln.py                  # AdaLN-Zero + SinusoidalTimestepEmbedding
│       │   ├── embeddings.py             # patchify/unpatchify + BagelGridPositionEmbedding
│       │   ├── qwen3vl_backbone.py       # DecoderLayer, TextAttention, TextMLP, RMSNorm
│       │   ├── rope.py                   # MRoPE（text）+ VisionRotaryEmbedding
│       │   └── vision_encoder.py         # Qwen3-VL ViT
│       ├── data/
│       │   ├── dataset.py / prepared_dataset.py
│       │   └── collator.py
│       ├── vae/
│       │   └── frozen_codec.py           # Frozen BAGEL VAE
│       └── utils/
│           ├── ema.py
│           └── diagnostics.py
├── scripts/
│   ├── train.py
│   └── align_check.py                   # 数值对齐验证脚本
└── pyproject.toml
```

---

## 4. 模型架构详细设计

### 4.1 总览

```
┌─────────────────────────────────────────────────────────────┐
│                       WorldModel                             │
│                                                             │
│  ┌──────────────┐                ┌────────────────────────┐ │
│  │  VLMBranch   │                │  StateDiffusionBranch  │ │
│  │  (frozen)    │  cross_kv_concat│                        │ │
│  │              │────────────────▶│  LatentEmbed + PosEmb  │ │
│  │  VisionEnc   │  KV concat to  │  Timestep(input_add)   │ │
│  │  TextEmbed   │  self-attn KV  │  AdaLN-Zero per layer  │ │
│  │  QwenLayers  │                │  DiffusionDecoderLayers│ │
│  │  (36 layers) │                │  (28 layers)           │ │
│  └──────────────┘                │  OutputHead            │ │
│                                  └────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  CrossAttentionStack (28 adapters)                     │ │
│  │  Per-layer: context_norm → k_proj, v_proj              │ │
│  │  (q_proj/o_proj/gate unused in cross_kv_concat mode)   │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 VLM Branch（condition encoder）

线上配置 `vlm_learning_rate: 0.0`，VLM 完全冻结，仅作为 condition encoder。

```python
class VLMBranch(nn.Module):
    """Qwen3-VL 4B — 产出 hidden states 供 cross attention。
    
    输入: condition text tokens + (可选) image
    输出: hidden_states: List[Tensor] [num_layers+1 × (B, S_vlm, 2560)]
    """
```

- 36 层 Qwen3-VL decoder + vision encoder
- 中间 hidden states 按 `layer_map + offset` 选取给 diffusion 的每一层
- 不计算 AR loss（线上 `vlm_microbatches_per_step: 0`）

### 4.3 Cross Attention — `cross_kv_concat` 策略

**核心区别于 `cross_kv_down`**：

- `cross_kv_down`：独立的 cross-attention 块，有自己的 Q/K/V/O proj + gate，在 self-attn 之后做 gated residual add
- `cross_kv_concat`：VLM context 只经过 `context_norm → k_proj / v_proj`，产出的 K/V **直接 prepend 到 self-attention 的 K/V** 中

```python
# cross_kv_concat 数据流（在 diffusion self-attention 内部）：
Q = self_attn.q_proj(diffusion_hidden)           # [B, H, S_diff, D]
K_self = self_attn.k_proj(diffusion_hidden)      # [B, H_kv, S_diff, D]
V_self = self_attn.v_proj(diffusion_hidden)      # [B, H_kv, S_diff, D]

K_ext, V_ext = adapter.project_context_kv(vlm_hidden)  # [B, H_kv, S_vlm, D]

K = cat(K_ext, K_self, dim=-2)  # [B, H_kv, S_vlm+S_diff, D]
V = cat(V_ext, V_self, dim=-2)  # [B, H_kv, S_vlm+S_diff, D]

out = SDPA(Q, K, V, mask=combined_mask)  # 一次统一 attention
```

**关键**：`cross_kv_concat` 模式下 adapter 的 `q_proj`、`o_proj`、`gate` 参数不使用（已被 freeze）。只有 `context_norm`、`k_proj`、`v_proj` 参与计算。

### 4.4 Layer Mapping

`layer_mapping_policy: middle_n` + `hidden_state_layer_offset: 1`：
- VLM 36 层，Diffusion 28 层
- `middle_n`：start = (36-28)//2 = 4，diffusion layer[i] → VLM layer[4+i]
- `offset=1`：取 VLM layer[4+i+1] 的输出（即 layer 之后的 hidden state）

### 4.5 Timestep Conditioning（双重）

线上使用 **input_add + adaln_zero** 双重 conditioning：

**1. Input Add（全局，在 decoder layers 之前）：**
```python
time_hidden = time_embedder(timesteps)           # [B, D]
time_cond = time_conditioner(time_hidden)        # [B, D]  (Linear, no bias)
hidden = hidden + time_cond[:, None]             # broadcast 到所有 token
```

**2. AdaLN-Zero（per-layer，modulate each decoder layer）：**
```python
class AdaLNZero:
    # Per layer: SiLU → Linear(D, 6D), zero-initialized
    # Produces: shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp

# In DiffusionDecoderLayer:
normed = input_layernorm(x) * (1 + scale_attn) + shift_attn  # modulated norm
attn_out = self_attn(normed, ...)
x = x + gate_attn * attn_out                                  # gated residual

normed = post_attn_layernorm(x) * (1 + scale_mlp) + shift_mlp
mlp_out = mlp(normed)
x = x + gate_mlp * mlp_out
```

Zero-init 意味着训练开始时 diffusion branch 退化为标准 pretrained LLM layer。

### 4.6 Latent Tokenization + Position Embedding

```python
# VAE (BAGEL AE, downsample 8x):
#   512x512 image → [B, 16, 64, 64] latent
#   448x608 image → [B, 16, 56, 76] latent
#   (实际 stride=16 with temporal_patch=2: H_lat = H_img/8, W_lat = W_img/8)

# Patchify (patch_size=2):
#   [B, 16, 64, 64] → [B, 32*32, 2*2*16] = [B, 1024, 64]
#   [B, 16, 56, 76] → [B, 28*38, 64] = [B, 1064, 64]

# Position Embedding: BagelGridPositionEmbedding
#   Frozen lookup table [max_position_size^2, hidden_size]
#   pos_id = row * max_position_size + col
#   Added to projected hidden states
```

`max_position_size=64` 允许 latent grid 最大 64×64 patches（对应 512×512 图像经 8x 下采样再 2x patchify）。

### 4.7 Timestep Sampling — logit-normal

线上使用 `logit_normal` 而非 uniform：

```python
def sample_timesteps_logit_normal(batch_size, *, mean=0.0, std=1.0, shift=1.0):
    """Sample t ~ sigmoid(Normal(mean, std)), then apply rectified flow shift."""
    z = torch.randn(batch_size) * std + mean
    t = torch.sigmoid(z)  # t ∈ (0, 1), concentrated around sigmoid(mean)=0.5
    if shift != 1.0:
        t = shift * t / (1 + (shift - 1) * t)
    return t.clamp(1e-5, 1 - 1e-5)
```

logit-normal 相比 uniform 更集中于中间 timestep，训练效率更高。

### 4.8 CFG Dropout（sentinel_only）

`text_condition_dropout_prob: 0.1` + `cfg_dropout_mode: sentinel_only`：
- 训练时 10% 概率将 condition text 替换为 unconditional sentinel
- `sentinel_only`：unconditional text = `condition_suffix.strip()` = `"<|wm_predict_image|>"`
- 推理时用 classifier-free guidance：`pred = uncond + scale * (cond - uncond)`

---

## 5. 静态化与 compile 策略

### 5.1 Resolution Buckets 下的 compile 策略

线上使用 7 种 bucket sizes，每种 bucket 产生不同的 latent token 数：

| Bucket | Latent (H/8) | Patch (÷2) | Tokens |
|--------|-------------|------------|--------|
| 512×512 | 64×64 | 32×32 | 1024 |
| 448×608 | 56×76 | 28×38 | 1064 |
| 608×448 | 76×56 | 38×28 | 1064 |
| 416×640 | 52×80 | 26×40 | 1040 |
| 640×416 | 80×52 | 40×26 | 1040 |
| 384×704 | 48×88 | 24×44 | 1056 |
| 704×384 | 88×48 | 44×24 | 1056 |

**compile 策略**：`dynamic=false`，每种 bucket shape 会触发 1 次 recompile，之后缓存。线上设 `cache_size_limit: 16` 足够覆盖 7 种 bucket。同一 batch 内所有 sample 来自同一 bucket（由 `DistributedResolutionBucketBatchSampler` 保证），所以每次 forward 的 shape 是固定的。

### 5.2 compile 兼容设计

| 问题 | 解决 |
|------|------|
| 动态 seq_len | 同 batch 同 bucket，compile 缓存多套 |
| VLM position_ids 的 `.item()` | 由 collator 预计算好再传入 |
| AdaLN 和 external_kv 的条件分支 | 所有 diffusion layer 总是有 AdaLN + external_kv（不做 None 分支） |
| Cross-attn KV projection 的 layer_idx 依赖 | 在 layer loop 之前 batch project 所有层 |
| attention_mask 形状不一致 | 统一为 `[B, 1, S_q, S_kv]` 4D mask（预构建） |

---

## 6. 训练数据流

### 6.1 当前训练模式（diffusion-only）

```
每步仅 1 个 diffusion microbatch（vlm_microbatches=0）:

1. Collator 准备 batch:
   - condition: tokenized text + (optional image)
   - state_target: target image pixels → VAE encode → patchified latent
   - timesteps: logit_normal sampling
   - noise: randn_like(clean_tokens)

2. Forward:
   vlm_hidden = model.vlm_forward(condition)     # frozen, 产出 hidden_states
   prediction = model.diffusion_forward(
       noisy_tokens, timesteps,
       cross_attention_stack=model.cross_attention,
       vlm_hidden_states=vlm_hidden,
   )

3. Loss:
   loss = MSE(prediction, velocity_target)       # state_loss_weight=1.0

4. Backward + optimizer step (only adapter + diffusion params)
```

### 6.2 CFG Dropout 在 collator 中

```python
if random() < text_condition_dropout_prob:
    condition_text = "<|wm_predict_image|>"  # sentinel_only mode
else:
    condition_text = f"Caption: {caption} <|wm_predict_image|>"
```

---

## 7. Optimizer & Schedule

```python
# 两组参数（VLM frozen，不参与优化）:
param_groups = [
    {"params": adapter_params, "lr": 1e-4},      # cross_attention k_proj/v_proj/context_norm
    {"params": diffusion_params, "lr": 1e-4},    # diffusion backbone layers + adaln + output_head
]

optimizer = torch.optim.AdamW(
    param_groups,
    betas=(0.9, 0.95),
    weight_decay=0.0,
    fused=True,
)

scheduler = constant with warmup(2500 steps)
grad_clip = 1.0
```

---

## 8. 数值对齐验证计划

### 8.1 对齐策略

按 M1/M3 里程碑要求：**相同权重 + 相同输入 → forward 输出 bit-exact**。

对齐分三级：
1. **逐层对齐**（单 GPU，固定 seed）：每个模块独立对比
2. **Forward 对齐**：完整 forward pass 数值一致
3. **训练对齐**：100 步 loss 对比

### 8.2 逐层对齐方案（优先实现）

```
环境：单卡，bf16，固定 seed，不开 FSDP

输入：保存一个 fixture batch（tokenized condition + state_target + timesteps + noise）

对比点：
  ┌─ VLM ─────────────────────────────────────┐
  │ 1. vision_encoder output                   │
  │ 2. embed_tokens output                     │
  │ 3. per-layer hidden state (36 layers)      │
  └────────────────────────────────────────────┘
  ┌─ Cross Attention ─────────────────────────┐
  │ 4. per-layer projected K/V (28 layers)     │
  └────────────────────────────────────────────┘
  ┌─ Diffusion ───────────────────────────────┐
  │ 5. input_proj output                       │
  │ 6. position_embedding output               │
  │ 7. time_embedder + time_conditioner output │
  │ 8. per-layer AdaLN params (28 layers)      │
  │ 9. per-layer hidden state (28 layers)      │
  │ 10. final prediction                       │
  │ 11. loss value                             │
  └────────────────────────────────────────────┘

报告格式：
  Layer X: max_abs_err=1.2e-7, rel_err=3.4e-6, shape=[B,S,D] ✓
```

### 8.3 工具设计

两个脚本协作：

1. `scripts/dump_online_activations.py`：
   - 加载 wm-training checkpoint
   - 注册 forward hooks 捕获中间结果
   - 保存为 `activations.pt`

2. `scripts/align_check.py`：
   - 加载同一 checkpoint 到 wm-raw
   - 用相同 fixture 输入做 forward
   - 逐层对比与 `activations.pt` 的数值差异
   - 输出对齐报告

### 8.4 Seed 固定方案

```python
def set_deterministic(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)  # 部分 op 不支持
```

---

## 9. 与 refactor_v3.md 的 Milestone 对应

| 本文档内容 | 对应 Milestone |
|-----------|---------------|
| 模型架构实现（§4） | M1：模型结构迁移 |
| 数值对齐（§8） | M1 验收 + M3 |
| 训练循环（§6-7） | M2：跑通训练 |
| compile 策略（§5） | M2 性能优化 |

---

## 10. 实施优先级（更新）

| 阶段 | 内容 | 状态 |
|------|------|------|
| P0 | 模型架构（backbone + cross_attn + adaln + diffusion） | ✅ 已完成 |
| P1 | 权重加载（HF + online DCP mapping） | ✅ 已完成 |
| P2 | logit-normal timestep + variable latent sizes | 🔄 进行中 |
| P3 | 数值对齐工具（align_check.py） | 🔄 进行中 |
| P4 | FSDP2 训练循环 | ✅ 基本完成 |
| P5 | Resolution bucket collator | 待完成 |
| P6 | 2 卡 FSDP 对齐验证 | 待完成 |
