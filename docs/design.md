# wm-raw 设计文档

> 目标：构建一个静态、可读、CUDAGraph-ready 的 World Model 训练 codebase，
> 模型结构对齐 training_code 中 `Qwen3VLBagelModel`，优先支持 GPIC 数据 + VLM/State Diffusion 分支。

---

## 1. 整体目标与约束

| 维度 | 说明 |
|------|------|
| **数值对齐** | 给定相同的 checkpoint 和输入，forward/loss 数值必须 bit-exact（bf16 精度内）|
| **静态化** | 所有 tensor 路径的 shape 在 compile-time 确定，不使用 `nonzero`/`.item()`/`tolist()`/动态 indexing |
| **CUDAGraph 友好** | 最终可在非 FSDP 场景（如 DDP + CUDAGraph）下 capture 整个 forward+backward |
| **可读性** | 单文件 < 500 行，模块职责单一，关键 tensor 操作有 shape 注释 |
| **最小范围** | 本期只实现 GPIC 数据的 VLM + State Diffusion 路径（不含 action、trajectory） |

---

## 2. 当前 codebase 问题分析

### 2.1 可读性问题
- `modeling.py` 3467 行，混杂了 VLM backbone 加载、视觉特征提取、cross-attention、diffusion branch、loss 计算
- 大量 `_call_first_supported_signature` / `_backbone_feature_extractor` 等兼容性 workaround，为了适配不同 transformers 版本
- 通过运行时 monkey-patch `attention.forward` 实现 cross_kv_concat（`_forward_qwen3vl_decoder_layer_cross_kv_concat`）
- 模块间通过 `Mapping[str, Any]` 字典传递，缺乏类型信息

### 2.2 静态化问题
- transformers 的 Qwen3VL 实现中 `compute_3d_position_ids` 使用 `.item()`，vision model 使用 `.tolist()`
- 当前通过 `torch.compiler.disable` 包装绕过，不是根本解决
- `_deepstack_process` 中 boolean indexing → cumsum+gather 的 patch 是编译期 workaround

### 2.3 架构耦合
- VLM backbone 通过 `AutoModelForImageTextToText.from_pretrained` 加载，整个 transformers 模型树成为依赖
- Diffusion backbone 同样如此，加载后再定位 layers/norm/rotary 等子模块
- Cross attention 的 KV projection 与 layer mapping 逻辑分散在多处

---

## 3. 新 Codebase 目录结构

```
wm-raw/
├── configs/
│   └── gpic_vlm_diffusion.yaml          # 对标当前 v1.95 config
├── docs/
│   └── design.md                         # 本文档
├── src/
│   └── wm_raw/
│       ├── __init__.py
│       ├── config.py                     # 全局配置 dataclass（强类型）
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   ├── qwen3vl_backbone.py       # Qwen3-VL transformer blocks（自包含实现）
│       │   ├── vision_encoder.py         # ViT / visual 前端（静态 shape）
│       │   ├── vlm.py                    # VLM 分支：embed → layers → lm_head
│       │   ├── diffusion_branch.py       # State diffusion 分支
│       │   ├── cross_attention.py        # Cross-attention stack（独立文件）
│       │   ├── adaln.py                  # AdaLN-Zero timestep conditioning
│       │   ├── embeddings.py             # Latent codec、position embeddings
│       │   ├── rope.py                   # MRoPE 实现（静态化，无 .item()）
│       │   └── model.py                  # 顶层 WorldModel：组装 VLM + Diffusion
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── gpic_dataset.py           # GPIC prepared dataset 加载
│       │   ├── collator.py               # VLM / Diffusion collator
│       │   └── schedule.py               # Deterministic microbatch schedule
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py                # 训练循环主逻辑
│       │   ├── optimizer.py              # 分组 optimizer 构建
│       │   ├── distributed.py            # FSDP2 / DDP 设置
│       │   ├── ema.py                    # EMA manager
│       │   └── checkpoint.py             # checkpoint save/load
│       │
│       ├── vae/
│       │   ├── __init__.py
│       │   └── frozen_codec.py           # Frozen BAGEL VAE（encode only）
│       │
│       └── utils/
│           ├── __init__.py
│           └── diagnostics.py            # Tensor stats / logging utilities
│
├── scripts/
│   └── train.py                          # 入口脚本
├── pyproject.toml
└── README.md
```

---

## 4. 模型架构详细设计

### 4.1 总览

```
┌─────────────────────────────────────────────────────────┐
│                    WorldModel                            │
│                                                         │
│  ┌──────────────┐                ┌───────────────────┐  │
│  │  VLMBranch   │                │ StateDiffusion    │  │
│  │              │    cross-attn  │    Branch         │  │
│  │  VisionEnc   │───────────────▶│                   │  │
│  │  TextEmbed   │  (KV from VLM) │  LatentEmbed     │  │
│  │  QwenLayers  │                │  QwenLayers      │  │
│  │  LMHead      │                │  AdaLN + Gate    │  │
│  │              │                │  OutputHead      │  │
│  └──────────────┘                └───────────────────┘  │
│                                                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │  CrossAttentionStack                               │ │
│  │  Per-layer: K_proj, V_proj, Gate (from VLM→Diff)   │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 4.2 VLM Branch (`vlm.py`)

**职责**：接收 text tokens + visual tokens，产出 hidden states（供 cross attention）和 AR loss。

```python
class VLMBranch(nn.Module):
    """Qwen3-VL 4B VLM backbone for conditioning.

    Produces:
      - ar_loss: cross-entropy on text tokens
      - hidden_states: List[Tensor]  # [B, S_vlm, D_vlm] per selected layer
    """

    def __init__(self, config: VLMConfig):
        # vision_encoder: VisionEncoder
        # embed_tokens: nn.Embedding
        # layers: nn.ModuleList[QwenDecoderLayer]  (N=36 for 4B)
        # norm: RMSNorm
        # lm_head: nn.Linear
        ...

    def forward(
        self,
        input_ids: Tensor,          # [B, S_vlm]
        pixel_values: Tensor,       # [B, N_patches, C_patch]
        image_grid_thw: Tensor,     # [B, 3]  (static: 固定 image size → 固定 grid)
        attention_mask: Tensor,     # [B, S_vlm]
        position_ids: Tensor,       # [3, B, S_vlm]  (MRoPE: temporal, height, width)
        labels: Optional[Tensor],   # [B, S_vlm]
    ) -> VLMOutput:
        ...
```

**关键设计决策**：
- **不依赖 transformers 的 model 类**，自己用 `QwenDecoderLayer` 堆叠
- `image_grid_thw` 在 fixed_seqlen 模式下是常量（256x256 图 → 固定 patch 数），编译时可折叠
- 中间 hidden states 通过 layer index 列表抽取（`hidden_state_layer_offset=1` → 取 layer[i+1] 给 diffusion layer[i]）

### 4.3 Qwen3-VL Decoder Layer (`qwen3vl_backbone.py`)

**核心复用单元**：VLM 和 Diffusion 共享同一套 layer 实现，只是参数不同。

```python
class QwenDecoderLayer(nn.Module):
    """Single Qwen3-VL transformer block.

    Components:
      - input_layernorm: RMSNorm
      - self_attn: QwenAttention (GQA with MRoPE)
      - post_attention_layernorm: RMSNorm
      - mlp: QwenMLP (SwiGLU)

    Shape annotations:
      hidden: [B, S, D]
      attention_mask: [B, 1, S, S+S_kv]  (causal or custom)
    """
```

**静态化要点**：
- Attention 计算使用 `F.scaled_dot_product_attention`，不走 eager math path
- RoPE 的 cos/sin 表预计算为固定 buffer（max_seq_len 已知），forward 时只做 slice
- 无动态 shape 操作：pad_to_max、attention_mask 预构建

### 4.4 Vision Encoder (`vision_encoder.py`)

```python
class VisionEncoder(nn.Module):
    """Qwen3-VL ViT (675M params in 4B model).

    固定输入：
      - image_size=256 → patch_size=14 → grid=(1, 18, 18) → 324 visual tokens
      - 由于 vlm_image_size=256 固定，visual token 数量在编译期确定

    输出：[B, 324, D_vision]，经过 merger 后维度对齐到 D_text
    """
```

**静态化**：
- 原始 transformers 实现中 `_get_vision_grid` 使用 `.tolist()`
- 我们对固定 image_size 直接硬编码 grid shape，消除动态计算

### 4.5 Diffusion Branch (`diffusion_branch.py`)

```python
class StateDiffusionBranch(nn.Module):
    """Image latent flow-matching diffusion，conditioned on VLM hidden states.

    输入：
      - noisy_latent: [B, S_lat, D_lat]   # VAE latent tokens (patchified)
      - timesteps: [B]                     # flow matching t ∈ [0, 1]
      - vlm_hidden_states: List[Tensor]    # cross-attention 条件

    输出：
      - prediction: [B, S_lat, D_lat]      # velocity field v(x_t, t)
      - loss: scalar MSE
    """

    def __init__(self, config: DiffusionConfig):
        # latent_in_proj: nn.Linear(D_lat_raw, D_diff)
        # time_embedder: TimestepEmbedder → [B, D_time]
        # time_conditioner: nn.Linear(D_time, D_diff)  # input_add mode
        # peer_conditioner: PeerConditioner  (gate-based)
        # layers: nn.ModuleList[QwenDecoderLayer]  (N=28 for 2B)
        # norm: RMSNorm
        # output_head: nn.Linear(D_diff, D_lat_raw)
        ...
```

**数据流**（对齐 `_forward_single_diffusion_branch` → `_run_prepared_single_diffusion_branch`）：
1. VAE encode → patchify → `latent_in_proj`
2. Timestep embed + `time_conditioner` → add to hidden
3. 对每个 decoder layer：
   - Cross attention inject VLM KV
   - Self attention (causal=False for diffusion)
   - MLP
4. `norm` → `output_head` → velocity prediction
5. Flow matching MSE loss

### 4.6 Cross Attention (`cross_attention.py`)

当前 config 指定 `communication_policy: cross_kv_down`（K/V 从 VLM 映射到 diffusion space）。

```python
class CrossAttentionStack(nn.Module):
    """Per-layer cross-attention from diffusion to VLM.

    For each diffusion layer i, maps VLM hidden[layer_map[i]] through:
      - k_proj: Linear(D_vlm, num_kv_heads * head_dim)
      - v_proj: Linear(D_vlm, num_kv_heads * head_dim)
      - gate: learnable scalar (init=0.01)

    cross_kv_down 策略：
      diffusion_query @ (k_proj(vlm_hidden)).T → attention → v_proj(vlm_hidden) → gate → add to hidden

    区别于 cross_kv_concat（将 KV 拼接到 self-attention 的 KV 中），
    cross_kv_down 在 self-attention 之后做独立的 cross-attention add。
    """

    def __init__(self, config: CrossAttentionConfig):
        # layers: nn.ModuleList  (一层 per diffusion layer)
        # 每层包含: k_proj, v_proj, q_proj (from diffusion hidden), o_proj, gate
        ...

    def condition_layer(
        self,
        layer_idx: int,
        hidden: Tensor,              # [B, S_diff, D_diff]
        vlm_kv: Tensor,              # [B, S_vlm, D_vlm]
        attention_mask: Tensor,      # [B, S_diff, S_vlm] or broadcastable
    ) -> Tensor:
        """Returns hidden + gate * cross_attn(hidden, vlm_kv)"""
        ...
```

### 4.7 Layer Mapping (`models/model.py` 内部)

配置 `layer_mapping_policy: middle_n`：
- VLM 有 36 层（4B），Diffusion 有 28 层（2B）
- middle_n 策略：diffusion layer[i] → VLM layer[offset + i]
- `hidden_state_layer_offset=1`：取 VLM layer[mapped+1] 的输出作为 KV source

### 4.8 Latent Tokenization

```python
# VAE: 256x256 image → [B, 16, 32, 32] latent
# Patchify (patch_size=2): → [B, 16*16, 16*4] = [B, 256, 64]
# latent_in_proj: → [B, 256, D_diff]
```

Position embedding: `bagel_2d_sincos` 基于 patch grid 的 2D sincos 位置编码。

---

## 5. 静态化策略

### 5.1 消除动态 shape 来源

| 动态来源 | 当前 workaround | 新方案 |
|----------|----------------|--------|
| `compute_3d_position_ids` 中 `.item()` | `torch.compiler.disable` | 预计算固定 position_ids buffer |
| VisionModel `.tolist()` | `torch.compiler.disable` | 固定 grid，硬编码 reshape 参数 |
| `_deepstack_process` boolean indexing | cumsum+gather patch | 不需要 deepstack（自己实现 ViT） |
| 动态 sequence length | `fixed_seqlen` pad/truncate | 从 collator 层保证固定 shape |
| `attention_mask` 形状不一致 | 运行时判断 ndim | 统一为 `[B, 1, S_q, S_kv]` 4D mask |

### 5.2 固定 shape 契约

所有 forward 的 tensor shapes 在配置加载时即确定：

```python
@dataclass
class ShapeContract:
    """编译期可确定的所有 shape 常量"""
    batch_size: int = 4
    vlm_seq_len: int = 640          # fixed_seqlen.vlm_max_seq_len
    condition_seq_len: int = 256     # fixed_seqlen.condition_max_seq_len
    image_size: int = 256
    visual_tokens: int = 324         # (256/14)^2 ≈ 324 after merge
    latent_h: int = 32               # VAE latent height
    latent_w: int = 32               # VAE latent width
    latent_channels: int = 16
    patch_size: int = 2
    diffusion_seq_len: int = 256     # (32/2) * (32/2) = 256 patched tokens
    vlm_hidden_dim: int = 3584       # Qwen3-VL-4B
    diff_hidden_dim: int = 1536      # Qwen3-VL-2B
```

### 5.3 CUDAGraph Readiness

为后续 CUDAGraph capture 做好准备：
- **无 Python 控制流依赖 tensor 值**：所有 if/else 只依赖 config 常量
- **无动态 allocation**：buffer 预分配，mask 预构建
- **无 NCCL 调用在 capture 区域内**（DDP 模式下 grad sync 在 backward 后）
- 当前 FSDP2 模式下用 `mode=default`（inductor codegen），未来 DDP 模式可切 `reduce-overhead`

---

## 6. 从 transformers 加载权重的策略

不依赖 transformers model class，但需要兼容 HF checkpoint format：

```python
def load_qwen3vl_from_hf(path: str, *, model: WorldModel) -> None:
    """从 HF safetensors 加载权重到自定义模型结构。

    映射规则：
      HF: model.language_model.model.layers.{i}.self_attn.q_proj.weight
      Ours: vlm.layers[i].self_attn.q_proj.weight

      HF: model.visual.{...}
      Ours: vlm.vision_encoder.{...}

      HF (diffusion backbone): model.language_model.model.layers.{i}.*
      Ours: diffusion.layers[i].*
    """
```

提供一个显式的 state_dict 映射表，而不是运行时动态探测。

---

## 7. 数据流设计

### 7.1 GPIC Prepared Dataset

当前 config 使用 `dataset_type: wm_sequence_prepared`，`prepared_view: image_caption`。

数据已经预处理为 tokenized sequences，存储在 safetensors shards 中。

```
每个 sample 包含:
  - input_ids: [S_vlm]       (text tokens, 含 image placeholder)
  - attention_mask: [S_vlm]
  - labels: [S_vlm]          (AR target, -100 for non-text)
  - pixel_values: [N, C, H, W]  (原始图片 for VLM visual enc)
  - image_grid_thw: [N, 3]
  - target_image: [3, 256, 256]  (diffusion target)
```

### 7.2 Collator

VLM collator:
- Pad/truncate to `vlm_max_seq_len=640`
- Resize image to fixed 256x256（确保 visual_tokens=324）

Diffusion collator:
- Pad/truncate condition to `condition_max_seq_len=256`
- VAE encode target_image → patchify → produce `state_target: [B, 256, 64]`
- Sample timesteps, noise

### 7.3 Training Step

```
1. VLM microbatch:
   vlm_output = model.vlm_forward(vlm_batch)
   ar_loss = vlm_output.loss

2. Diffusion microbatch:
   # Condition encoding (VLM forward on condition text + image)
   vlm_hidden = model.vlm_forward(condition_batch).hidden_states

   # VAE encode (outside compile graph)
   latent = vae.encode(target_image)

   # Diffusion forward
   diff_output = model.diffusion_forward(latent, vlm_hidden, timesteps)
   state_loss = diff_output.loss

3. Total loss = vlm_loss_weight * ar_loss + state_loss_weight * state_loss
4. Backward + optimizer step
```

---

## 8. 训练循环设计

### 8.1 Deterministic Microbatch Schedule

每步交替：1 VLM microbatch + 1 Diffusion microbatch（对齐当前 config）。

### 8.2 Optimizer

三组学习率（对齐现有 config）：
- `adapter_learning_rate: 1e-4`（cross attention、gate 等新增参数）
- `diffusion_learning_rate: 1e-5`（diffusion backbone）
- `vlm_learning_rate: 1e-6`（VLM backbone）

AdamW, β=(0.9, 0.95), weight_decay=0, max_grad_norm=1.0

### 8.3 EMA

对 optimizer tracked params 做 EMA（decay=0.9999），存入 checkpoint。

---

## 9. 可读性规范

### 9.1 代码规范

- 每个文件 < 500 行（hard limit），超出则拆分
- 所有 `forward` 方法的输入输出加 shape 注释：
  ```python
  def forward(
      self,
      hidden: Tensor,          # [B, S, D]
      attention_mask: Tensor,  # [B, 1, S, S]
  ) -> Tensor:                 # [B, S, D]
  ```
- 关键 reshape/permute 操作附带 shape 变换注释：
  ```python
  # [B, H*W, C] → [B, H, W, C] → [B, pH, pS, pW, pS, C] → [B, pH*pW, pS*pS*C]
  grid = tokens.reshape(batch, height, width, channels)
  ```
- 模块 docstring 说明：职责、输入输出、与论文/原始实现的对应关系

### 9.2 命名规范

- Layer/module 用全称：`self_attention` 而非 `self_attn`（除非对齐 checkpoint key）
- Tensor 变量名体现含义：`vlm_hidden_states` 而非 `hs`
- 配置字段用完整描述性名称

### 9.3 不做的事

- 不做 transformers 版本兼容（`_call_first_supported_signature` 类逻辑全部移除）
- 不做动态模块发现（`_backbone_feature_extractor` / `_find_layers` 全部移除）
- 不支持多种 communication_policy 的运行时切换——本期只实现 `cross_kv_down`
- 不支持 action branch（本期不需要）

---

## 10. 数值对齐验证计划

1. **逐层对齐**：加载相同 checkpoint，对比每层 hidden state 的数值差异
2. **Forward 对齐**：给定相同输入 batch，对比 VLM logits、diffusion prediction
3. **Loss 对齐**：对比 ar_loss、state_diffusion_loss
4. **训练对齐**：跑 100 步，对比 loss curve 是否 match（允许 bf16 累积误差）

---

## 11. 实施优先级

| 阶段 | 内容 | 预期工作量 |
|------|------|-----------|
| P0 | `qwen3vl_backbone.py` + `rope.py`：Qwen3-VL decoder layer 自包含实现 | 2-3 天 |
| P1 | `vision_encoder.py`：静态化 ViT 实现 | 1-2 天 |
| P2 | `vlm.py`：VLM 分支组装 + AR loss | 1 天 |
| P3 | `cross_attention.py` + `diffusion_branch.py`：扩散分支 + cross attn | 2 天 |
| P4 | `embeddings.py` + `model.py`：latent tokenization + 顶层组装 | 1 天 |
| P5 | `data/` + `training/`：数据流 + 训练循环 | 2-3 天 |
| P6 | 权重加载 + 数值对齐验证 | 2 天 |
| P7 | torch.compile + CUDAGraph 验证 | 1-2 天 |

---

## 12. 与现有 Repo 的关键差异总结

| 维度 | 现有 repo (training_code) | 新 repo (wm-raw) |
|------|--------------------------|-------------------|
| 模型加载 | `AutoModelForImageTextToText` 继承 | 自包含实现 + HF weight mapping |
| 可读性 | 3400+ 行单文件 | 多文件，每文件 < 500 行 |
| 静态化 | 编译期 patch + compiler.disable | 原生静态设计 |
| 兼容性 | 多 transformers 版本兼容 | 只支持目标 checkpoint format |
| 灵活性 | 多 communication_policy、多任务 | 只实现 cross_kv_down + GPIC |
| CUDAGraph | 不支持（FSDP2 冲突） | 设计为 CUDAGraph-ready |
| 类型安全 | `Mapping[str, Any]` 字典 | 强类型 dataclass + TypedDict |
