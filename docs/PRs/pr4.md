# PR: Inference Alignment with wm-training

## Summary

This PR aligns wm-raw's inference pipeline with the online wm-training model, enabling correct image generation from shared checkpoints. Previously, generated images showed severe artifacts (color blocks, grey noise, NaN). After this PR, wm-raw produces visually identical results to wm-training's `batch_generate_image.py`.

## Critical Fixes (in order of impact)

### 1. MRoPE Interleave Implementation (root cause of color blocks)

**Problem**: wm-raw implemented MRoPE as simple concatenation `[T_freqs, H_freqs, W_freqs]`, but HF Qwen3-VL uses true interleaved assignment: T at indices 0,3,6...; H at 1,4,7...; W at 2,5,8...

**Also**: Diffusion position IDs were set to sequential `arange(1024)` (all 3 axes identical), but online uses 2D grid `[temporal=0, height=row_idx, width=col_idx]`.

**Impact**: Completely destroyed spatial structure in attention → mosaic/block artifacts.

**Fix**: `rope.py` `_apply_interleaved_mrope` + `diffusion_branch.py` position_ids construction.

### 2. RMSNorm dtype Conversion Order

**Problem**: wm-raw computed `(weight * x_fp32).to(bf16)`, but HF computes `weight * x_fp32.to(bf16)`. The multiplication happens in different precision.

**Impact**: 0.5 max error per norm call, accumulating through 36 VLM + 28 diffusion layers → complete divergence.

**Fix**: `qwen3vl_backbone.py` line 37: `return self.weight * x.to(input_dtype)`

### 3. Cross-Attention RMSNorm Type

**Problem**: Cross-attention `context_norm`/`query_norm` used Qwen's custom RMSNorm (with HF bf16 cast order), but online uses native `nn.RMSNorm`.

**Impact**: Introduced 0.06 error in projected K/V per layer.

**Fix**: `cross_attention.py` uses `nn.RMSNorm` instead of custom `RMSNorm`.

### 4. Timestep Conditioning Mode

**Problem**: wm-raw always applied `input_add` (time_conditioner added to hidden), but with `adaln_zero` config, online skips `input_add` entirely — timestep info is injected exclusively via per-layer AdaLN modulation.

**Impact**: Double timestep conditioning → grey noise output (model outputs collapse to mean).

**Fix**: `diffusion_branch.py` `prepare_inputs` checks `latent_config.timestep_conditioning`.

### 5. Attention Mask Format

**Problem**: VLM causal mask used `tril(ones)` (0/1 multiplicative) instead of `triu(full(-inf))` (additive for SDPA). Also, `0 * -inf = NaN` when no padding.

**Impact**: VLM produced garbage hidden states → NaN or meaningless conditioning.

**Fix**: Proper additive mask with `masked_fill` for padding, `is_causal=True` fallback.

### 6. Resolution & Checkpoint Mapping

- Image resolution: 256px → 512px (1024 tokens instead of 256)
- `timestep_shift`: 2.0 → 1.0 (match training config)
- Checkpoint key mapping: added `input_projector.linear_1→proj`, `time_embedder.linear_1/2→mlp.0/2`, `position_embedding→pos_embed`

## Infrastructure Added

- `scripts/compare_inference.py` — side-by-side numerical comparison tool
- `scripts/align/` — alignment testing utilities
- `src/wm_raw/utils/timing.py` — reusable Timer/log_duration
- `generate_image.py` — supports .pt/.dcp checkpoints, batch eval mode with 9 difficulty levels, HF VLM for condition encoding

## Verification

- Layer-by-layer diff: **0.000000** (28 diffusion layers, identical inputs)
- MRoPE cos/sin: **0.000000** vs HF Qwen3VLTextRotaryEmbedding  
- Block ratio: **7.58 → 1.00** (smooth, no artifacts)
- Generated image visually matches wm-training output

---

# PR: 推理对齐 wm-training

## 概要

本 PR 将 wm-raw 的推理流程与线上 wm-training 模型完全对齐，实现从共享 checkpoint 正确生成图片。修复前生成图片有严重伪影（色块、灰色噪声、NaN）。修复后 wm-raw 生成效果与 wm-training 的 `batch_generate_image.py` 视觉一致。

## 关键修复（按影响程度排序）

### 1. MRoPE 交错实现（色块根因）

**问题**：wm-raw 把 MRoPE 实现为简单拼接 `[T_freqs, H_freqs, W_freqs]`，但 HF Qwen3-VL 使用真正的交错分配：T 在索引 0,3,6...；H 在 1,4,7...；W 在 2,5,8...

**同时**：Diffusion 的 position IDs 设为顺序 `arange(1024)`（3 轴相同），实际应该用 2D 网格 `[temporal=0, height=行号, width=列号]`。

**影响**：完全破坏 attention 中的空间结构 → 马赛克/色块。

**修复**：`rope.py` 的 `_apply_interleaved_mrope` + `diffusion_branch.py` 的 position_ids 构建。

### 2. RMSNorm dtype 转换顺序

**问题**：wm-raw 计算 `(weight * x_fp32).to(bf16)`，HF 计算 `weight * x_fp32.to(bf16)`。乘法精度不同。

**影响**：每次 norm 调用 max 误差 0.5，经 36 层 VLM + 28 层 diffusion 累积后完全发散。

**修复**：`qwen3vl_backbone.py` 第 37 行：`return self.weight * x.to(input_dtype)`

### 3. Cross-Attention 的 RMSNorm 类型

**问题**：cross-attention 的 `context_norm`/`query_norm` 用了 Qwen 自定义 RMSNorm（HF 的 bf16 转换顺序），但线上用的是原生 `nn.RMSNorm`。

**影响**：每层投影 K/V 有 0.06 误差。

**修复**：`cross_attention.py` 使用 `nn.RMSNorm`。

### 4. Timestep 条件注入模式

**问题**：wm-raw 总是执行 `input_add`（将 time_conditioner 加到 hidden），但 `adaln_zero` 模式下线上不做 `input_add`——时间信息仅通过逐层 AdaLN 注入。

**影响**：双重 timestep 条件 → 灰色噪声输出（模型输出坍缩到均值）。

**修复**：`diffusion_branch.py` 的 `prepare_inputs` 根据 `timestep_conditioning` 判断。

### 5. Attention Mask 格式

**问题**：VLM causal mask 使用 `tril(ones)`（0/1 乘性）而非 `triu(full(-inf))`（SDPA 加性）。且无 padding 时 `0 * -inf = NaN`。

**影响**：VLM 输出垃圾 hidden states → NaN 或无意义条件。

**修复**：正确的加性 mask + `masked_fill` 处理 padding。

### 6. 分辨率与 Checkpoint 映射

- 图片分辨率：256px → 512px（1024 tokens）
- `timestep_shift`：2.0 → 1.0（匹配训练配置）
- Checkpoint key 映射：增加 `input_projector.linear_1→proj`、`time_embedder.linear_1/2→mlp.0/2`、`position_embedding→pos_embed`

## 新增基础设施

- `scripts/compare_inference.py` — 逐层数值对比工具
- `scripts/align/` — 对齐测试工具集
- `src/wm_raw/utils/timing.py` — 可复用的 Timer/log_duration
- `generate_image.py` — 支持 .pt/.dcp checkpoint、批量 eval（9 个难度级别）、HF VLM 条件编码

## 验证结果

- 逐层 diff：**0.000000**（28 层 diffusion，相同输入）
- MRoPE cos/sin：**0.000000**（vs HF Qwen3VLTextRotaryEmbedding）
- Block ratio：**7.58 → 1.00**（平滑无伪影）
- 生成图片视觉上与 wm-training 输出一致
