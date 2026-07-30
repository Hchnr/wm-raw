# 推理效果对比表

对比 wm-raw 与 wm-training 在相同 prompt 下的生成效果。

## 实验配置

| 参数 | wm-raw | wm-training |
|------|--------|-------------|
| 模型 | wm-raw (本仓库) | wm-training (线上) |
| Checkpoint | step_295000 / step_345000 | step_295000 / step_345000 |
| 采样器 | Euler, 50 steps | Euler, 50 steps |
| CFG scale | 5.0 | 5.0 |
| 图片尺寸 | 512×512 | 512×512 |
| Seed | 42-50 (per prompt) | 1234-1242 (per prompt) |
| Condition 格式 | `Caption: {prompt} <\|wm_predict_image\|>` | 同左 |
| VLM condition | HF Qwen3-VL-4B-Instruct | HF Qwen3-VL-4B-Instruct |

> **注**：两侧 seed 不同（wm-raw 42+, wm-training 1234+），因此生成内容不会 pixel-exact 一致，但风格/质量应当可比。

## 结果文件路径

| 来源 | 路径 |
|------|------|
| wm-raw 295k | `scripts/align/eval_results_295k/` |
| wm-raw 345k | `scripts/align/eval_results_345k/` |
| wm-training 295k | `/share/project/eai_pwm/home/hcr/repos/wm-training/outputs/eval_batch_295k/` |
| wm-training 345k | `/share/project/eai_pwm/home/hcr/repos/wm-training/outputs/eval_batch_345k/` |

---

## Level 1 — Simple Natural Landscape

**Prompt**: A peaceful lake surrounded by green mountains, clear blue sky, soft sunlight, realistic photography style.

| | wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|---|---|---|---|---|
| File | `L1_simple_natural_landscape_seed42.png` | `L1_simple_natural_landscape_seed42.png` | `eval_L1_simple_natural_landscape_seed1234.png` | `eval_L1_simple_natural_landscape_seed1234.png` |

---

## Level 2 — Complex Natural Environment

**Prompt**: A vast snowy mountain valley at sunrise, a crystal clear river flowing through the valley, pine forests covering the slopes, golden sunlight reflecting on the snow, ultra realistic landscape photography.

| | wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|---|---|---|---|---|
| File | `L2_complex_natural_environment_seed43.png` | `L2_complex_natural_environment_seed43.png` | `eval_L2_complex_natural_environment_seed1235.png` | `eval_L2_complex_natural_environment_seed1235.png` |

---

## Level 3 — Cinematic Landscape

**Prompt**: A cinematic view of an ancient castle on a cliff above the ocean, surrounded by mist and dark clouds, dramatic lighting, waves crashing against the rocks, fantasy movie scene, highly detailed.

| | wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|---|---|---|---|---|
| File | `L3_cinematic_landscape_seed44.png` | `L3_cinematic_landscape_seed44.png` | `eval_L3_cinematic_landscape_seed1236.png` | `eval_L3_cinematic_landscape_seed1236.png` |

---

## Level 4 — City Architecture & Crowd

**Prompt**: A futuristic Tokyo street at night, neon signs glowing everywhere, rain-soaked streets reflecting colorful lights, pedestrians walking with umbrellas, cyberpunk atmosphere, cinematic photography, ultra detailed.

| | wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|---|---|---|---|---|
| File | `L4_city_architecture_crowd_seed45.png` | `L4_city_architecture_crowd_seed45.png` | `eval_L4_city_architecture_crowd_seed1237.png` | `eval_L4_city_architecture_crowd_seed1237.png` |

---

## Level 5 — Basic Portrait

**Prompt**: Portrait of a young woman with long brown hair, wearing a white dress, natural expression, soft studio lighting, realistic photography, high detail.

| | wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|---|---|---|---|---|
| File | `L5_basic_portrait_seed46.png` | `L5_basic_portrait_seed46.png` | `eval_L5_basic_portrait_seed1238.png` | `eval_L5_basic_portrait_seed1238.png` |

---

## Level 6 — Portrait with Environment & Clothing

**Prompt**: A detailed portrait of a young Asian woman wearing a traditional red silk dress, standing in an ancient Chinese garden, delicate embroidery patterns on the fabric, natural makeup, soft afternoon sunlight, shallow depth of field, professional photography.

| | wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|---|---|---|---|---|
| File | `L6_portrait_environment_clothing_seed47.png` | `L6_portrait_environment_clothing_seed47.png` | `eval_L6_portrait_environment_clothing_seed1239.png` | `eval_L6_portrait_environment_clothing_seed1239.png` |

---

## Level 7 — Complex Character Scene

**Prompt**: A cinematic portrait of an elderly Japanese craftsman working in a traditional wooden workshop, wearing a worn blue apron, detailed wrinkles on his face, wooden tools and handmade objects around him, warm sunlight coming through the window, emotional storytelling photography, ultra realistic.

| | wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|---|---|---|---|---|
| File | `L7_complex_character_scene_seed48.png` | `L7_complex_character_scene_seed48.png` | `eval_L7_complex_character_scene_seed1240.png` | `eval_L7_complex_character_scene_seed1240.png` |

---

## Level 8 — Multi-Person Group

**Prompt**: A group portrait of five people from different generations standing together in a cozy living room, each person has unique facial features and clothing styles, grandmother, parents and children smiling naturally, warm indoor lighting, realistic photography, highly detailed faces, accurate human anatomy.

| | wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|---|---|---|---|---|
| File | `L8_multi_person_group_seed49.png` | `L8_multi_person_group_seed49.png` | `eval_L8_multi_person_group_seed1241.png` | `eval_L8_multi_person_group_seed1241.png` |

---

## Level 9 — Extreme Composite Test

**Prompt**: A cinematic photo of a young female astronaut exploring an alien planet, wearing a detailed futuristic spacesuit with realistic fabric textures, holding a holographic device, a massive alien city in the background, strange plants and creatures around her, dramatic sunset lighting, reflections on the helmet glass, Hollywood sci-fi movie style, ultra realistic, 8K detail.

| | wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|---|---|---|---|---|
| File | `L9_extreme_composite_test_seed50.png` | `L9_extreme_composite_test_seed50.png` | `eval_L9_extreme_composite_test_seed1242.png` | `eval_L9_extreme_composite_test_seed1242.png` |

---

## 观察总结

| 维度 | 结论 |
|------|------|
| 结构完整性 | wm-raw 不再有色块/马赛克伪影，空间结构正确 |
| 风格一致性 | 两侧在同 checkpoint 下风格/质量可比（seed 不同导致内容不同） |
| 345k vs 295k | 345k 步数更多，细节更丰富，符合预期 |
| Landscape vs Portrait | landscape 效果好于 portrait（与模型训练阶段一致） |
