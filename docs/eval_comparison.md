# 推理效果对比

对比 wm-raw 与 wm-training 在相同 prompt 下的生成效果。

## 实验配置

| 参数 | wm-raw | wm-training |
|------|--------|-------------|
| Checkpoint | step_295000 / step_345000 | step_295000 / step_345000 |
| 采样器 | Euler, 50 steps | Euler, 50 steps |
| CFG scale | 5.0 | 5.0 |
| 图片尺寸 | 512×512 | 512×512 |
| Seed | 1234-1242 | 1234-1242 |
| Condition | `Caption: {prompt} <\|wm_predict_image\|>` | 同左 |

---

## Level 1 — Simple Natural Landscape

**Prompt**: A peaceful lake surrounded by green mountains, clear blue sky, soft sunlight, realistic photography style.

| wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|:---:|:---:|:---:|:---:|
| ![](eval_images/raw_295k/L1_simple_natural_landscape_seed1234.png) | ![](eval_images/raw_345k/L1_simple_natural_landscape_seed1234.png) | ![](eval_images/online_295k/eval_L1_simple_natural_landscape_seed1234.png) | ![](eval_images/online_345k/eval_L1_simple_natural_landscape_seed1234.png) |

---

## Level 2 — Complex Natural Environment

**Prompt**: A vast snowy mountain valley at sunrise, a crystal clear river flowing through the valley, pine forests covering the slopes, golden sunlight reflecting on the snow, ultra realistic landscape photography.

| wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|:---:|:---:|:---:|:---:|
| ![](eval_images/raw_295k/L2_complex_natural_environment_seed1235.png) | ![](eval_images/raw_345k/L2_complex_natural_environment_seed1235.png) | ![](eval_images/online_295k/eval_L2_complex_natural_environment_seed1235.png) | ![](eval_images/online_345k/eval_L2_complex_natural_environment_seed1235.png) |

---

## Level 3 — Cinematic Landscape

**Prompt**: A cinematic view of an ancient castle on a cliff above the ocean, surrounded by mist and dark clouds, dramatic lighting, waves crashing against the rocks, fantasy movie scene, highly detailed.

| wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|:---:|:---:|:---:|:---:|
| ![](eval_images/raw_295k/L3_cinematic_landscape_seed1236.png) | ![](eval_images/raw_345k/L3_cinematic_landscape_seed1236.png) | ![](eval_images/online_295k/eval_L3_cinematic_landscape_seed1236.png) | ![](eval_images/online_345k/eval_L3_cinematic_landscape_seed1236.png) |

---

## Level 4 — City Architecture & Crowd

**Prompt**: A futuristic Tokyo street at night, neon signs glowing everywhere, rain-soaked streets reflecting colorful lights, pedestrians walking with umbrellas, cyberpunk atmosphere, cinematic photography, ultra detailed.

| wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|:---:|:---:|:---:|:---:|
| ![](eval_images/raw_295k/L4_city_architecture_crowd_seed1237.png) | ![](eval_images/raw_345k/L4_city_architecture_crowd_seed1237.png) | ![](eval_images/online_295k/eval_L4_city_architecture_crowd_seed1237.png) | ![](eval_images/online_345k/eval_L4_city_architecture_crowd_seed1237.png) |

---

## Level 5 — Basic Portrait

**Prompt**: Portrait of a young woman with long brown hair, wearing a white dress, natural expression, soft studio lighting, realistic photography, high detail.

| wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|:---:|:---:|:---:|:---:|
| ![](eval_images/raw_295k/L5_basic_portrait_seed1238.png) | ![](eval_images/raw_345k/L5_basic_portrait_seed1238.png) | ![](eval_images/online_295k/eval_L5_basic_portrait_seed1238.png) | ![](eval_images/online_345k/eval_L5_basic_portrait_seed1238.png) |

---

## Level 6 — Portrait with Environment & Clothing

**Prompt**: A detailed portrait of a young Asian woman wearing a traditional red silk dress, standing in an ancient Chinese garden, delicate embroidery patterns on the fabric, natural makeup, soft afternoon sunlight, shallow depth of field, professional photography.

| wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|:---:|:---:|:---:|:---:|
| ![](eval_images/raw_295k/L6_portrait_environment_clothing_seed1239.png) | ![](eval_images/raw_345k/L6_portrait_environment_clothing_seed1239.png) | ![](eval_images/online_295k/eval_L6_portrait_environment_clothing_seed1239.png) | ![](eval_images/online_345k/eval_L6_portrait_environment_clothing_seed1239.png) |

---

## Level 7 — Complex Character Scene

**Prompt**: A cinematic portrait of an elderly Japanese craftsman working in a traditional wooden workshop, wearing a worn blue apron, detailed wrinkles on his face, wooden tools and handmade objects around him, warm sunlight coming through the window, emotional storytelling photography, ultra realistic.

| wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|:---:|:---:|:---:|:---:|
| ![](eval_images/raw_295k/L7_complex_character_scene_seed1240.png) | ![](eval_images/raw_345k/L7_complex_character_scene_seed1240.png) | ![](eval_images/online_295k/eval_L7_complex_character_scene_seed1240.png) | ![](eval_images/online_345k/eval_L7_complex_character_scene_seed1240.png) |

---

## Level 8 — Multi-Person Group

**Prompt**: A group portrait of five people from different generations standing together in a cozy living room, each person has unique facial features and clothing styles, grandmother, parents and children smiling naturally, warm indoor lighting, realistic photography, highly detailed faces, accurate human anatomy.

| wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|:---:|:---:|:---:|:---:|
| ![](eval_images/raw_295k/L8_multi_person_group_seed1241.png) | ![](eval_images/raw_345k/L8_multi_person_group_seed1241.png) | ![](eval_images/online_295k/eval_L8_multi_person_group_seed1241.png) | ![](eval_images/online_345k/eval_L8_multi_person_group_seed1241.png) |

---

## Level 9 — Extreme Composite Test

**Prompt**: A cinematic photo of a young female astronaut exploring an alien planet, wearing a detailed futuristic spacesuit with realistic fabric textures, holding a holographic device, a massive alien city in the background, strange plants and creatures around her, dramatic sunset lighting, reflections on the helmet glass, Hollywood sci-fi movie style, ultra realistic, 8K detail.

| wm-raw 295k | wm-raw 345k | wm-training 295k | wm-training 345k |
|:---:|:---:|:---:|:---:|
| ![](eval_images/raw_295k/L9_extreme_composite_test_seed1242.png) | ![](eval_images/raw_345k/L9_extreme_composite_test_seed1242.png) | ![](eval_images/online_295k/eval_L9_extreme_composite_test_seed1242.png) | ![](eval_images/online_345k/eval_L9_extreme_composite_test_seed1242.png) |
