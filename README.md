# 🎨 ComfyUI SiliconFlow Specialized Nodes

A production-grade collection of specialized ComfyUI nodes for the [SiliconFlow](https://siliconflow.cn/) Image Generation API.

This extension provides **granular nodes** for every major model variant in the SiliconFlow documentation. Each node is surgically tailored to expose **every single parameter** supported by that specific model, ensuring full control without UI clutter.

---

## 🚀 Key Features

- **Exhaustive Control**: Every mandatory and optional parameter from the SiliconFlow API is exposed.
- **Zero UI Clutter**: Parameters irrelevant to a model are physically hidden via specialized nodes.
- **Live Tooltips**: Hover over any parameter in ComfyUI for real-time documentation.
- **Metadata Recovery**: All nodes output the `actual_seed` used by the API for perfect reproducibility.
- **Native Batching**: Returns standard ComfyUI `IMAGE` batch tensors.

---

## 🧩 Comprehensive Node & Parameter Reference

All nodes return `(IMAGE, INT)` representing the image tensor and the `actual_seed`.

### 1. FLUX.2 Series

#### 🎨 SiliconFlow — FLUX.2 Pro
*High-quality professional generation.*
- **model** (Required): `black-forest-labs/FLUX.2-pro`
- **prompt** (Required): Text description of the image.
- **image_size** (Required): [`512x512`, `768x1024`, `1024x768`, `576x1024`, `1024x576`]
- **seed** (Required): `0 - 9999999999`.
- **output_format** (Required): [`png`, `jpeg`]

#### 🎨 SiliconFlow — FLUX.2 Flex
*Flexible generation with CFG and step control.*
- **model** (Required): `black-forest-labs/FLUX.2-flex`
- **prompt** (Required): Text description.
- **image_size** (Required): [`512x512`, `768x1024`, `1024x768`, `576x1024`, `1024x576`]
- **seed** (Required): `0 - 9999999999`.
- **num_inference_steps** (Optional): `1 - 50` (Default: `25`).
- **cfg** (Optional): `0.1 - 20.0` (Default: `4.0`).
- **output_format** (Optional): [`png`, `jpeg`]

---

### 2. FLUX.1 Series (Advanced)

#### 🎨 SiliconFlow — FLUX-1.1 Pro
*High-performance resolution control.*
- **model** (Required): `black-forest-labs/FLUX-1.1-pro`
- **prompt** (Required): Text description.
- **width** (Required): `256 - 1440` (Must be multiple of 32, Default: `1024`).
- **height** (Required): `256 - 1440` (Must be multiple of 32, Default: `768`).
- **seed** (Required): `0 - 9999999999`.
- **image_prompt** (Optional): `IMAGE` input for visual guidance.
- **prompt_upsampling** (Optional): `True/False` (Automatic creative expansion).
- **safety_tolerance** (Optional): `0 - 6` (0: Strictest, 6: Lenient, Default: `2`).
- **output_format** (Optional): [`png`, `jpeg`]

#### 🎨 SiliconFlow — FLUX-1.1 Pro Ultra
*The flagship generation node with every available feature.*
- **model** (Required): `black-forest-labs/FLUX-1.1-pro-Ultra`
- **prompt** (Required): Text description.
- **image_size** (Required): [`1024x1024`, `960x1280`, `768x1024`, `720x1440`, `720x1280`, `others`]
- **seed** (Required): `0 - 9999999999`.
- **negative_prompt** (Optional): Elements to avoid in the image.
- **batch_size** (Optional): `1 - 4` (Generate multiple images in one go).
- **aspect_ratio** (Optional): String between `21:9` and `9:21` (Default: `1:1`).
- **safety_tolerance** (Optional): `0 - 6` (Default: `2`).
- **output_format** (Optional): [`png`, `jpeg`]
- **raw** (Optional): `True/False` (Generates more natural, photographic textures).
- **image_prompt** (Optional): `IMAGE` to remix.
- **image_prompt_strength** (Optional): `0.0 - 1.0` (Blend between text and image prompt, Default: `0.1`).

#### 🎨 SiliconFlow — FLUX.1 Dev
- **model** (Required): `black-forest-labs/FLUX.1-dev`
- **prompt** (Required): Text description.
- **image_size** (Required): [`1024x1024`, `960x1280`, `768x1024`, `720x1440`, `720x1280`, `others`]
- **num_inference_steps** (Required): `1 - 30` (Default: `20`).
- **seed** (Required): `0 - 9999999999`.
- **prompt_enhancement** (Optional): `True/False` (Optimize for model-friendliness).

#### 🎨 SiliconFlow — FLUX.1 Schnell
- **model** (Required): `black-forest-labs/FLUX.1-schnell`
- **prompt** (Required): Text description.
- **image_size** (Required): [`1024x1024`, `512x1024`, `768x512`, `768x1024`, `1024x576`, `576x1024`]
- **seed** (Required): `0 - 9999999999`.
- **prompt_enhancement** (Optional): `True/False`.

---

### 3. Kontext (Image-to-Image)

#### 🎨 SiliconFlow — FLUX.1 Kontext
*Advanced context-aware generation.*
- **model** (Required): `black-forest-labs/FLUX.1-Kontext-max`, `black-forest-labs/FLUX.1-Kontext-pro`
- **prompt** (Required): Text description.
- **image** (Required): `IMAGE` input (The base image).
- **seed** (Required): `0 - 9999999999`.
- **aspect_ratio** (Optional): String between `21:9` and `9:21` (Default: `1:1`).
- **output_format** (Optional): [`png`, `jpeg`]
- **prompt_upsampling** (Optional): `True/False`.
- **safety_tolerance** (Optional): `0 - 6` (Note: Img2Img is usually capped at `2`).

#### 🎨 SiliconFlow — FLUX.1 Kontext Dev
- **model** (Required): `black-forest-labs/FLUX.1-Kontext-dev`
- **prompt** (Required): Text description.
- **image** (Required): `IMAGE` input.
- **seed** (Required): `0 - 9999999999`.
- **prompt_enhancement** (Optional): `True/False`.

---

### 4. Qwen & Z-Image

#### 🎨 SiliconFlow — Qwen Image
*Complete Qwen implementation supporting both generation and editing.*
- **model** (Required): `Qwen/Qwen-Image`, `Qwen/Qwen-Image-Edit`
- **prompt** (Required): Text description.
- **image_size** (Required): [`1328x1328`, `1664x928`, `928x1664`, `1472x1140`, `1140x1472`, `1584x1056`, `1056x1584`]
- **seed** (Required): `0 - 9999999999`.
- **negative_prompt** (Optional): Elements to exclude.
- **batch_size** (Optional): `1 - 4`.
- **num_inference_steps** (Optional): `1 - 100` (Default: `20`).
- **guidance_scale** (Optional): `0.0 - 20.0` (Stricter vs Creative adherence, Default: `7.5`).
- **cfg** (Optional): `0.1 - 20.0` (Required for text generation, Default: `4.0`).
- **image** (Optional): `IMAGE` input for editing models.

#### 🎨 SiliconFlow — Z-Image
*Ultra-fast turbo generation.*
- **model** (Required): `Tongyi-MAI/Z-Image-Turbo`
- **prompt** (Required): Text description.
- **image_size** (Required): [`512x512`, `768x1024`, `1024x576`, `576x1024`]
- **seed** (Required): `0 - 9999999999`.
- **negative_prompt** (Optional): Elements to avoid.

---

## 🛠️ Setup

1. **API Key**: Create `apikey.txt` in the root folder and paste your key.
2. **Category**: All nodes are under `SiliconFlow` in the ComfyUI menu.

## ⚖️ License
MIT
