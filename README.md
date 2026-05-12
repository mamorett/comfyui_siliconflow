# 🎨 ComfyUI SiliconFlow Specialized Nodes

A production-grade collection of specialized ComfyUI nodes for the [SiliconFlow](https://siliconflow.cn/) Image Generation API. 

Unlike generic implementations, this extension provides **granular nodes** for every major model variant in the SiliconFlow documentation. Each node is surgically tailored to expose **only** the parameters supported by that specific model, eliminating UI clutter and "total chaos."

---

## 🚀 Key Features

- **Strict Documentation Adherence**: Every parameter, enum, and default is pulled directly from the SiliconFlow OpenAPI specification.
- **Zero UI Clutter**: Irrelevant parameters are physically hidden by using separate nodes for different model families.
- **Batch Processing**: Native ComfyUI batch support (returns B,H,W,C tensors) for models supporting multiple outputs.
- **Secure Key Management**: API keys are stored locally in `apikey.txt` and never embedded in your workflow files.
- **No Dependencies**: Pure Python implementation using standard ComfyUI primitives.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/your-repo/comfyui_siliconflow.git
   ```
2. **Configure API Key**:
   - Create a file named `apikey.txt` inside the `comfyui_siliconflow` directory.
   - Paste your SiliconFlow API key (starting with `sk-`) into this file.
3. **Restart ComfyUI**: The nodes will appear under the `SiliconFlow` category.

---

## 🧩 Node Reference

All nodes are found under the `SiliconFlow` category in the node search menu.

### 1. FLUX.2 Series

#### 🎨 SiliconFlow — FLUX.2 Pro
*High-quality professional generation.*
- **model**: `black-forest-labs/FLUX.2-pro`
- **image_size**: `512x512`, `768x1024`, `1024x768`, `576x1024`, `1024x576`
- **seed**: 0 - 9999999999
- **output_format**: `png`, `jpeg`

#### 🎨 SiliconFlow — FLUX.2 Flex
*Flexible generation with CFG and step control.*
- **model**: `black-forest-labs/FLUX.2-flex`
- **image_size**: Same as Pro.
- **seed**: 0 - 9999999999
- **num_inference_steps**: 1 - 50 (Default: 25)
- **cfg**: 0.1 - 20.0 (Default: 4.0)
- **output_format**: `png`, `jpeg`

---

### 2. FLUX.1 Series (Advanced)

#### 🎨 SiliconFlow — FLUX-1.1 Pro
*The high-performance 1.1 variant.*
- **model**: `black-forest-labs/FLUX-1.1-pro`
- **width/height**: 256 - 1440 (Multiples of 32)
- **seed**: 0 - 9999999999
- **image_prompt**: (Optional) Input `IMAGE` to use as a visual prompt.
- **prompt_upsampling**: `True/False`
- **safety_tolerance**: 0 - 6 (Default: 2)
- **output_format**: `png`, `jpeg`

#### 🎨 SiliconFlow — FLUX-1.1 Pro Ultra
*The ultimate generation node with batch and raw support.*
- **model**: `black-forest-labs/FLUX-1.1-pro-Ultra`
- **image_size**: `1024x1024`, `960x1280`, `768x1024`, `720x1440`, `720x1280`, `others`
- **aspect_ratio**: `21:9` to `9:21` (Default: `1:1`)
- **batch_size**: 1 - 4
- **raw**: `True/False` (Less processed, more natural look)
- **image_prompt**: Input `IMAGE` to remix.
- **image_prompt_strength**: 0.0 - 1.0 (Blend between text and image prompt)
- **safety_tolerance**: 0 - 6
- **output_format**: `png`, `jpeg`

#### 🎨 SiliconFlow — FLUX.1 Dev / Schnell
*Efficient base models.*
- **model**: `black-forest-labs/FLUX.1-dev`, `black-forest-labs/FLUX.1-schnell`
- **image_size**: Spec-optimized presets (up to 2.3M pixels for Dev).
- **num_inference_steps**: 1 - 30 (Dev only)
- **prompt_enhancement**: `True/False`

---

### 3. Context & Image-to-Image

#### 🎨 SiliconFlow — FLUX.1 Kontext
*Advanced image-to-image with aspect ratio control.*
- **model**: `black-forest-labs/FLUX.1-Kontext-max`, `black-forest-labs/FLUX.1-Kontext-pro`
- **image**: Required `IMAGE` input.
- **aspect_ratio**: e.g., `16:9`, `1:1`.
- **prompt_upsampling**: `True/False`
- **safety_tolerance**: 0 - 6

#### 🎨 SiliconFlow — FLUX.1 Kontext Dev
*Context editing with prompt enhancement.*
- **model**: `black-forest-labs/FLUX.1-Kontext-dev`
- **image**: Required `IMAGE` input.
- **prompt_enhancement**: `True/False`

---

### 4. Qwen & Z-Image

#### 🎨 SiliconFlow — Qwen Image
*Supports both standard generation and image editing.*
- **model**: `Qwen/Qwen-Image`, `Qwen/Qwen-Image-Edit`
- **image_size**: Specialized Qwen resolutions (e.g., `1328x1328`, `1664x928`, etc.)
- **batch_size**: 1 - 4
- **guidance_scale**: 0.0 - 20.0 (Match degree between prompt and image)
- **cfg**: 0.1 - 20.0 (Official recommendation: 4.0)
- **image**: (Optional) For `Qwen-Image-Edit` models.
- **num_inference_steps**: 1 - 100

#### 🎨 SiliconFlow — Z-Image
*Turbo-charged generation.*
- **model**: `Tongyi-MAI/Z-Image-Turbo`
- **image_size**: `512x512`, `768x1024`, `1024x576`, `576x1024`
- **negative_prompt**: Elements to avoid.

---

## 💡 Usage Tips

- **Model Refresh**: The node dropdowns automatically populate based on your API account permissions.
- **Tensors**: All outputs are standard `(B, H, W, C)` tensors. If you generate a batch of 4, you can use standard ComfyUI `Batch Image` nodes to split them.
- **Error Handling**: If a model request fails, check the ComfyUI console for detailed API error responses.

## ⚖️ License
MIT
