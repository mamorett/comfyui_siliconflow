# ComfyUI — SiliconFlow Custom Nodes

ComfyUI nodes for generating images via **SiliconFlow** APIs.

## Structure

```
comfyui_siliconflow/
├── __init__.py              # Entry point and node registration
├── config.py                # API key management
├── api_client.py            # SiliconFlow HTTP client
├── node_model_selector.py   # Node: model selection
├── node_inference.py        # Node: image generation
├── apikey.txt               # ← API key (DO NOT share!)
└── .gitignore               # Excludes apikey.txt from repository
```

## Installation

1. **Copy the folder** to the ComfyUI custom nodes directory:
   ```
   ComfyUI/custom_nodes/comfyui_siliconflow/
   ```

2. **Insert your API key** into the `apikey.txt` file:
   ```
   sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   > ⚠️ Never share this file. It is automatically excluded by `.gitignore`.

3. **Restart ComfyUI**. The nodes will appear in the **SiliconFlow** category.

## Available Nodes

### 🤖 SiliconFlow — Model Selector

Dynamically retrieves the list of image generation models from SiliconFlow and presents it as a dropdown.

| Input | Type | Description |
|-------|------|-------------|
| `model` | Dropdown | Selected model |
| `refresh_models` | Boolean | Force list refresh |

| Output | Type | Description |
|--------|------|-------------|
| `model` | SFMODEL | Model ID to pass to the inference node |

---

### 🎨 SiliconFlow — Image Generation

Main inference node. Supports text-to-image and image editing.

| Input | Type | Required | Description |
|-------|------|:---:|-------------|
| `model` | SFMODEL | ✅ | Model from the selector node |
| `prompt` | String | ✅ | Prompt text |
| `width` | Int | ✅ | Output width (256–2048) |
| `height` | Int | ✅ | Output height (256–2048) |
| `seed` | Int | ✅ | Seed for generation |
| `random_seed` | Boolean | ✅ | If enabled, uses random seed |
| `num_steps` | Int | ✅ | Diffusion steps (1–100) |
| `guidance_scale` | Float | ✅ | CFG scale (0–20) |
| `negative_prompt` | String | ❌ | Elements to avoid |
| `image_1` | IMAGE | ❌ | Input image 1 (edit/img2img) |
| `image_2` | IMAGE | ❌ | Input image 2 |
| `image_3` | IMAGE | ❌ | Input image 3 |
| `image_4` | IMAGE | ❌ | Input image 4 |

| Output | Type | Description |
|--------|------|-------------|
| `image` | IMAGE | Generated image (ComfyUI tensor) |

## Example Workflow

```
[SiliconFlow Model Selector] → model
                                     ↘
[Load Image (opt.)] → image_1   [SiliconFlow Inference] → image → [Preview Image]
                                     ↗
                  prompt, width, height, seed...
```

## Notes

- The **model list** is cached for 5 minutes to reduce API calls.
- With `random_seed = True`, each execution produces a different result.
- Input images (image_1–4) are **optional**: if omitted, the node works in pure text-to-image mode.
- Compatibility with edit/img2img models depends on the specific model's support on SiliconFlow.

## Requirements

- ComfyUI (any recent version)
- Python 3.8+
- `Pillow` (PIL) — already included in ComfyUI
- No additional external dependencies