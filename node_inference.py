"""
node_inference.py — ComfyUI node for image inference via SiliconFlow.

Input:
  - model          : model string (from SiliconFlowModelSelector node)
  - model_family   : model family string (from SiliconFlowModelSelector node)
  - prompt         : prompt text
  - image          : optional image (ComfyUI tensor) for edit/img2img models
  - image_size     : output size as "widthxheight" string
  - seed           : seed (−1 = random)
  - random_seed    : if True, ignores seed and uses random value on each run

Output:
  - IMAGE          : ComfyUI image tensor (B,H,W,C) float32 [0,1]
"""

import io
import random
import time

import numpy as np

# Import opzionale torch — ComfyUI lo ha sempre disponibile
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import PIL — presente nell'ambiente ComfyUI
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .api_client import run_inference
from .node_model_selector import IMAGE_SIZE_PRESETS, get_model_family


def _tensor_to_base64(tensor) -> str:
    """
    Converts a ComfyUI tensor (B,H,W,C) float32 [0,1] to a base64 PNG string.
    If there are multiple batches, uses the first frame.
    """
    import base64

    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        img_array = tensor[0].cpu().numpy()
    else:
        img_array = np.array(tensor)[0]

    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)

    if HAS_PIL:
        pil_img = Image.fromarray(img_array, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        # Fallback without PIL: raw PPM → base64
        import base64 as b64mod
        h, w, c = img_array.shape
        header = f"P6\n{w} {h}\n255\n".encode()
        raw = header + img_array.tobytes()
        return b64mod.b64encode(raw).decode("utf-8")


def _bytes_to_tensor(image_bytes: bytes):
    """
    Converts PNG/JPEG image bytes to a ComfyUI tensor (1,H,W,C) float32 [0,1].
    """
    if HAS_PIL:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(pil_img, dtype=np.float32) / 255.0
    else:
        raise RuntimeError(
            "[SiliconFlow] PIL not available: unable to decode the image."
        )

    if HAS_TORCH:
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, C)
    else:
        return arr[np.newaxis, ...]  # (1, H, W, C) numpy


class SiliconFlowInference:
    """
    Main inference node for SiliconFlow.
    Supports text-to-image and image editing (single input image).
    Dynamically shows/hides parameters based on model family.
    """

    CATEGORY = "SiliconFlow"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        # Default image sizes (will be dynamically updated based on model)
        default_sizes = ["1024x1024", "512x512", "768x1024", "1024x768"]

        return {
            "required": {
                "model": ("SFMODEL", {}),
                "model_family": ("STRING", {"forceInput": True}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A beautiful landscape, photorealistic, 8k",
                        "placeholder": "Describe the image to generate...",
                    },
                ),
                "image_size": (
                    default_sizes,
                    {"default": default_sizes[0]},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 9999999999},
                ),
                "random_seed": (
                    "BOOLEAN",
                    {"default": False, "label_on": "🎲 Random", "label_off": "Fixed"},
                ),
                "num_steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 100, "step": 1},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.5},
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Elements to avoid in the image...",
                    },
                ),
                "image": ("IMAGE", {}),
                "output_format": (
                    ["png", "jpeg"],
                    {"default": "png"},
                ),
                # FLUX.2-flex and Qwen-Image specific
                "cfg": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1},
                ),
                # FLUX.1-schnell, FLUX.1-dev specific
                "prompt_enhancement": (
                    "BOOLEAN",
                    {"default": False, "label": "✨ Prompt Enhancement"},
                ),
                # FLUX.1-Kontext, FLUX-1.1-pro specific
                "safety_tolerance": (
                    "INT",
                    {"default": 2, "min": 0, "max": 6, "step": 1},
                ),
                # FLUX-1.1-pro specific (uses width/height instead of image_size)
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 1440, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 256, "max": 1440, "step": 32},
                ),
                # FLUX.1-Kontext specific (uses aspect_ratio instead of image_size)
                "aspect_ratio": (
                    "STRING",
                    {"default": "1:1", "placeholder": "e.g., 16:9, 9:16, 1:1"},
                ),
                # FLUX-1.1-pro-Ultra specific
                "image_prompt": ("IMAGE", {}),
                "image_prompt_strength": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.1},
                ),
                "raw": (
                    "BOOLEAN",
                    {"default": False, "label": "🎨 Raw Mode"},
                ),
            },
        }

    # Force re-execution when random_seed=True
    @classmethod
    def IS_CHANGED(cls, random_seed: bool = False, **kwargs):
        if random_seed:
            return time.time()  # valore sempre diverso → riesegue sempre
        return False

    def generate(
        self,
        model: str,
        model_family: str,
        prompt: str,
        image_size: str,
        seed: int,
        random_seed: bool,
        num_steps: int,
        guidance_scale: float,
        negative_prompt: str = "",
        image=None,
        output_format: str = "png",
        cfg: float = None,
        prompt_enhancement: bool = None,
        safety_tolerance: int = None,
        width: int = None,
        height: int = None,
        aspect_ratio: str = None,
        image_prompt=None,
        image_prompt_strength: float = None,
        raw: bool = None,
    ) -> tuple:

        # Resolve seed
        effective_seed = random.randint(0, 2**32 - 1) if random_seed else seed
        print(f"[SiliconFlow] Generating with seed: {effective_seed}")

        # Convert input image to base64 if provided
        input_image_b64 = None
        if image is not None:
            try:
                input_image_b64 = _tensor_to_base64(image)
                print(f"[SiliconFlow] Input image converted to base64.")
            except Exception as e:
                print(f"[SiliconFlow] Input image conversion error: {e}")

        # Convert image_prompt to base64 if provided
        image_prompt_b64 = None
        if image_prompt is not None:
            try:
                image_prompt_b64 = _tensor_to_base64(image_prompt)
                print(f"[SiliconFlow] Image prompt converted to base64.")
            except Exception as e:
                print(f"[SiliconFlow] Image prompt conversion error: {e}")

        # Get image size presets for this model family
        size_presets = IMAGE_SIZE_PRESETS.get(model_family, [])
        use_image_size = image_size if size_presets else None

        # For FLUX-1.1-pro, use width/height instead of image_size
        if model_family == "FLUX-1.1-pro":
            use_image_size = None

        # For FLUX.1-Kontext, use aspect_ratio instead of image_size
        if "FLUX.1-Kontext" in model_family:
            use_image_size = None

        print(
            f"[SiliconFlow] Starting inference — model: {model} | "
            f"family: {model_family} | size: {image_size}"
        )

        try:
            image_bytes = run_inference(
                model=model,
                prompt=prompt,
                image_size=use_image_size,
                seed=effective_seed,
                input_image=input_image_b64,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                cfg=cfg if model_family in ("FLUX.2-flex", "Qwen-Image") else None,
                output_format=output_format,
                prompt_enhancement=prompt_enhancement if model_family in ("FLUX.1-schnell", "FLUX.1-dev") else None,
                prompt_upsampling=prompt_enhancement if "FLUX.1-Kontext" in model_family else None,
                safety_tolerance=safety_tolerance if model_family in ("FLUX.1-Kontext", "FLUX-1.1-pro", "FLUX-1.1-pro-Ultra") else None,
                image_prompt=image_prompt_b64 if model_family == "FLUX-1.1-pro-Ultra" else None,
                image_prompt_strength=image_prompt_strength if model_family == "FLUX-1.1-pro-Ultra" else None,
                raw=raw if model_family == "FLUX-1.1-pro-Ultra" else None,
                width=width if model_family == "FLUX-1.1-pro" else None,
                height=height if model_family == "FLUX-1.1-pro" else None,
                aspect_ratio=aspect_ratio if "FLUX.1-Kontext" in model_family else None,
            )
        except Exception as e:
            raise RuntimeError(f"[SiliconFlow] Inference error: {e}") from e

        # Convert bytes → ComfyUI tensor
        output_tensor = _bytes_to_tensor(image_bytes)
        print(f"[SiliconFlow] ✅ Image generated successfully! Shape: {output_tensor.shape}")

        return (output_tensor,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SiliconFlowInference": SiliconFlowInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SiliconFlowInference": "🎨 SiliconFlow — Image Generation",
}
