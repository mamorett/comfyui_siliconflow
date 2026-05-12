"""
node_inference.py — Unified ComfyUI node for SiliconFlow image generation.

Combines model selection and inference in a single node.
Dynamically shows/hides parameters based on the selected model family.
"""

import io
import random
import time

import numpy as np

# Optional torch import — ComfyUI always has it available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# PIL import — available in ComfyUI environment
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .api_client import run_inference, fetch_image_models


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


# Model family detection
def get_model_family(model_id: str) -> str:
    """Detects the model family from the model ID string."""
    model_lower = model_id.lower()

    if "flux.2-pro" in model_lower:
        return "FLUX.2-pro"
    elif "flux.2-flex" in model_lower:
        return "FLUX.2-flex"
    elif "flux-1.1-pro-ultra" in model_lower:
        return "FLUX-1.1-pro-Ultra"
    elif "flux-1.1-pro" in model_lower:
        return "FLUX-1.1-pro"
    elif "flux.1-kontext-dev" in model_lower:
        return "FLUX.1-Kontext-dev"
    elif "flux.1-kontext" in model_lower:
        return "FLUX.1-Kontext"
    elif "flux.1-schnell" in model_lower:
        return "FLUX.1-schnell"
    elif "flux.1-dev" in model_lower:
        return "FLUX.1-dev"
    elif "qwen-image" in model_lower:
        return "Qwen-Image"
    elif "z-image" in model_lower:
        return "Z-Image"
    else:
        return "Unknown"


# Image size presets by model family (from API docs)
IMAGE_SIZE_PRESETS = {
    "FLUX.2-pro": ["512x512", "768x1024", "1024x768", "576x1024", "1024x576"],
    "FLUX.2-flex": ["512x512", "768x1024", "1024x768", "576x1024", "1024x576"],
    "FLUX.1-schnell": ["1024x1024", "512x1024", "768x512", "768x1024", "1024x576", "576x1024"],
    "FLUX.1-dev": ["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280", "others"],
    "FLUX-1.1-pro": None,  # Uses width/height
    "FLUX-1.1-pro-Ultra": ["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280", "others"],
    "FLUX.1-Kontext": None,  # Uses aspect_ratio
    "FLUX.1-Kontext-dev": None,  # Uses image input only
    "Qwen-Image": ["1328x1328", "1664x928", "928x1664", "1472x1140", "1140x1472", "1584x1056", "1056x1584"],
    "Z-Image": ["512x512", "768x1024", "1024x576", "576x1024"],
    "Unknown": ["1024x1024", "512x512", "768x1024", "1024x768"],
}

# Model parameters mapping (from API docs)
# Indicates which parameters each model family supports
MODEL_PARAMS = {
    "FLUX.2-pro": {
        "image_size": True,
        "seed": True,
        "output_format": True,
        "num_inference_steps": False,
        "guidance_scale": False,
        "cfg": False,
        "prompt_enhancement": False,
        "prompt_upsampling": False,
        "safety_tolerance": False,
        "negative_prompt": False,
        "image": False,
        "aspect_ratio": False,
        "width_height": False,
        "image_prompt": False,
        "raw": False,
    },
    "FLUX.2-flex": {
        "image_size": True,
        "seed": True,
        "cfg": True,
        "num_inference_steps": True,
        "output_format": True,
        "guidance_scale": False,
        "prompt_enhancement": False,
        "prompt_upsampling": False,
        "safety_tolerance": False,
        "negative_prompt": False,
        "image": False,
        "aspect_ratio": False,
        "width_height": False,
        "image_prompt": False,
        "raw": False,
    },
    "Qwen-Image": {
        "image_size": True,
        "seed": True,
        "guidance_scale": True,
        "cfg": True,
        "num_inference_steps": True,
        "negative_prompt": True,
        "image": True,
        "output_format": False,
        "prompt_enhancement": False,
        "prompt_upsampling": False,
        "safety_tolerance": False,
        "aspect_ratio": False,
        "width_height": False,
        "image_prompt": False,
        "raw": False,
    },
    "Z-Image": {
        "image_size": True,
        "seed": True,
        "negative_prompt": True,
        "output_format": False,
        "num_inference_steps": False,
        "guidance_scale": False,
        "cfg": False,
        "prompt_enhancement": False,
        "prompt_upsampling": False,
        "safety_tolerance": False,
        "image": False,
        "aspect_ratio": False,
        "width_height": False,
        "image_prompt": False,
        "raw": False,
    },
    "FLUX.1-Kontext": {
        "image_size": False,
        "seed": True,
        "aspect_ratio": True,
        "output_format": True,
        "prompt_upsampling": True,
        "safety_tolerance": True,
        "input_image": True,
        "num_inference_steps": False,
        "guidance_scale": False,
        "cfg": False,
        "prompt_enhancement": False,
        "negative_prompt": False,
        "width_height": False,
        "image_prompt": False,
        "raw": False,
    },
    "FLUX.1-Kontext-dev": {
        "image_size": False,
        "seed": True,
        "prompt_enhancement": True,
        "image": True,
        "num_inference_steps": False,
        "guidance_scale": False,
        "cfg": False,
        "output_format": False,
        "prompt_upsampling": False,
        "safety_tolerance": False,
        "negative_prompt": False,
        "aspect_ratio": False,
        "width_height": False,
        "image_prompt": False,
        "raw": False,
    },
    "FLUX-1.1-pro": {
        "width_height": True,
        "seed": True,
        "prompt_upsampling": True,
        "safety_tolerance": True,
        "output_format": True,
        "image_prompt": True,
        "image_size": False,
        "num_inference_steps": False,
        "guidance_scale": False,
        "cfg": False,
        "prompt_enhancement": False,
        "negative_prompt": False,
        "aspect_ratio": False,
        "image": False,
        "raw": False,
    },
    "FLUX-1.1-pro-Ultra": {
        "image_size": True,
        "seed": True,
        "negative_prompt": True,
        "aspect_ratio": True,
        "safety_tolerance": True,
        "output_format": True,
        "image_prompt": True,
        "image_prompt_strength": True,
        "raw": True,
        "num_inference_steps": False,
        "guidance_scale": False,
        "cfg": False,
        "prompt_enhancement": False,
        "prompt_upsampling": False,
        "width_height": False,
        "image": False,
    },
    "FLUX.1-schnell": {
        "image_size": True,
        "seed": True,
        "prompt_enhancement": True,
        "num_inference_steps": False,
        "guidance_scale": False,
        "cfg": False,
        "output_format": False,
        "negative_prompt": False,
        "prompt_upsampling": False,
        "safety_tolerance": False,
        "image": False,
        "aspect_ratio": False,
        "width_height": False,
        "image_prompt": False,
        "raw": False,
    },
    "FLUX.1-dev": {
        "image_size": True,
        "seed": True,
        "num_inference_steps": True,
        "prompt_enhancement": True,
        "guidance_scale": False,
        "cfg": False,
        "output_format": False,
        "negative_prompt": False,
        "prompt_upsampling": False,
        "safety_tolerance": False,
        "image": False,
        "aspect_ratio": False,
        "width_height": False,
        "image_prompt": False,
        "raw": False,
    },
}


class SiliconFlowImageGeneration:
    """
    Unified node for SiliconFlow image generation.
    Combines model selection and inference in a single node.
    """

    CATEGORY = "SiliconFlow"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        # Fetch model list
        try:
            models = fetch_image_models()
        except Exception as e:
            print(f"[SiliconFlow] Unable to retrieve models: {e}")
            models = ["— Error: check apikey.txt —"]

        # Default image sizes
        default_sizes = ["1024x1024", "512x512", "768x1024", "1024x768"]

        return {
            "required": {
                "model": (models, {"default": models[0] if models else ""}),
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
            },
            "optional": {
                "refresh_button": (
                    "BOOLEAN",
                    {"default": False, "label": "🔄 Refresh Models"},
                ),
                "num_steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 100, "step": 1},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.5},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.1, "max": 20.0, "step": 0.1},
                ),
                "output_format": (
                    ["png", "jpeg"],
                    {"default": "png"},
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Elements to avoid in the image...",
                    },
                ),
                "image": ("IMAGE", {}),
                "prompt_enhancement": (
                    "BOOLEAN",
                    {"default": False, "label": "✨ Prompt Enhancement"},
                ),
                "prompt_upsampling": (
                    "BOOLEAN",
                    {"default": False, "label": "📈 Prompt Upsampling"},
                ),
                "safety_tolerance": (
                    "INT",
                    {"default": 2, "min": 0, "max": 6, "step": 1},
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 1440, "step": 32},
                ),
                "height": (
                    "INT",
                    {"default": 768, "min": 256, "max": 1440, "step": 32},
                ),
                "aspect_ratio": (
                    "STRING",
                    {"default": "1:1", "placeholder": "e.g., 16:9, 9:16, 1:1"},
                ),
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

    @classmethod
    def IS_CHANGED(cls, refresh_button: bool = False, **kwargs):
        if refresh_button:
            return float("random")  # Triggers re-execution
        return False

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        """Validate inputs and return True if valid."""
        model = kwargs.get("model", "")
        if not model or model.startswith("—"):
            return "Please select a valid model"
        return True

    def generate(
        self,
        model: str,
        prompt: str,
        image_size: str,
        seed: int,
        random_seed: bool,
        refresh_button: bool = False,
        num_steps: int = 20,
        guidance_scale: float = 7.5,
        cfg: float = None,
        output_format: str = "png",
        negative_prompt: str = "",
        image=None,
        prompt_enhancement: bool = None,
        prompt_upsampling: bool = None,
        safety_tolerance: int = None,
        width: int = None,
        height: int = None,
        aspect_ratio: str = None,
        image_prompt=None,
        image_prompt_strength: float = None,
        raw: bool = None,
    ) -> tuple:

        # Handle refresh button
        if refresh_button:
            try:
                models = fetch_image_models(force_refresh=True)
                print(f"[SiliconFlow] Model list updated: {len(models)} models found.")
            except Exception as e:
                print(f"[SiliconFlow] Model refresh error: {e}")

        # Detect model family
        model_family = get_model_family(model)
        params = MODEL_PARAMS.get(model_family, {})

        # Resolve seed
        effective_seed = random.randint(0, 2**32 - 1) if random_seed else seed
        print(f"[SiliconFlow] Generating with seed: {effective_seed}")

        # Convert input image to base64 if provided
        input_image_b64 = None
        if image is not None and (params.get("image", False) or params.get("input_image", False)):
            try:
                input_image_b64 = _tensor_to_base64(image)
                print(f"[SiliconFlow] Input image converted to base64.")
            except Exception as e:
                print(f"[SiliconFlow] Input image conversion error: {e}")

        # Convert image_prompt to base64 if provided
        image_prompt_b64 = None
        if image_prompt is not None and params.get("image_prompt", False):
            try:
                image_prompt_b64 = _tensor_to_base64(image_prompt)
                print(f"[SiliconFlow] Image prompt converted to base64.")
            except Exception as e:
                print(f"[SiliconFlow] Image prompt conversion error: {e}")

        # Determine image size parameter
        use_image_size = None
        if params.get("image_size", False):
            use_image_size = image_size
        elif params.get("width_height", False):
            # FLUX-1.1-pro uses width/height
            pass  # Handled separately

        # For FLUX.1-Kontext, use aspect_ratio
        if not params.get("aspect_ratio", False):
            aspect_ratio = None

        print(
            f"[SiliconFlow] Starting inference — model: {model} | "
            f"family: {model_family}"
        )

        # Build API call parameters - only include what the model supports
        try:
            image_bytes = run_inference(
                model=model,
                prompt=prompt,
                image_size=use_image_size if params.get("image_size", False) else None,
                seed=effective_seed,
                input_image=input_image_b64 if params.get("input_image", False) else None,
                negative_prompt=negative_prompt if params.get("negative_prompt", False) else "",
                num_inference_steps=num_steps if params.get("num_inference_steps", False) else None,
                guidance_scale=guidance_scale if params.get("guidance_scale", False) else None,
                cfg=cfg if params.get("cfg", False) else None,
                output_format=output_format if params.get("output_format", False) else None,
                prompt_enhancement=prompt_enhancement if params.get("prompt_enhancement", False) else None,
                prompt_upsampling=prompt_upsampling if params.get("prompt_upsampling", False) else None,
                safety_tolerance=safety_tolerance if params.get("safety_tolerance", False) else None,
                width=width if params.get("width_height", False) else None,
                height=height if params.get("width_height", False) else None,
                aspect_ratio=aspect_ratio if params.get("aspect_ratio", False) else None,
                image_prompt=image_prompt_b64 if params.get("image_prompt", False) else None,
                image_prompt_strength=image_prompt_strength if params.get("image_prompt_strength", False) else None,
                raw=raw if params.get("raw", False) else None,
            )
        except Exception as e:
            raise RuntimeError(f"[SiliconFlow] Inference error: {e}") from e

        # Convert bytes → ComfyUI tensor
        output_tensor = _bytes_to_tensor(image_bytes)
        print(f"[SiliconFlow] ✅ Image generated successfully! Shape: {output_tensor.shape}")

        return (output_tensor,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SiliconFlowImageGeneration": SiliconFlowImageGeneration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SiliconFlowImageGeneration": "🎨 SiliconFlow — Image Generation",
}