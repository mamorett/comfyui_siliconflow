"""
node_inference.py — ComfyUI node for image inference via SiliconFlow.

Input:
  - model       : model string (from SiliconFlowModelSelector node)
  - prompt      : prompt text
  - image_1..4  : optional images (ComfyUI tensors) for edit/img2img models
  - width       : output width
  - height      : output height
  - seed        : seed (−1 = random)
  - random_seed : if True, ignores seed and uses random value on each run

Output:
  - IMAGE        : ComfyUI image tensor (B,H,W,C) float32 [0,1]
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
    Supports text-to-image and image editing (up to 4 input images).
    """

    CATEGORY = "SiliconFlow"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("SFMODEL", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                    "default": "A beautiful landscape, photorealistic, 8k",
                         "placeholder": "Describe the image to generate...",
                    },
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**32 - 1},
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
                "image_1": ("IMAGE", {}),
                "image_2": ("IMAGE", {}),
                "image_3": ("IMAGE", {}),
                "image_4": ("IMAGE", {}),
            },
        }

    # Forza riesecuzione quando random_seed=True
    @classmethod
    def IS_CHANGED(cls, random_seed: bool = False, **kwargs):
        if random_seed:
            return time.time()  # valore sempre diverso → riesegue sempre
        return False

    def generate(
        self,
        model: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        random_seed: bool,
        num_steps: int,
        guidance_scale: float,
        negative_prompt: str = "",
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
    ) -> tuple:

        # Resolve seed
        effective_seed = random.randint(0, 2**32 - 1) if random_seed else seed
        print(f"[SiliconFlow] Generating with seed: {effective_seed}")

        # Collect optional input images
        input_images_b64 = []
        for idx, img_tensor in enumerate([image_1, image_2, image_3, image_4], start=1):
            if img_tensor is not None:
                try:
                    b64 = _tensor_to_base64(img_tensor)
                    input_images_b64.append(b64)
                    print(f"[SiliconFlow] Image {idx} converted to base64.")
                except Exception as e:
                    print(f"[SiliconFlow] Image {idx} conversion error: {e}")

        print(
            f"[SiliconFlow] Starting inference — model: {model} | "
            f"size: {width}x{height} | input images: {len(input_images_b64)}"
        )

        try:
            image_bytes = run_inference(
                model=model,
                prompt=prompt,
                width=width,
                height=height,
                seed=effective_seed,
                input_images=input_images_b64 if input_images_b64 else None,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
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
