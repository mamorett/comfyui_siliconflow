from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models
from .utils import tensor_to_base64

class SiliconFlowFlux11ProUltra(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "flux-1.1-pro-ultra" in m.lower()]
        if not models: models = ["black-forest-labs/FLUX-1.1-pro-Ultra"]
        return {
            "required": {
                "model": (models, {"default": models[0], "tooltip": "FLUX-1.1 Pro Ultra model ID."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Description."}),
                "image_size": (["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280", "others"], {"default": "1024x1024", "tooltip": "Supported resolution presets."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999, "tooltip": "Random seed."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "tooltip": "Negative prompt."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Batch generation size (max 4)."}),
                "aspect_ratio": ("STRING", {"default": "1:1", "tooltip": "Aspect ratio between 21:9 and 9:21."}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6, "tooltip": "Filter strictness (0-6)."}),
                "output_format": (["png", "jpeg"], {"default": "png", "tooltip": "Image format."}),
                "raw": ("BOOLEAN", {"default": False, "tooltip": "Generate less processed, more natural-looking images."}),
                "image_prompt": ("IMAGE", {"tooltip": "Image to remix in base64 format."}),
                "image_prompt_strength": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.1, "tooltip": "Blend strength between text and image prompt."}),
            }
        }

    def generate(self, model, prompt, image_size, seed, negative_prompt="", batch_size=1, aspect_ratio="1:1", safety_tolerance=2, output_format="png", raw=False, image_prompt=None, image_prompt_strength=0.1):
        img_p_b64 = tensor_to_base64(image_prompt) if image_prompt is not None else None
        return self._generate_common(model, prompt, image_size=image_size, seed=seed, negative_prompt=negative_prompt, batch_size=batch_size, aspect_ratio=aspect_ratio, safety_tolerance=safety_tolerance, output_format=output_format, raw=raw, image_prompt=img_p_b64, image_prompt_strength=image_prompt_strength)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux11ProUltra": SiliconFlowFlux11ProUltra}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux11ProUltra": "🎨 SiliconFlow — FLUX-1.1 Pro Ultra"}
