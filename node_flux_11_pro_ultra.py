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
                "aspect_ratio": (["1:1", "2:3", "3:4", "5:8", "9:16", "9:19", "9:21", "3:2", "4:3", "8:5", "16:9", "19:9", "21:9"], {"default": "1:1", "tooltip": "Aspect ratio between 21:9 and 9:21."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 9999999999, "tooltip": "Random seed. Use -1 for random."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "tooltip": "Negative prompt."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4, "tooltip": "Batch generation size (max 4)."}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6, "tooltip": "Filter strictness (0-6)."}),
                "output_format": (["png", "jpeg"], {"default": "png", "tooltip": "Image format."}),
                "raw": ("BOOLEAN", {"default": False, "tooltip": "Generate less processed, more natural-looking images."}),
                "image_prompt": ("IMAGE", {"tooltip": "Image to remix in base64 format."}),
                "image_prompt_strength": ("FLOAT", {"default": 0.1, "min": 0, "max": 1, "step": 0.1, "tooltip": "Blend strength between text and image prompt."}),
            }
        }

    def generate(self, model, prompt, aspect_ratio, seed, negative_prompt="", batch_size=1, safety_tolerance=2, output_format="png", raw=False, image_prompt=None, image_prompt_strength=0.1):
        img_p_b64 = tensor_to_base64(image_prompt) if image_prompt is not None else None
        kwargs = {"aspect_ratio": aspect_ratio}
        if img_p_b64 is not None:
            kwargs["image_prompt"] = img_p_b64
            kwargs["image_prompt_strength"] = image_prompt_strength
        return self._generate_common(model, prompt, seed=seed, negative_prompt=negative_prompt, batch_size=batch_size, safety_tolerance=safety_tolerance, output_format=output_format, raw=raw, **kwargs)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux11ProUltra": SiliconFlowFlux11ProUltra}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux11ProUltra": "🎨 SiliconFlow — FLUX-1.1 Pro Ultra"}
