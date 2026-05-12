from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models
from .utils import tensor_to_base64

class SiliconFlowFlux11Pro(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "flux-1.1-pro" in m.lower() and "ultra" not in m.lower()]
        if not models: models = ["black-forest-labs/FLUX-1.1-pro"]
        return {
            "required": {
                "model": (models, {"default": models[0], "tooltip": "FLUX-1.1 Pro model ID."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Description."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1440, "step": 32, "tooltip": "Image width (multiple of 32)."}),
                "height": ("INT", {"default": 768, "min": 256, "max": 1440, "step": 32, "tooltip": "Image height (multiple of 32)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999, "tooltip": "Random seed."}),
            },
            "optional": {
                "image_prompt": ("IMAGE", {"tooltip": "Visual prompt for image-guided generation."}),
                "prompt_upsampling": ("BOOLEAN", {"default": False, "tooltip": "Creative prompt expansion."}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6, "tooltip": "Filter strictness (0-6)."}),
                "output_format": (["png", "jpeg"], {"default": "png", "tooltip": "Image format."}),
            }
        }

    def generate(self, model, prompt, width, height, seed, image_prompt=None, prompt_upsampling=False, safety_tolerance=2, output_format="png"):
        img_p_b64 = tensor_to_base64(image_prompt) if image_prompt is not None else None
        return self._generate_common(model, prompt, width=width, height=height, seed=seed, image_prompt=img_p_b64, prompt_upsampling=prompt_upsampling, safety_tolerance=safety_tolerance, output_format=output_format)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux11Pro": SiliconFlowFlux11Pro}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux11Pro": "🎨 SiliconFlow — FLUX-1.1 Pro"}
