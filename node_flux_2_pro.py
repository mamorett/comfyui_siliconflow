from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models

class SiliconFlowFlux2Pro(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "flux.2-pro" in m.lower()]
        if not models: models = ["black-forest-labs/FLUX.2-pro"]
        return {
            "required": {
                "model": (models, {"default": models[0]}),
                "prompt": ("STRING", {"multiline": True}),
                "image_size": (["512x512", "768x1024", "1024x768", "576x1024", "1024x576"], {"default": "512x512"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
            }
        }

    def generate(self, model, prompt, image_size, seed, output_format):
        return self._generate_common(model, prompt, image_size=image_size, seed=seed, output_format=output_format)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux2Pro": SiliconFlowFlux2Pro}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux2Pro": "🎨 SiliconFlow — FLUX.2 Pro"}
