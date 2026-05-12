from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models

class SiliconFlowFlux1Schnell(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "flux.1-schnell" in m.lower()]
        if not models: models = ["black-forest-labs/FLUX.1-schnell"]
        return {
            "required": {
                "model": (models, {"default": models[0], "tooltip": "FLUX.1 Schnell model ID."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Image prompt."}),
                "image_size": (["1024x1024", "512x1024", "768x512", "768x1024", "1024x576", "576x1024"], {"default": "1024x1024", "tooltip": "Size presets."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999, "tooltip": "Random seed."}),
            },
            "optional": {
                "prompt_enhancement": ("BOOLEAN", {"default": False, "tooltip": "Detail-optimized prompt."}),
            }
        }

    def generate(self, model, prompt, image_size, seed, prompt_enhancement=False):
        return self._generate_common(model, prompt, image_size=image_size, seed=seed, prompt_enhancement=prompt_enhancement)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux1Schnell": SiliconFlowFlux1Schnell}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux1Schnell": "🎨 SiliconFlow — FLUX.1 Schnell"}
