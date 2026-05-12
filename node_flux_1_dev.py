from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models

class SiliconFlowFlux1Dev(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "flux.1-dev" in m.lower() and "kontext" not in m.lower()]
        if not models: models = ["black-forest-labs/FLUX.1-dev"]
        return {
            "required": {
                "model": (models, {"default": models[0], "tooltip": "FLUX.1 Dev model ID."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Description."}),
                "image_size": (["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280", "others"], {"default": "1024x1024", "tooltip": "Size presets (max 2.3M pixels)."}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 30, "tooltip": "Inference steps (1-30)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999, "tooltip": "Random seed."}),
            },
            "optional": {
                "prompt_enhancement": ("BOOLEAN", {"default": False, "tooltip": "Detail-optimized prompt."}),
            }
        }

    def generate(self, model, prompt, image_size, num_inference_steps, seed, prompt_enhancement=False):
        return self._generate_common(model, prompt, image_size=image_size, num_inference_steps=num_inference_steps, seed=seed, prompt_enhancement=prompt_enhancement)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux1Dev": SiliconFlowFlux1Dev}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux1Dev": "🎨 SiliconFlow — FLUX.1 Dev"}
