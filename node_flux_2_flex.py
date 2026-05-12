from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models

class SiliconFlowFlux2Flex(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "flux.2-flex" in m.lower()]
        if not models: models = ["black-forest-labs/FLUX.2-flex"]
        return {
            "required": {
                "model": (models, {"default": models[0]}),
                "prompt": ("STRING", {"multiline": True}),
                "image_size": (["512x512", "768x1024", "1024x768", "576x1024", "1024x576"], {"default": "512x512"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
            },
            "optional": {
                "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20.0}),
                "output_format": (["png", "jpeg"], {"default": "png"}),
            }
        }

    def generate(self, model, prompt, image_size, seed, num_inference_steps=25, cfg=4.0, output_format="png"):
        return self._generate_common(model, prompt, image_size=image_size, seed=seed, num_inference_steps=num_inference_steps, cfg=cfg, output_format=output_format)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux2Flex": SiliconFlowFlux2Flex}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux2Flex": "🎨 SiliconFlow — FLUX.2 Flex"}
