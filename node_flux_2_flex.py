from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models

class SiliconFlowFlux2Flex(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "flux.2-flex" in m.lower()]
        if not models: models = ["black-forest-labs/FLUX.2-flex"]
        return {
            "required": {
                "model": (models, {"default": models[0], "tooltip": "Black Forest Labs FLUX.2 Flex model ID."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Text description of the image."}),
                "image_size": (["512x512", "768x1024", "1024x768", "576x1024", "1024x576"], {"default": "512x512", "tooltip": "Supported resolution presets."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 9999999999, "tooltip": "Random seed. Use -1 for random."}),
            },
            "optional": {
                "num_inference_steps": ("INT", {"default": 25, "min": 1, "max": 50, "tooltip": "Number of denoising steps. More steps = higher quality, slower generation."}),
                "output_format": (["png", "jpeg"], {"default": "png", "tooltip": "Final image format."}),
            }
        }

    def generate(self, model, prompt, image_size, seed, num_inference_steps=25, output_format="png"):
        return self._generate_common(model, prompt, image_size=image_size, seed=seed, num_inference_steps=num_inference_steps, output_format=output_format)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux2Flex": SiliconFlowFlux2Flex}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux2Flex": "🎨 SiliconFlow — FLUX.2 Flex"}
