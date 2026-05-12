from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models

class SiliconFlowZImage(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "z-image" in m.lower() or "tongyi" in m.lower()]
        if not models: models = ["Tongyi-MAI/Z-Image-Turbo"]
        return {
            "required": {
                "model": (models, {"default": models[0]}),
                "prompt": ("STRING", {"multiline": True}),
                "image_size": (["512x512", "768x1024", "1024x576", "576x1024"], {"default": "512x512"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True}),
            }
        }

    def generate(self, model, prompt, image_size, seed, negative_prompt=""):
        return self._generate_common(model, prompt, image_size=image_size, seed=seed, negative_prompt=negative_prompt)

NODE_CLASS_MAPPINGS = {"SiliconFlowZImage": SiliconFlowZImage}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowZImage": "🎨 SiliconFlow — Z-Image"}
