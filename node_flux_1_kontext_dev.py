from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models
from .utils import tensor_to_base64

class SiliconFlowFlux1KontextDev(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "flux.1-kontext-dev" in m.lower()]
        if not models: models = ["black-forest-labs/FLUX.1-Kontext-dev"]
        return {
            "required": {
                "model": (models, {"default": models[0]}),
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE", {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
            },
            "optional": {
                "prompt_enhancement": ("BOOLEAN", {"default": False}),
            }
        }

    def generate(self, model, prompt, image, seed, prompt_enhancement=False):
        img_b64 = tensor_to_base64(image)
        return self._generate_common(model, prompt, image=img_b64, seed=seed, prompt_enhancement=prompt_enhancement)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux1KontextDev": SiliconFlowFlux1KontextDev}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux1KontextDev": "🎨 SiliconFlow — FLUX.1 Kontext Dev"}
