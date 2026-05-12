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
                "model": (models, {"default": models[0], "tooltip": "FLUX.1 Kontext Dev model ID."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Text prompt."}),
                "image": ("IMAGE", {"tooltip": "REQUIRED: Input image."}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 9999999999, "tooltip": "Random seed. Use -1 for random."}),
            },
            "optional": {
                "prompt_enhancement": ("BOOLEAN", {"default": False, "tooltip": "Optimizes prompt to be more detailed and model-friendly."}),
            }
        }

    def generate(self, model, prompt, image, seed, prompt_enhancement=False):
        img_b64 = tensor_to_base64(image)
        return self._generate_common(model, prompt, image=img_b64, seed=seed, prompt_enhancement=prompt_enhancement)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux1KontextDev": SiliconFlowFlux1KontextDev}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux1KontextDev": "🎨 SiliconFlow — FLUX.1 Kontext Dev"}
