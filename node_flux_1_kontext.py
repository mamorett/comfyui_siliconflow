from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models
from .utils import tensor_to_base64

class SiliconFlowFlux1Kontext(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "flux.1-kontext" in m.lower() and "dev" not in m.lower()]
        if not models: models = ["black-forest-labs/FLUX.1-Kontext-max", "black-forest-labs/FLUX.1-Kontext-pro"]
        return {
            "required": {
                "model": (models, {"default": models[0], "tooltip": "FLUX.1 Kontext model ID."}),
                "prompt": ("STRING", {"multiline": True, "tooltip": "Text prompt."}),
                "image": ("IMAGE", {"tooltip": "REQUIRED: Input image for Kontext model."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999, "tooltip": "Random seed."}),
            },
            "optional": {
                "aspect_ratio": ("STRING", {"default": "1:1", "tooltip": "Aspect ratio between 21:9 and 9:21."}),
                "output_format": (["png", "jpeg"], {"default": "png", "tooltip": "Output format."}),
                "prompt_upsampling": ("BOOLEAN", {"default": False, "tooltip": "Automatically modifies prompt for more creative results."}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6, "tooltip": "Filter strictness (0: strictest, 6: most lenient). Img2Img capped at 2."}),
            }
        }

    def generate(self, model, prompt, image, seed, aspect_ratio="1:1", output_format="png", prompt_upsampling=False, safety_tolerance=2):
        img_b64 = tensor_to_base64(image)
        return self._generate_common(model, prompt, input_image=img_b64, seed=seed, aspect_ratio=aspect_ratio, output_format=output_format, prompt_upsampling=prompt_upsampling, safety_tolerance=safety_tolerance)

NODE_CLASS_MAPPINGS = {"SiliconFlowFlux1Kontext": SiliconFlowFlux1Kontext}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowFlux1Kontext": "🎨 SiliconFlow — FLUX.1 Kontext"}
