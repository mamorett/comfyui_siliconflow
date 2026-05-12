from .base import SiliconFlowBaseNode
from .api_client import fetch_image_models
from .utils import tensor_to_base64

class SiliconFlowQwenImage(SiliconFlowBaseNode):
    @classmethod
    def INPUT_TYPES(cls):
        models = [m for m in fetch_image_models() if "qwen-image" in m.lower()]
        if not models: models = ["Qwen/Qwen-Image", "Qwen/Qwen-Image-Edit"]
        return {
            "required": {
                "model": (models, {"default": models[0]}),
                "prompt": ("STRING", {"multiline": True}),
                "image_size": (["1328x1328", "1664x928", "928x1664", "1472x1140", "1140x1472", "1584x1056", "1056x1584"], {"default": "1328x1328"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4}),
                "num_inference_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": ("FLOAT", {"default": 7.5, "min": 0, "max": 20}),
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 20}),
                "image": ("IMAGE", {}),
            }
        }

    def generate(self, model, prompt, image_size, seed, negative_prompt="", batch_size=1, num_inference_steps=20, guidance_scale=7.5, cfg=4.0, image=None):
        img_b64 = tensor_to_base64(image) if image is not None else None
        return self._generate_common(model, prompt, image_size=image_size, seed=seed, negative_prompt=negative_prompt, batch_size=batch_size, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, cfg=cfg, image=img_b64)

NODE_CLASS_MAPPINGS = {"SiliconFlowQwenImage": SiliconFlowQwenImage}
NODE_DISPLAY_NAME_MAPPINGS = {"SiliconFlowQwenImage": "🎨 SiliconFlow — Qwen Image"}
