"""
base.py — Base classes for SiliconFlow nodes.
"""

from .api_client import run_inference
from .utils import bytes_to_tensor

class SiliconFlowBaseNode:
    CATEGORY = "SiliconFlow"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "seed")
    FUNCTION = "generate"
    OUTPUT_NODE = False

    def _generate_common(self, model, prompt, **kwargs):
        try:
            image_bytes_list, actual_seed = run_inference(
                model=model,
                prompt=prompt,
                **kwargs
            )
            output_tensor = bytes_to_tensor(image_bytes_list)
            return (output_tensor, actual_seed)
        except Exception as e:
            raise RuntimeError(f"[SiliconFlow] Inference error: {e}")
