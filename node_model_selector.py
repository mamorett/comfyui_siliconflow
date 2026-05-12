"""
node_model_selector.py — ComfyUI node for selecting the SiliconFlow model.

Dynamically retrieves the list of image generation models
via API and presents it as a dropdown.
Also provides model family information for conditional parameters.
"""

from .api_client import fetch_image_models


# Model family detection based on model ID
def get_model_family(model_id: str) -> str:
    """
    Detects the model family from the model ID string.
    Returns a family identifier used to show/hide relevant parameters.
    """
    model_lower = model_id.lower()

    if "flux.2-pro" in model_lower:
        return "FLUX.2-pro"
    elif "flux.2-flex" in model_lower:
        return "FLUX.2-flex"
    elif "flux-1.1-pro-ultra" in model_lower:
        return "FLUX-1.1-pro-Ultra"
    elif "flux-1.1-pro" in model_lower:
        return "FLUX-1.1-pro"
    elif "flux.1-kontext-dev" in model_lower:
        return "FLUX.1-Kontext-dev"
    elif "flux.1-kontext" in model_lower:
        return "FLUX.1-Kontext"
    elif "flux.1-schnell" in model_lower:
        return "FLUX.1-schnell"
    elif "flux.1-dev" in model_lower:
        return "FLUX.1-dev"
    elif "qwen-image" in model_lower:
        return "Qwen-Image"
    elif "z-image" in model_lower:
        return "Z-Image"
    else:
        return "Unknown"


# Image size presets by model family
IMAGE_SIZE_PRESETS = {
    "FLUX.2-pro": ["512x512", "768x1024", "1024x768", "576x1024", "1024x576"],
    "FLUX.2-flex": ["512x512", "768x1024", "1024x768", "576x1024", "1024x576"],
    "FLUX.1-schnell": ["1024x1024", "512x1024", "768x512", "768x1024", "1024x576", "576x1024"],
    "FLUX.1-dev": ["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280"],
    "FLUX-1.1-pro": [],  # Uses width/height integers
    "FLUX-1.1-pro-Ultra": ["1024x1024", "960x1280", "768x1024", "720x1440", "720x1280", "others"],
    "FLUX.1-Kontext": [],  # Uses aspect_ratio
    "FLUX.1-Kontext-dev": [],  # Uses image input only
    "Qwen-Image": ["1328x1328", "1664x928", "928x1664", "1472x1140", "1140x1472", "1584x1056", "1056x1584"],
    "Z-Image": ["512x512", "768x1024", "1024x576", "576x1024"],
    "Unknown": ["1024x1024", "512x512", "768x1024", "1024x768"],
}


class SiliconFlowModelSelector:
    """
    Node that exposes a dropdown with available SiliconFlow models
    for image generation.

    The list is dynamically updated from the APIs.
    Also outputs the model family for conditional parameter display.
    """

    CATEGORY = "SiliconFlow"
    FUNCTION = "select_model"
    RETURN_TYPES = ("SFMODEL", "STRING")
    RETURN_NAMES = ("model", "model_family")

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        # Retrieve live model list (with cache)
        try:
            models = fetch_image_models()
        except Exception as e:
            print(f"[SiliconFlow] Unable to retrieve models: {e}")
            models = ["— Error: check apikey.txt —"]

        return {
            "required": {
                "model": (models, {"default": models[0] if models else ""}),
            },
            "optional": {
                "refresh_button": ("BOOLEAN", {"default": False, "label": "🔄 Refresh Models"}),
            },
        }

    # Force re-execution when refresh_button is clicked
    @classmethod
    def IS_CHANGED(cls, refresh_button: bool = False, **kwargs):
        if refresh_button:
            return float("random")  # triggers re-execution
        return False

    def select_model(self, model: str, refresh_button: bool = False) -> tuple:
        if refresh_button:
            try:
                models = fetch_image_models(force_refresh=True)
                print(f"[SiliconFlow] Model list updated: {len(models)} models found.")
            except Exception as e:
                print(f"[SiliconFlow] Model refresh error: {e}")

        model_family = get_model_family(model)
        return (model, model_family)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SiliconFlowModelSelector": SiliconFlowModelSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SiliconFlowModelSelector": "🤖 SiliconFlow — Model Selector",
}
