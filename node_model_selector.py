"""
node_model_selector.py — ComfyUI node for selecting the SiliconFlow model.

Dynamically retrieves the list of image generation models
via API and presents it as a dropdown.
"""

from .api_client import fetch_image_models


class SiliconFlowModelSelector:
    """
    Node that exposes a dropdown with available SiliconFlow models
    for image generation.

    The list is dynamically updated from the APIs.
    """

    CATEGORY = "SiliconFlow"
    FUNCTION = "select_model"
    RETURN_TYPES = ("SFMODEL",)
    RETURN_NAMES = ("model",)

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

        return (model,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "SiliconFlowModelSelector": SiliconFlowModelSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SiliconFlowModelSelector": "🤖 SiliconFlow — Model Selector",
}
