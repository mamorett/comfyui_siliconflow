"""
node_model_selector.py — Nodo ComfyUI per selezionare il modello SiliconFlow.

Recupera dinamicamente la lista dei modelli di image generation
tramite API e la presenta come dropdown.
"""

from .api_client import fetch_image_models


class SiliconFlowModelSelector:
    """
    Nodo che espone un dropdown con i modelli SiliconFlow
    disponibili per la generazione di immagini.

    La lista viene aggiornata dinamicamente dalle API.
    """

    CATEGORY = "SiliconFlow"
    FUNCTION = "select_model"
    RETURN_TYPES = ("SFMODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        # Recupera la lista modelli live (con cache)
        try:
            models = fetch_image_models()
        except Exception as e:
            print(f"[SiliconFlow] Impossibile recuperare i modelli: {e}")
            models = ["— Errore: controlla apikey.txt —"]

        return {
            "required": {
                "model": (models, {"default": models[0] if models else ""}),
            },
            "optional": {
                "refresh_models": ("BOOLEAN", {"default": False, "label_on": "🔄 Refresh", "label_off": "Usa cache"}),
            },
        }

    def select_model(self, model: str, refresh_models: bool = False) -> tuple:
        if refresh_models:
            try:
                models = fetch_image_models(force_refresh=True)
                print(f"[SiliconFlow] Lista modelli aggiornata: {len(models)} modelli trovati.")
            except Exception as e:
                print(f"[SiliconFlow] Errore refresh modelli: {e}")

        return (model,)


# Registrazione nodo
NODE_CLASS_MAPPINGS = {
    "SiliconFlowModelSelector": SiliconFlowModelSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SiliconFlowModelSelector": "🤖 SiliconFlow — Model Selector",
}
