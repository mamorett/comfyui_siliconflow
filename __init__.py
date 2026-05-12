"""
__init__.py — Entry point del custom node SiliconFlow per ComfyUI.

Registra tutti i nodi disponibili e li espone a ComfyUI
tramite le variabili standard NODE_CLASS_MAPPINGS e NODE_DISPLAY_NAME_MAPPINGS.

Struttura del progetto:
  comfyui_siliconflow/
  ├── __init__.py              ← questo file
  ├── config.py                ← gestione API key
  ├── api_client.py            ← client HTTP SiliconFlow
  ├── node_model_selector.py   ← nodo selezione modello
  ├── node_inference.py        ← nodo inferenza immagine
  ├── apikey.txt               ← API key (NON includere nei workflow!)
  └── .gitignore               ← esclude apikey.txt dal git
"""

from .node_model_selector import (
    NODE_CLASS_MAPPINGS as _MODEL_SELECTOR_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MODEL_SELECTOR_NAMES,
)
from .node_inference import (
    NODE_CLASS_MAPPINGS as _INFERENCE_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _INFERENCE_NAMES,
)

# Merge di tutti i mapping — aggiungi qui nuovi nodi in futuro
NODE_CLASS_MAPPINGS = {
    **_MODEL_SELECTOR_CLASSES,
    **_INFERENCE_CLASSES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_MODEL_SELECTOR_NAMES,
    **_INFERENCE_NAMES,
}

# Metadati del pacchetto (opzionali ma utili per ComfyUI Manager)
WEB_DIRECTORY = None  # nessun asset JS/CSS frontend

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(
    f"[SiliconFlow] Custom nodes caricati: "
    + ", ".join(NODE_DISPLAY_NAME_MAPPINGS.values())
)
