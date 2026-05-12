"""
__init__.py — Entry point for SiliconFlow custom node for ComfyUI.

Registers all available nodes and exposes them to ComfyUI
via the standard variables NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

Project structure:
  comfyui_siliconflow/
  ├── __init__.py              ← this file
  ├── config.py                ← API key management
  ├── api_client.py            ← SiliconFlow HTTP client
  ├── node_model_selector.py   ← model selection node
  ├── node_inference.py        ← image inference node
  ├── apikey.txt               ← API key (DO NOT include in workflows!)
  └── .gitignore               ← excludes apikey.txt from git
"""

from .node_model_selector import (
    NODE_CLASS_MAPPINGS as _MODEL_SELECTOR_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _MODEL_SELECTOR_NAMES,
)
from .node_inference import (
    NODE_CLASS_MAPPINGS as _INFERENCE_CLASSES,
    NODE_DISPLAY_NAME_MAPPINGS as _INFERENCE_NAMES,
)

# Merge all mappings — add new nodes here in the future
NODE_CLASS_MAPPINGS = {
    **_MODEL_SELECTOR_CLASSES,
    **_INFERENCE_CLASSES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_MODEL_SELECTOR_NAMES,
    **_INFERENCE_NAMES,
}

# Package metadata (optional but useful for ComfyUI Manager)
WEB_DIRECTORY = None  # no JS/CSS frontend assets

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(
    f"[SiliconFlow] Custom nodes loaded: "
    + ", ".join(NODE_DISPLAY_NAME_MAPPINGS.values())
)
