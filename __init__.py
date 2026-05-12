"""
__init__.py — Entry point for SiliconFlow custom node for ComfyUI.

Registers all available nodes and exposes them to ComfyUI
via the standard variables NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS.

Project structure:
  comfyui_siliconflow/
  ├── __init__.py              ← this file
  ├── config.py                ← API key management
  ├── api_client.py            ← SiliconFlow HTTP client
  ├── node_inference.py        ← unified image generation node
  ├── apikey.txt               ← API key (DO NOT include in workflows!)
  └── .gitignore               ← excludes apikey.txt from git
"""

from .node_inference import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

# Package metadata (optional but useful for ComfyUI Manager)
WEB_DIRECTORY = None  # no JS/CSS frontend assets

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(
    f"[SiliconFlow] Custom nodes loaded: "
    + ", ".join(NODE_DISPLAY_NAME_MAPPINGS.values())
)