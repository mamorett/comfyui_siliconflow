"""
__init__.py — Entry point for SiliconFlow custom nodes for ComfyUI.
"""

from .node_flux_2_pro import NODE_CLASS_MAPPINGS as n1, NODE_DISPLAY_NAME_MAPPINGS as d1
from .node_flux_2_flex import NODE_CLASS_MAPPINGS as n2, NODE_DISPLAY_NAME_MAPPINGS as d2
from .node_qwen_image import NODE_CLASS_MAPPINGS as n3, NODE_DISPLAY_NAME_MAPPINGS as d3
from .node_z_image import NODE_CLASS_MAPPINGS as n4, NODE_DISPLAY_NAME_MAPPINGS as d4
from .node_flux_1_kontext import NODE_CLASS_MAPPINGS as n5, NODE_DISPLAY_NAME_MAPPINGS as d5
from .node_flux_1_kontext_dev import NODE_CLASS_MAPPINGS as n6, NODE_DISPLAY_NAME_MAPPINGS as d6
from .node_flux_11_pro import NODE_CLASS_MAPPINGS as n7, NODE_DISPLAY_NAME_MAPPINGS as d7
from .node_flux_11_pro_ultra import NODE_CLASS_MAPPINGS as n8, NODE_DISPLAY_NAME_MAPPINGS as d8
from .node_flux_1_schnell import NODE_CLASS_MAPPINGS as n9, NODE_DISPLAY_NAME_MAPPINGS as d9
from .node_flux_1_dev import NODE_CLASS_MAPPINGS as n10, NODE_DISPLAY_NAME_MAPPINGS as d10

NODE_CLASS_MAPPINGS = {**n1, **n2, **n3, **n4, **n5, **n6, **n7, **n8, **n9, **n10}
NODE_DISPLAY_NAME_MAPPINGS = {**d1, **d2, **d3, **d4, **d5, **d6, **d7, **d8, **d9, **d10}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"[SiliconFlow] {len(NODE_CLASS_MAPPINGS)} specialized nodes loaded.")
