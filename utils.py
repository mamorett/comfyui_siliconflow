"""
utils.py — Shared utility functions for SiliconFlow nodes.
"""

import io
import base64
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def tensor_to_base64(tensor) -> str:
    """
    Converts a ComfyUI tensor (B,H,W,C) float32 [0,1] to a base64 PNG string.
    If there are multiple batches, uses the first frame.
    """
    if tensor is None:
        return None

    if HAS_TORCH and isinstance(tensor, torch.Tensor):
        img_array = tensor[0].cpu().numpy()
    else:
        img_array = np.array(tensor)[0]

    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)

    if HAS_PIL:
        pil_img = Image.fromarray(img_array, mode="RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    else:
        raise RuntimeError("[SiliconFlow] PIL not available for image encoding.")


def bytes_to_tensor(image_bytes_list: list[bytes]):
    """
    Converts a list of PNG/JPEG image bytes to a single ComfyUI tensor (B,H,W,C) float32 [0,1].
    """
    tensors = []
    for image_bytes in image_bytes_list:
        if HAS_PIL:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            arr = np.array(pil_img, dtype=np.float32) / 255.0
        else:
            raise RuntimeError("[SiliconFlow] PIL not available for image decoding.")

        if HAS_TORCH:
            tensors.append(torch.from_numpy(arr))
        else:
            tensors.append(arr)

    if not tensors:
        return None

    if HAS_TORCH:
        return torch.stack(tensors)  # (B, H, W, C)
    else:
        return np.stack(tensors)  # (B, H, W, C)
