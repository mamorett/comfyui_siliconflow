"""
node_inference.py — Nodo ComfyUI per l'inferenza immagine tramite SiliconFlow.

Input:
  - model       : stringa modello (dal nodo SiliconFlowModelSelector)
  - prompt      : testo del prompt
  - image_1..4  : immagini opzionali (tensori ComfyUI) per modelli edit/img2img
  - width       : larghezza output
  - height      : altezza output
  - seed        : seed (−1 = random)
  - random_seed : se True, ignora seed e usa casuale ad ogni run

Output:
  - IMAGE        : tensore immagine ComfyUI (B,H,W,C) float32 [0,1]
"""

import io
import random
import time

import numpy as np

# Import opzionale torch — ComfyUI lo ha sempre disponibile
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import PIL — presente nell'ambiente ComfyUI
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from .api_client import run_inference


def _tensor_to_base64(tensor) -> str:
    """
    Converte un tensore ComfyUI (B,H,W,C) float32 [0,1] in stringa base64 PNG.
    Se ci sono più batch, usa il primo frame.
    """
    import base64

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
        # Fallback senza PIL: PPM grezzo → base64
        import base64 as b64mod
        h, w, c = img_array.shape
        header = f"P6\n{w} {h}\n255\n".encode()
        raw = header + img_array.tobytes()
        return b64mod.b64encode(raw).decode("utf-8")


def _bytes_to_tensor(image_bytes: bytes):
    """
    Converte bytes di immagine PNG/JPEG in tensore ComfyUI (1,H,W,C) float32 [0,1].
    """
    if HAS_PIL:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.array(pil_img, dtype=np.float32) / 255.0
    else:
        raise RuntimeError(
            "[SiliconFlow] PIL non disponibile: impossibile decodificare l'immagine."
        )

    if HAS_TORCH:
        return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W, C)
    else:
        return arr[np.newaxis, ...]  # (1, H, W, C) numpy


class SiliconFlowInference:
    """
    Nodo di inferenza principale per SiliconFlow.
    Supporta text-to-image e image editing (fino a 4 immagini input).
    """

    CATEGORY = "SiliconFlow"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("SFMODEL", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "A beautiful landscape, photorealistic, 8k",
                        "placeholder": "Descrivi l'immagine da generare...",
                    },
                ),
                "width": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 256, "max": 2048, "step": 64},
                ),
                "seed": (
                    "INT",
                    {"default": 42, "min": 0, "max": 2**32 - 1},
                ),
                "random_seed": (
                    "BOOLEAN",
                    {"default": False, "label_on": "🎲 Random", "label_off": "Fisso"},
                ),
                "num_steps": (
                    "INT",
                    {"default": 20, "min": 1, "max": 100, "step": 1},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 7.5, "min": 0.0, "max": 20.0, "step": 0.5},
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Elementi da evitare nell'immagine...",
                    },
                ),
                "image_1": ("IMAGE", {}),
                "image_2": ("IMAGE", {}),
                "image_3": ("IMAGE", {}),
                "image_4": ("IMAGE", {}),
            },
        }

    # Forza riesecuzione quando random_seed=True
    @classmethod
    def IS_CHANGED(cls, random_seed: bool = False, **kwargs):
        if random_seed:
            return time.time()  # valore sempre diverso → riesegue sempre
        return False

    def generate(
        self,
        model: str,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        random_seed: bool,
        num_steps: int,
        guidance_scale: float,
        negative_prompt: str = "",
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
    ) -> tuple:

        # Risolvi seed
        effective_seed = random.randint(0, 2**32 - 1) if random_seed else seed
        print(f"[SiliconFlow] Generazione con seed: {effective_seed}")

        # Raccogli le immagini input opzionali
        input_images_b64 = []
        for idx, img_tensor in enumerate([image_1, image_2, image_3, image_4], start=1):
            if img_tensor is not None:
                try:
                    b64 = _tensor_to_base64(img_tensor)
                    input_images_b64.append(b64)
                    print(f"[SiliconFlow] Immagine {idx} convertita in base64.")
                except Exception as e:
                    print(f"[SiliconFlow] Errore conversione immagine {idx}: {e}")

        print(
            f"[SiliconFlow] Avvio inferenza — modello: {model} | "
            f"size: {width}x{height} | immagini input: {len(input_images_b64)}"
        )

        try:
            image_bytes = run_inference(
                model=model,
                prompt=prompt,
                width=width,
                height=height,
                seed=effective_seed,
                input_images=input_images_b64 if input_images_b64 else None,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
            )
        except Exception as e:
            raise RuntimeError(f"[SiliconFlow] Errore inferenza: {e}") from e

        # Converti bytes → tensore ComfyUI
        output_tensor = _bytes_to_tensor(image_bytes)
        print(f"[SiliconFlow] ✅ Immagine generata con successo! Shape: {output_tensor.shape}")

        return (output_tensor,)


# Registrazione nodo
NODE_CLASS_MAPPINGS = {
    "SiliconFlowInference": SiliconFlowInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SiliconFlowInference": "🎨 SiliconFlow — Image Generation",
}
