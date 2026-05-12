"""
api_client.py — Client HTTP per le API SiliconFlow.

Gestisce:
- Recupero lista modelli (filtrati per image generation)
- Inferenza immagine (text-to-image e image editing)
- Cache della lista modelli per evitare chiamate ripetute
"""

import base64
import io
import json
import time
import urllib.request
import urllib.error
from typing import Optional

from .config import get_api_key, SILICONFLOW_BASE_URL

# Cache modelli: (timestamp, lista_modelli)
_model_cache: tuple[float, list[str]] | None = None
CACHE_TTL_SECONDS = 300  # 5 minuti

# Task type SiliconFlow che corrispondono a generazione immagini
IMAGE_GENERATION_TASK_TYPES = {
    "text2image",
    "image2image",
    "inpainting",
    "image-editing",
}

# Prefissi noti di modelli image generation come fallback
IMAGE_MODEL_PREFIXES = (
    "black-forest-labs/",
    "stabilityai/",
    "Kwai-Kolors/",
    "ByteDance/",
    "Pro/",
    "FLUX",
    "stable-diffusion",
    "sd-",
    "sdxl",
    "Image",
)


def _make_request(
    method: str,
    endpoint: str,
    payload: Optional[dict] = None,
    timeout: int = 120,
) -> dict:
    """Esegue una richiesta HTTP autenticata alle API SiliconFlow."""
    api_key = get_api_key()
    url = f"{SILICONFLOW_BASE_URL}{endpoint}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    data = json.dumps(payload).encode("utf-8") if payload else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"[SiliconFlow] Errore HTTP {e.code} su {endpoint}:\n{body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"[SiliconFlow] Errore di rete su {endpoint}: {e.reason}"
        ) from e


def _is_image_model(model: dict) -> bool:
    """
    Determina se un modello è per generazione immagini.
    Usa il campo 'supportedGenerationMethods' o 'task_type' se disponibile,
    altrimenti si basa su euristiche sul nome.
    """
    # Controllo task_type diretto
    task_type = model.get("task_type", "") or ""
    if task_type.lower() in {t.lower() for t in IMAGE_GENERATION_TASK_TYPES}:
        return True

    # Controllo array supportedGenerationMethods
    supported = model.get("supportedGenerationMethods", []) or []
    for method in supported:
        if any(img_task in str(method).lower() for img_task in ("image", "inpaint")):
            return True

    # Fallback: euristiche sul model id
    model_id = model.get("id", "") or ""
    if any(model_id.lower().startswith(p.lower()) for p in IMAGE_MODEL_PREFIXES):
        return True
    if any(
        kw in model_id.lower()
        for kw in ("flux", "stable-diff", "sdxl", "kolors", "imagen", "dall-e", "wanx")
    ):
        return True

    return False


def fetch_image_models(force_refresh: bool = False) -> list[str]:
    """
    Recupera e filtra i modelli di image generation da SiliconFlow.
    Usa una cache con TTL per ridurre le chiamate API.

    Returns:
        Lista ordinata di model ID per image generation.
    """
    global _model_cache

    now = time.time()
    if not force_refresh and _model_cache is not None:
        cached_time, cached_models = _model_cache
        if now - cached_time < CACHE_TTL_SECONDS:
            return cached_models

    all_models = []
    page = 1
    page_size = 100

    while True:
        try:
            resp = _make_request(
                "GET",
                f"/models?type=image&page={page}&page_size={page_size}",
                timeout=30,
            )
        except Exception:
            # Se la paginazione non è supportata, prova senza parametri
            resp = _make_request("GET", "/models", timeout=30)
            raw = resp.get("data", resp.get("models", []))
            all_models.extend(raw)
            break

        raw = resp.get("data", [])
        if not raw:
            break
        all_models.extend(raw)

        # Controlla se ci sono altre pagine
        total = resp.get("total", 0)
        if len(all_models) >= total or len(raw) < page_size:
            break
        page += 1

    # Filtra solo modelli image generation
    image_models = [
        m["id"] for m in all_models if isinstance(m, dict) and _is_image_model(m) and m.get("id")
    ]

    # Deduplica e ordina
    image_models = sorted(set(image_models))

    if not image_models:
        # Fallback hardcoded se l'API non restituisce nulla di utile
        image_models = [
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
            "stabilityai/stable-diffusion-3-5-large",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "Kwai-Kolors/Kolors",
        ]

    _model_cache = (now, image_models)
    return image_models


def run_inference(
    model: str,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    seed: int = -1,
    input_images: Optional[list[str]] = None,  # lista di base64 strings
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
) -> bytes:
    """
    Esegue l'inferenza immagine tramite SiliconFlow.

    Args:
        model: ID del modello da usare.
        prompt: Testo del prompt.
        width: Larghezza output in pixel.
        height: Altezza output in pixel.
        seed: Seed per la generazione (-1 = random).
        input_images: Lista di immagini in base64 (per modelli edit/img2img).
        negative_prompt: Prompt negativo (opzionale).
        num_inference_steps: Passi di diffusione.
        guidance_scale: Scala di guida CFG.

    Returns:
        Bytes dell'immagine PNG/JPEG generata.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "image_size": f"{width}x{height}",
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }

    if negative_prompt:
        payload["negative_prompt"] = negative_prompt

    if seed >= 0:
        payload["seed"] = seed

    # Se sono presenti immagini input, usale per editing/img2img
    if input_images:
        if len(input_images) == 1:
            payload["image"] = f"data:image/png;base64,{input_images[0]}"
        else:
            payload["images"] = [
                f"data:image/png;base64,{img}" for img in input_images
            ]

    resp = _make_request("POST", "/images/generations", payload=payload, timeout=120)

    # Estrai l'immagine dalla risposta
    images_data = resp.get("images", [])
    if not images_data:
        raise RuntimeError(
            f"[SiliconFlow] Nessuna immagine nella risposta: {resp}"
        )

    first = images_data[0]

    # La risposta può essere base64 o URL
    if isinstance(first, dict):
        if "b64_json" in first:
            return base64.b64decode(first["b64_json"])
        elif "url" in first:
            return _download_image(first["url"])
        else:
            raise RuntimeError(f"[SiliconFlow] Formato risposta immagine sconosciuto: {first}")
    elif isinstance(first, str):
        # Stringa diretta: potrebbe essere base64 o URL
        if first.startswith("http"):
            return _download_image(first)
        else:
            return base64.b64decode(first)
    else:
        raise RuntimeError(f"[SiliconFlow] Tipo risposta inatteso: {type(first)}")


def _download_image(url: str) -> bytes:
    """Scarica un'immagine da URL e restituisce i bytes."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-SiliconFlow/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except Exception as e:
        raise RuntimeError(f"[SiliconFlow] Errore download immagine da {url}: {e}") from e
