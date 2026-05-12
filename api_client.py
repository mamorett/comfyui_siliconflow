"""
api_client.py — HTTP client for SiliconFlow APIs.
"""

import base64
import io
import json
import time
import urllib.request
import urllib.error
from typing import Optional

from .config import get_api_key, SILICONFLOW_BASE_URL

# Model cache: (timestamp, model_list)
_model_cache: tuple[float, list[str]] | None = None
CACHE_TTL_SECONDS = 300  # 5 minutes

# SiliconFlow task types that correspond to image generation
IMAGE_GENERATION_TASK_TYPES = {
    "text2image",
    "image2image",
    "inpainting",
    "image-editing",
}

# Known image generation model prefixes as fallback (strictly spec-compliant)
IMAGE_MODEL_PREFIXES = (
    "black-forest-labs/",
    "Qwen/",
    "Tongyi-MAI/",
)


def _make_request(
    method: str,
    endpoint: str,
    payload: Optional[dict] = None,
    timeout: int = 120,
) -> dict:
    """Executes an authenticated HTTP request to SiliconFlow APIs."""
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
            f"[SiliconFlow] HTTP Error {e.code} on {endpoint}:\n{body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"[SiliconFlow] Network error on {endpoint}: {e.reason}"
        ) from e


def _is_image_model(model: dict) -> bool:
    """
    Determines if a model is for image generation.
    """
    # Check direct task_type
    task_type = model.get("task_type", "") or ""
    if task_type.lower() in {t.lower() for t in IMAGE_GENERATION_TASK_TYPES}:
        return True

    # Check supportedGenerationMethods array
    supported = model.get("supportedGenerationMethods", []) or []
    for method in supported:
        if any(img_task in str(method).lower() for img_task in ("image", "inpaint")):
            return True

    # Fallback: heuristics on model id (strict)
    model_id = model.get("id", "") or ""
    if any(model_id.lower().startswith(p.lower()) for p in IMAGE_MODEL_PREFIXES):
        return True
    if any(
        kw in model_id.lower()
        for kw in ("flux", "qwen", "z-image")
    ):
        return True

    return False


def fetch_image_models(force_refresh: bool = False) -> list[str]:
    """
    Retrieves and filters image generation models from SiliconFlow.
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
            resp = _make_request("GET", "/models", timeout=30)
            raw = resp.get("data", resp.get("models", []))
            all_models.extend(raw)
            break

        raw = resp.get("data", [])
        if not raw:
            break
        all_models.extend(raw)

        total = resp.get("total", 0)
        if len(all_models) >= total or len(raw) < page_size:
            break
        page += 1

    image_models = [
        m["id"] for m in all_models if isinstance(m, dict) and _is_image_model(m) and m.get("id")
    ]

    image_models = sorted(set(image_models))

    if not image_models:
        image_models = [
            "black-forest-labs/FLUX.1-dev",
            "black-forest-labs/FLUX.1-schnell",
            "black-forest-labs/FLUX.1-pro",
            "black-forest-labs/FLUX-1.1-pro",
            "black-forest-labs/FLUX-1.1-pro-Ultra",
            "Qwen/Qwen-Image",
            "Tongyi-MAI/Z-Image-Turbo",
        ]

    _model_cache = (now, image_models)
    return image_models


def run_inference(
    model: str,
    prompt: str,
    **kwargs
) -> tuple[list[bytes], int]:
    """
    Executes image inference via SiliconFlow.
    Returns (list of image bytes, seed used).
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
    }

    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        if key == "seed" and value < 0:
            continue
        if key in ("image", "input_image", "image_prompt") and isinstance(value, str):
            if not value.startswith("data:"):
                payload[key] = f"data:image/png;base64,{value}"
            else:
                payload[key] = value
        else:
            payload[key] = value

    resp = _make_request("POST", "/images/generations", payload=payload, timeout=120)

    images_data = resp.get("images", [])
    if not images_data:
        raise RuntimeError(f"[SiliconFlow] No image in response: {resp}")

    results = []
    for item in images_data:
        if isinstance(item, dict):
            if "b64_json" in item:
                results.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                results.append(_download_image(item["url"]))
        elif isinstance(item, str):
            if item.startswith("http"):
                results.append(_download_image(item))
            else:
                results.append(base64.b64decode(item))
    
    # Extract seed from response
    actual_seed = resp.get("seed", kwargs.get("seed", -1))
    
    return results, actual_seed


def _download_image(url: str) -> bytes:
    """Downloads an image from URL and returns the bytes."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-SiliconFlow/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except Exception as e:
        raise RuntimeError(f"[SiliconFlow] Error downloading image from {url}: {e}") from e
