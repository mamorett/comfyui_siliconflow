"""
api_client.py — HTTP client for SiliconFlow APIs.

Handles:
- Model list retrieval (filtered for image generation)
- Image inference (text-to-image and image editing)
- Model list caching to avoid repeated API calls
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

# Known image generation model prefixes as fallback
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
    Uses the 'supportedGenerationMethods' or 'task_type' field if available,
    otherwise relies on heuristics on the name.
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

    # Fallback: heuristics on model id
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
    Retrieves and filters image generation models from SiliconFlow.
    Uses a cache with TTL to reduce API calls.

    Returns:
        Sorted list of model IDs for image generation.
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
            # If pagination is not supported, try without parameters
            resp = _make_request("GET", "/models", timeout=30)
            raw = resp.get("data", resp.get("models", []))
            all_models.extend(raw)
            break

        raw = resp.get("data", [])
        if not raw:
            break
        all_models.extend(raw)

        # Check if there are more pages
        total = resp.get("total", 0)
        if len(all_models) >= total or len(raw) < page_size:
            break
        page += 1

    # Filter only image generation models
    image_models = [
        m["id"] for m in all_models if isinstance(m, dict) and _is_image_model(m) and m.get("id")
    ]

    # Deduplicate and sort
    image_models = sorted(set(image_models))

    if not image_models:
        # Hardcoded fallback if the API returns nothing useful
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
    image_size: str = "1024x1024",
    seed: int = -1,
    input_image: Optional[str] = None,  # single base64 string
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    cfg: Optional[float] = None,
    output_format: str = "png",
    prompt_upsampling: Optional[bool] = None,
    prompt_enhancement: Optional[bool] = None,
    safety_tolerance: Optional[int] = None,
    image_prompt: Optional[str] = None,
    image_prompt_strength: Optional[float] = None,
    raw: Optional[bool] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    aspect_ratio: Optional[str] = None,
) -> bytes:
    """
    Executes image inference via SiliconFlow.

    Args:
        model: Model ID to use.
        prompt: Prompt text.
        image_size: Image size as "widthxheight" string.
        seed: Seed for generation (-1 = random).
        input_image: Single base64 image (for edit/img2img models).
        negative_prompt: Negative prompt (optional).
        num_inference_steps: Diffusion steps.
        guidance_scale: CFG guidance scale.
        cfg: CFG scale (for FLUX.2-flex and Qwen-Image models).
        output_format: Output format (png/jpeg).
        prompt_upsampling: Whether to upsample the prompt (FLUX.1-Kontext, FLUX-1.1-pro).
        prompt_enhancement: Prompt enhancement switch (FLUX.1-schnell, FLUX.1-dev).
        safety_tolerance: Tolerance level 0-6 (FLUX.1-Kontext, FLUX-1.1-pro).
        image_prompt: Base64 image prompt (FLUX-1.1-pro-Ultra).
        image_prompt_strength: Blend strength 0-1 (FLUX-1.1-pro-Ultra).
        raw: Generate less processed images (FLUX-1.1-pro-Ultra).
        width: Width in pixels (for FLUX-1.1-pro).
        height: Height in pixels (for FLUX-1.1-pro).
        aspect_ratio: Aspect ratio string (for FLUX.1-Kontext).

    Returns:
        Bytes of the generated PNG/JPEG image.
    """
    payload: dict = {
        "model": model,
        "prompt": prompt,
    }

    # Add parameters only if they are set (not None/empty)
    if image_size:
        payload["image_size"] = image_size
    if num_inference_steps is not None:
        payload["num_inference_steps"] = num_inference_steps
    if guidance_scale is not None:
        payload["guidance_scale"] = guidance_scale
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    if seed >= 0:
        payload["seed"] = seed
    if cfg is not None:
        payload["cfg"] = cfg
    if output_format:
        payload["output_format"] = output_format
    if prompt_upsampling is not None:
        payload["prompt_upsampling"] = prompt_upsampling
    if prompt_enhancement is not None:
        payload["prompt_enhancement"] = prompt_enhancement
    if safety_tolerance is not None:
        payload["safety_tolerance"] = safety_tolerance
    if image_prompt:
        payload["image_prompt"] = f"data:image/png;base64,{image_prompt}"
    if image_prompt_strength is not None:
        payload["image_prompt_strength"] = image_prompt_strength
    if raw is not None:
        payload["raw"] = raw
    if width is not None:
        payload["width"] = width
    if height is not None:
        payload["height"] = height
    if aspect_ratio:
        payload["aspect_ratio"] = aspect_ratio

    # Handle input image - API uses "image" for some models, "input_image" for others
    if input_image:
        # Check if this is a FLUX.1-Kontext model (uses input_image)
        if "flux.1-kontext" in model.lower() and "dev" not in model.lower():
            payload["input_image"] = f"data:image/png;base64,{input_image}"
        else:
            payload["image"] = f"data:image/png;base64,{input_image}"

    resp = _make_request("POST", "/images/generations", payload=payload, timeout=120)

    # Extract image from response
    images_data = resp.get("images", [])
    if not images_data:
        raise RuntimeError(
            f"[SiliconFlow] No image in response: {resp}"
        )

    first = images_data[0]

    # Response can be base64 or URL
    if isinstance(first, dict):
        if "b64_json" in first:
            return base64.b64decode(first["b64_json"])
        elif "url" in first:
            return _download_image(first["url"])
        else:
            raise RuntimeError(f"[SiliconFlow] Unknown image response format: {first}")
    elif isinstance(first, str):
        # Direct string: could be base64 or URL
        if first.startswith("http"):
            return _download_image(first)
        else:
            return base64.b64decode(first)
    else:
        raise RuntimeError(f"[SiliconFlow] Unexpected response type: {type(first)}")


def _download_image(url: str) -> bytes:
    """Downloads an image from URL and returns the bytes."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-SiliconFlow/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except Exception as e:
        raise RuntimeError(f"[SiliconFlow] Error downloading image from {url}: {e}") from e