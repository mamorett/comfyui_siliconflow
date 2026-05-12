"""
config.py — Configuration management and SiliconFlow API key handling.

The API key is read from 'apikey.txt' in the custom node directory,
so it is never included in ComfyUI workflows.
"""

import os

# Root directory of the custom node (where this file is located)
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
APIKEY_FILE = os.path.join(NODE_DIR, "apikey.txt")

SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"


def get_api_key() -> str:
    """
    Reads the API key from the apikey.txt file.
    Raises a clear error if the file does not exist or is empty.
    """
    if not os.path.exists(APIKEY_FILE):
        raise FileNotFoundError(
            f"[SiliconFlow] API key file not found: {APIKEY_FILE}\n"
            f"Create the file and insert your SiliconFlow API key."
        )

    with open(APIKEY_FILE, "r", encoding="utf-8") as f:
        key = f.read().strip()

    if not key or key == "YOUR_SILICONFLOW_API_KEY_HERE":
        raise ValueError(
            f"[SiliconFlow] API key not configured in: {APIKEY_FILE}\n"
            f"Replace the placeholder text with your actual API key."
        )

    return key