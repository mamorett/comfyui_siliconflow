"""
config.py — Gestione configurazione e API key SiliconFlow.

La API key viene letta da 'apikey.txt' nella directory del custom node,
così non viene mai inclusa nei workflow di ComfyUI.
"""

import os

# Directory radice del custom node (dove si trova questo file)
NODE_DIR = os.path.dirname(os.path.abspath(__file__))
APIKEY_FILE = os.path.join(NODE_DIR, "apikey.txt")

SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"


def get_api_key() -> str:
    """
    Legge la API key dal file apikey.txt.
    Solleva un errore chiaro se il file non esiste o è vuoto.
    """
    if not os.path.exists(APIKEY_FILE):
        raise FileNotFoundError(
            f"[SiliconFlow] File API key non trovato: {APIKEY_FILE}\n"
            f"Crea il file e inserisci la tua API key SiliconFlow."
        )

    with open(APIKEY_FILE, "r", encoding="utf-8") as f:
        key = f.read().strip()

    if not key or key == "YOUR_SILICONFLOW_API_KEY_HERE":
        raise ValueError(
            f"[SiliconFlow] API key non configurata in: {APIKEY_FILE}\n"
            f"Sostituisci il testo placeholder con la tua API key reale."
        )

    return key
