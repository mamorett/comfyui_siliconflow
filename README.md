# ComfyUI — SiliconFlow Custom Nodes

Nodi ComfyUI per generare immagini tramite le API di **SiliconFlow**.

## Struttura

```
comfyui_siliconflow/
├── __init__.py              # Entry point e registrazione nodi
├── config.py                # Gestione API key
├── api_client.py            # Client HTTP SiliconFlow
├── node_model_selector.py   # Nodo: selezione modello
├── node_inference.py        # Nodo: generazione immagine
├── apikey.txt               # ← API key (NON condividere!)
└── .gitignore               # Esclude apikey.txt dal repository
```

## Installazione

1. **Copia la cartella** nella directory dei custom nodes di ComfyUI:
   ```
   ComfyUI/custom_nodes/comfyui_siliconflow/
   ```

2. **Inserisci la tua API key** nel file `apikey.txt`:
   ```
   sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```
   > ⚠️ Non condividere mai questo file. È escluso automaticamente da `.gitignore`.

3. **Riavvia ComfyUI**. I nodi appariranno nella categoria **SiliconFlow**.

## Nodi disponibili

### 🤖 SiliconFlow — Model Selector

Recupera dinamicamente la lista dei modelli di image generation da SiliconFlow e la presenta come dropdown.

| Input | Tipo | Descrizione |
|-------|------|-------------|
| `model` | Dropdown | Modello selezionato |
| `refresh_models` | Boolean | Forza aggiornamento lista |

| Output | Tipo | Descrizione |
|--------|------|-------------|
| `model` | SFMODEL | ID modello da passare al nodo inferenza |

---

### 🎨 SiliconFlow — Image Generation

Nodo di inferenza principale. Supporta text-to-image e image editing.

| Input | Tipo | Obbligatorio | Descrizione |
|-------|------|:---:|-------------|
| `model` | SFMODEL | ✅ | Modello dal nodo selector |
| `prompt` | String | ✅ | Testo del prompt |
| `width` | Int | ✅ | Larghezza output (256–2048) |
| `height` | Int | ✅ | Altezza output (256–2048) |
| `seed` | Int | ✅ | Seed per la generazione |
| `random_seed` | Boolean | ✅ | Se attivo, usa seed casuale |
| `num_steps` | Int | ✅ | Passi di diffusione (1–100) |
| `guidance_scale` | Float | ✅ | Scala CFG (0–20) |
| `negative_prompt` | String | ❌ | Elementi da evitare |
| `image_1` | IMAGE | ❌ | Immagine input 1 (edit/img2img) |
| `image_2` | IMAGE | ❌ | Immagine input 2 |
| `image_3` | IMAGE | ❌ | Immagine input 3 |
| `image_4` | IMAGE | ❌ | Immagine input 4 |

| Output | Tipo | Descrizione |
|--------|------|-------------|
| `image` | IMAGE | Immagine generata (tensore ComfyUI) |

## Workflow di esempio

```
[SiliconFlow Model Selector] → model
                                    ↘
[Load Image (opz.)] → image_1   [SiliconFlow Inference] → image → [Preview Image]
                                    ↗
                 prompt, width, height, seed...
```

## Note

- La **lista modelli** viene cachata per 5 minuti per ridurre le chiamate API.
- Con `random_seed = True`, ogni esecuzione produce un risultato diverso.
- Le immagini input (image_1–4) sono **opzionali**: se omesse, il nodo funziona in modalità text-to-image pura.
- La compatibilità con modelli edit/img2img dipende dal supporto del modello specifico su SiliconFlow.

## Requisiti

- ComfyUI (qualsiasi versione recente)
- Python 3.8+
- `Pillow` (PIL) — già incluso in ComfyUI
- Nessuna dipendenza esterna aggiuntiva
