"""
models_routes.py — Selector de modelos real (vía Ollama, host de config.json).

- GET /api/models    → catálogo instalado (ollama list) + activo por config.
- GET /api/ollama/ps → modelos RESIDENTES en memoria (equivale a `ollama ps`).

  OJO: "activo" (/api/models) es el MODELO de config.json+config.local.json —
  el que el agente USARÁ en la próxima inferencia. /api/ollama/ps muestra lo
  que Ollama tiene cargado AHORA en RAM/VRAM; pueden diferir (otro proceso
  puede haber cargado otro modelo, o el activo puede no estar residente aún).
"""

from fastapi import APIRouter

import requests

from core.config.config_manager import get_config_manager

router = APIRouter()


@router.get("/models")
async def listar_modelos():
    """Lista los modelos instalados en Ollama y marca el activo."""
    cfg = get_config_manager()
    host = (cfg.get("OLLAMA_HOST", "") or "http://localhost:11434").rstrip("/")
    activo = cfg.get("MODELO", "")

    try:
        resp = requests.get(f"{host}/api/tags", timeout=10)
        resp.raise_for_status()
        modelos = [
            {"name": m.get("name", ""), "size": m.get("size"), "active": m.get("name") == activo}
            for m in resp.json().get("models", [])
        ]
        return {"ok": True, "host": host, "active": activo, "models": modelos}
    except Exception as e:  # noqa: BLE001 — la UI muestra el error
        return {"ok": False, "host": host, "active": activo, "models": [], "error": str(e)}


@router.get("/ollama/ps")
async def modelos_en_memoria():
    """Modelos realmente cargados en memoria de Ollama (`ollama ps`).

    Complementa a /api/models: acá se ve el estado REAL del runtime (qué hay
    residente, cuánto ocupa y cuándo expira), no la configuración.
    """
    cfg = get_config_manager()
    host = (cfg.get("OLLAMA_HOST", "") or "http://localhost:11434").rstrip("/")
    activo = cfg.get("MODELO", "")

    try:
        resp = requests.get(f"{host}/api/ps", timeout=10)
        resp.raise_for_status()
        modelos = [
            {
                "name": m.get("name", ""),
                "size": m.get("size"),
                "size_vram": m.get("size_vram"),
                "expires_at": m.get("expires_at"),
                "processor": m.get("processor", ""),
                "active": m.get("name") == activo,
            }
            for m in resp.json().get("models", [])
        ]
        return {"ok": True, "host": host, "active": activo, "models": modelos}
    except Exception as e:  # noqa: BLE001 — la UI muestra el error
        return {"ok": False, "host": host, "active": activo, "models": [], "error": str(e)}
