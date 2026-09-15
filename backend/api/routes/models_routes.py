"""
models_routes.py — Selector de modelos real (vía Ollama, host de config.json).
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
