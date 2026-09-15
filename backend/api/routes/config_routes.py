"""
config_routes.py — Rutas de configuración general (config.json).

Es la MISMA configuración que la TUI: los cambios hechos desde la Web UI
los ve la TUI en su próximo arranque (y viceversa).
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any

from core.config.config_manager import get_config_manager

router = APIRouter()


@router.get("/config")
async def get_config():
    """Devuelve la configuración completa (misma fuente que la TUI)."""
    return {"config": get_config_manager().get_all()}


class ConfigSetRequest(BaseModel):
    """Body para POST /api/config (key+value, o varios en 'values')."""
    key: str | None = None
    value: Any = None
    values: dict[str, Any] | None = None


@router.post("/config")
async def set_config(req: ConfigSetRequest):
    """
    Setea una clave (key/value) o varias (values: {clave: valor}).
    Siempre valida. Devuelve {ok, results: {clave: bool}}.
    """
    cfg = get_config_manager()

    if req.values is not None:
        results = cfg.set_multiple(req.values, validate=True)
    elif req.key is not None:
        results = {req.key: cfg.set(req.key, req.value, validate=True)}
    else:
        return JSONResponse(
            {"detail": "Enviá 'key'+'value' o 'values' ({clave: valor})"},
            status_code=400,
        )

    return {"ok": all(results.values()), "results": results}
