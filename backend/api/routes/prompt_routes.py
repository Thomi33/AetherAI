"""
prompt_routes.py — Rutas del system prompt editable (sin tocar código).

Endpoints:
- GET  /api/system-prompt          → estado actual (3 claves)
- POST /api/system-prompt          → setea override / extra / sintesis_extra
- GET  /api/system-prompt/preview  → prompts finales que ve el modelo ahora

Semántica (core/config/config_manager.py + core/agent/prompts.py):
- override: reemplaza la persona por defecto (la memoria se sigue agregando)
- extra: se agrega a TODOS los prompts con máxima prioridad
- sintesis_extra: solo para la persona de síntesis
"""

from fastapi import APIRouter
from pydantic import BaseModel

from core.config.config_manager import get_config_manager

router = APIRouter()

# campo del body  →  clave de config.json
_CLAVES = {
    "override": "SYSTEM_PROMPT_OVERRIDE",
    "extra": "SYSTEM_PROMPT_EXTRA",
    "sintesis_extra": "SYSTEM_PROMPT_SINTESIS_EXTRA",
}


def _estado_actual() -> dict:
    cfg = get_config_manager()
    return {
        "override": cfg.get("SYSTEM_PROMPT_OVERRIDE", ""),
        "extra": cfg.get("SYSTEM_PROMPT_EXTRA", ""),
        "sintesis_extra": cfg.get("SYSTEM_PROMPT_SINTESIS_EXTRA", ""),
    }


@router.get("/system-prompt")
async def get_system_prompt():
    """Estado actual del system prompt configurable."""
    estado = _estado_actual()
    estado["ok"] = True
    return estado


class PromptUpdate(BaseModel):
    """Body para POST /api/system-prompt. Solo los campos enviados cambian."""
    override: str | None = None
    extra: str | None = None
    sintesis_extra: str | None = None


@router.post("/system-prompt")
async def set_system_prompt(upd: PromptUpdate):
    """Setea override/extra/sintesis_extra. Vacío = deshabilitado."""
    cfg = get_config_manager()
    results: dict[str, bool] = {}

    for campo, clave in _CLAVES.items():
        valor = getattr(upd, campo)
        if valor is not None:
            results[campo] = cfg.set(clave, valor, validate=True)

    if not results:
        return {"ok": False, "detail": "No se envió ningún campo", **_estado_actual()}

    return {"ok": all(results.values()), "results": results, **_estado_actual()}


@router.get("/system-prompt/preview")
async def preview_system_prompt():
    """Prompts finales que verá el modelo en su próximo turno."""
    from core.agent.prompts import construir_backstory, construir_persona_sintesis

    return {
        "ok": True,
        "backstory": construir_backstory(""),
        "persona_sintesis": construir_persona_sintesis(""),
    }
