"""
prompt_routes.py — Capa de comportamiento personalizable del system prompt.

Arquitectura de capas (core/agent/prompts.py): las instrucciones internas
del agente (tools/shell/planning/protocolos) y el comportamiento
predeterminado de Aether viven en el código y NINGUNA de estas claves los
reemplaza. El usuario solo configura la capa USER_CUSTOM_BEHAVIOR.

Endpoints:
- GET  /api/system-prompt          → estado actual (behavior + claves legadas)
- POST /api/system-prompt          → setea behavior (+ legadas extra/sintesis)
- GET  /api/system-prompt/preview  → prompts finales que ve el modelo ahora

Semántica:
- behavior: la casilla principal — personalización conversacional
  (personalidad, tono, estilo). NUNCA toca el funcionamiento interno.
- extra: legado, otra capa de comportamiento general.
- sintesis_extra: extra solo para la persona de síntesis.
- override (LEGADO): antes reemplazaba TODO el prompt; ahora se trata como
  una capa más de comportamiento. Ya no puede reemplazar nada.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from core.config.config_manager import get_config_manager

router = APIRouter()

# campo del body  →  clave de config.json
_CLAVES = {
    "behavior": "SYSTEM_PROMPT_BEHAVIOR",       # casilla principal
    "override": "SYSTEM_PROMPT_OVERRIDE",       # legado: ya no reemplaza nada
    "extra": "SYSTEM_PROMPT_EXTRA",             # legado
    "sintesis_extra": "SYSTEM_PROMPT_SINTESIS_EXTRA",
}


def _estado_actual() -> dict:
    cfg = get_config_manager()
    return {
        "behavior": cfg.get("SYSTEM_PROMPT_BEHAVIOR", ""),
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
    behavior: str | None = None
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
    """Prompts finales que verá el modelo en su próximo turno (los 3 reales)."""
    from core.agent.prompts import (
        construir_backstory,
        construir_persona_sintesis,
        construir_prompt_agent_loop,
    )

    return {
        "ok": True,
        "backstory": construir_backstory(""),
        "persona_sintesis": construir_persona_sintesis(""),
        "agent_loop": construir_prompt_agent_loop(""),
    }
