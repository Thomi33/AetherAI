"""
runtime_routes.py — Effort y agente activo (como /effort y /agents de la TUI).

EFFORT
──────
Mismo mapeo que tui/app.py::_abrir_effort_selector: el nivel de esfuerzo se
traduce a TEMPERATURE + NUM_CTX + NUM_PREDICT en config.json, con efecto
inmediato (graph_nodes.py relee la config en cada turno). El nivel en sí se
guarda en la clave EFFORT para que la TUI y la Web UI muestren lo último
elegido (default "medium", igual que la TUI al arrancar).

AGENTE
──────
Igual que en la TUI: build|plan se guarda como preferencia (clave AGENTE) y
se muestra en el status. Hoy NO cambia el comportamiento del grafo (en la TUI
tampoco: es una etiqueta del status bar) — queda persistido para cuando el
motor lo consuma.
"""

import re

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

# Mismo mapa que tui/app.py (_abrir_effort_selector). temperature baja a medida
# que sube el effort; num_predict/num_ctx suben (presupuesto de razonamiento).
EFFORT_MAP = {
    "low":    {"TEMPERATURE": 0.9,  "NUM_CTX": 4096,  "NUM_PREDICT": 768},
    "medium": {"TEMPERATURE": 0.6,  "NUM_CTX": 8192,  "NUM_PREDICT": 2048},
    "high":   {"TEMPERATURE": 0.35, "NUM_CTX": 16384, "NUM_PREDICT": 4096},
    "max":    {"TEMPERATURE": 0.15, "NUM_CTX": 32768, "NUM_PREDICT": 8192},
}

# Mismos agentes que tui/widgets/selector_screens.py:AGENTS.
AGENTES = [
    {"name": "build", "kind": "native",
     "descripcion": "Agente completo: planifica, ejecuta tools y responde."},
    {"name": "plan",  "kind": "native",
     "descripcion": "Modo planificación (preferencia; misma semántica que la TUI)."},
]

# Custom agents: clave en config.local.json -> CUSTOM_AGENTS = {name: {...}}
def _cargar_custom_agents():
    from core.config.config_manager import get_config_manager
    cfg = get_config_manager()
    return cfg.get("CUSTOM_AGENTS", {}) or {}


def _guardar_custom_agents(custom):
    from core.config.config_manager import get_config_manager
    cfg = get_config_manager()
    return cfg.set("CUSTOM_AGENTS", custom, validate=True)


@router.get("/effort")
async def get_effort():
    from core.config.config_manager import get_config_manager

    cfg = get_config_manager()
    nivel = cfg.get("EFFORT", "medium")
    if nivel not in EFFORT_MAP:
        nivel = "medium"
    return {
        "ok": True,
        "effort": nivel,
        "niveles": list(EFFORT_MAP.keys()),
        "params": EFFORT_MAP[nivel],
    }


class EffortBody(BaseModel):
    effort: str


@router.post("/effort")
async def set_effort(body: EffortBody):
    from core.config.config_manager import get_config_manager

    nivel = body.effort.strip().lower()
    if nivel not in EFFORT_MAP:
        return JSONResponse(
            {"ok": False,
             "error": f"Effort inválido '{nivel}'. Opciones: "
                      f"{', '.join(EFFORT_MAP)}"},
            status_code=400,
        )

    cfg = get_config_manager()
    # Misma semántica que la TUI: aplica los 3 parámetros sin validación de
    # rango extra (los valores del mapa ya son seguros) y persiste el nivel.
    for clave, valor in EFFORT_MAP[nivel].items():
        cfg.set(clave, valor, validate=False)
    cfg.set("EFFORT", nivel, validate=True)
    return {"ok": True, "effort": nivel, "params": EFFORT_MAP[nivel]}


@router.get("/agent")
async def get_agent():
    from core.config.config_manager import get_config_manager

    actual = get_config_manager().get("AGENTE", "build")
    return {"ok": True, "agent": actual, "agents": AGENTES}


class AgentBody(BaseModel):
    agent: str


@router.post("/agent")
async def set_agent(body: AgentBody):
    from core.config.config_manager import get_config_manager

    nombre = body.agent.strip().lower()
    # Nativos + custom (sin esto, seleccionar un agente custom creado desde
    # la Web UI pegaba 400 y "no hacía nada" para el usuario).
    validos = [a["name"] for a in AGENTES] + list(_cargar_custom_agents().keys())
    if nombre not in validos:
        return JSONResponse(
            {"ok": False,
             "error": f"Agente inválido '{nombre}'. Opciones: {', '.join(validos)}"},
            status_code=400,
        )
    ok = get_config_manager().set("AGENTE", nombre, validate=True)
    return {"ok": ok, "agent": nombre}

@router.get("/agents")
async def list_agents():
    """Lista todos los agentes (nativos + custom)."""
    from core.config.config_manager import get_config_manager  # local, como el resto
    custom = _cargar_custom_agents()
    todos = AGENTES + [
        {
            "name": name,
            "kind": data.get("kind", "custom"),
            "descripcion": data.get("descripcion", "Agente personalizado"),
            "prompt": data.get("prompt", ""),
            "tools": data.get("tools", []),
            "modelo": data.get("modelo", ""),
        }
        for name, data in custom.items()
    ]
    actual = get_config_manager().get("AGENTE", "build")
    return {"ok": True, "agents": todos, "actual": actual}


class CustomAgentBody(BaseModel):
    name: str
    descripcion: str = ""
    prompt: str = ""
    tools: list[str] = []
    modelo: str = ""

@router.post("/agents")
async def crear_agente(body: CustomAgentBody):
    name = body.name.strip().lower()
    if not re.match(r"^[a-z][a-z0-9-]{0,63}$", name):
        return JSONResponse({"ok": False, "error": "Nombre inválido: minúsculas, números, guiones (ej. mi-agente)"}, status_code=400)
    validos_nativos = [a["name"] for a in AGENTES]
    if name in validos_nativos:
        return JSONResponse({"ok": False, "error": f"'{name}' es un agente nativo, no se puede sobrescribir"}, status_code=400)
    custom = _cargar_custom_agents()
    if name in custom:
        return JSONResponse({"ok": False, "error": f"Ya existe agente custom '{name}'"}, status_code=409)
    data = {
        "kind": "custom",
        "descripcion": body.descripcion.strip(),
        "prompt": body.prompt.strip(),
        "tools": body.tools,
        "modelo": body.modelo.strip(),
    }
    custom[name] = data
    ok = _guardar_custom_agents(custom)
    return {"ok": ok, "name": name}

@router.delete("/agents/{name}")
async def borrar_agente(name: str):
    name = name.strip().lower()
    validos_nativos = [a["name"] for a in AGENTES]
    if name in validos_nativos:
        return JSONResponse({"ok": False, "error": "No se puede borrar agente nativo"}, status_code=400)
    custom = _cargar_custom_agents()
    if name not in custom:
        return JSONResponse({"ok": False, "error": f"No existe '{name}'"}, status_code=404)
    del custom[name]
    ok = _guardar_custom_agents(custom)
    return {"ok": ok}
