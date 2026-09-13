"""Estados reales y mensajes efímeros de personalidad de la TUI."""

from __future__ import annotations

import random

REAL_STATES = {
    "planner": "[•] Analizando solicitud...",
    "context_manager": "[•] Preparando contexto...",
    "plan_executor": "[•] Ejecutando plan...",
    "agent_loop": "[•] Razonando qué herramienta usar...",
    "shell": "[•] Ejecutando herramienta: shell",
    "fs_read": "[•] Leyendo archivos...",
    "fs_write": "[•] Escribiendo archivos...",
    "fs_mkdir": "[•] Creando carpetas...",
    "fs_list": "[•] Listando archivos...",
    "file_write": "[•] Escribiendo archivos...",
    "codigo": "[•] Escribiendo archivos...",
    "web": "[•] Consultando la web...",
    "vision": "[•] Analizando pantalla...",
    "computer_use": "[•] Controlando la interfaz...",
    "node_plan_synthesizer": "[•] Verificando resultado...",
    "node_finalize": "[✓] Operación completada",
}

FLAVOR_MESSAGES = (
    "Laburando...",
    "Cocinando...",
    "Haciendo lo que Claude puede pero local y charrúa 🇺🇾",
    "domando Python...",
    "haciendo la de Linux...",
    "procesando neuronas imaginarias...",
    "moviendo bits...",
    "fa, esto pinta bien",
    "modo ñoqui activado",
    "Aether.exe está laburando...",
    "compilando las ganas de vivir",
    "haciendo magia con bits...",
    "pensando... supuestamente",
)
RARE_FLAVOR_MESSAGES = ("six seven...",)

STATUS_CONFIG = {
    "flavor_enabled": True,
    "flavor_probability": 0.18,
    "rare_flavor_probability": 0.025,
    "flavor_interval_seconds": 8.0,
    "min_operation_seconds": 5.0,
}


def real_state_for_node(node: str, delta: dict | None = None) -> str | None:
    """Devuelve un estado verídico para un nodo del grafo, si existe."""
    key = node.rsplit(".", 1)[-1]
    if key == "context_manager" and isinstance(delta, dict):
        if delta.get("context_compactado"):
            return "aptretando contexto pa salvar tu VRAM..."
    if isinstance(delta, dict):
        tool = delta.get("tool_actual") or delta.get("herramienta")
        if tool in REAL_STATES:
            return REAL_STATES[tool]
    if key in REAL_STATES:
        return REAL_STATES[key]
    return None


def choose_flavor() -> str | None:
    """Selecciona flavor sin afectar la ejecución ni aparecer constantemente."""
    if not STATUS_CONFIG["flavor_enabled"]:
        return None
    roll = random.random()
    if roll <= STATUS_CONFIG["rare_flavor_probability"]:
        return random.choice(RARE_FLAVOR_MESSAGES)
    if roll > STATUS_CONFIG["flavor_probability"]:
        return None
    return random.choice(FLAVOR_MESSAGES)
