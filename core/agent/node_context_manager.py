"""
node_context_manager.py — Nodo LangGraph del Context Manager.

Responsabilidad única:
    Decidir QUÉ información entra al modelo y QUÉ se descarta,
    antes de cualquier inferencia.

Posición en el grafo:
    node_planner → node_context_manager → [node_text | node_shell | node_web | ...]

Qué hace:
    1. Lee el tema detectado por el planner (plan_pasos[0].tool)
    2. Actualiza core_memory con el tema y proyecto activo
    3. Construye los context_slots filtrando por relevancia
    4. Genera el context_dump para debug
    5. Propaga todo al estado para que los nodos downstream lo usen

Los nodos downstream NO construyen el prompt directamente:
    usan state["context_slots"] via _system_prompt().
"""

from core.agent.graph_state  import AetherState
from core.memory.context_builder import construir_contexto_memoria, construir_context_dump
from core.memory.memory_manager  import actualizar_core


def node_context_manager(state: AetherState) -> dict:
    """
    Ensambla el contexto relevante antes de cada inferencia.

    Lee el tema del plan activo, filtra la memoria por relevancia
    y genera el context_dump para facilitar el debug.
    """
    mem   = state.get("mem") or {}
    orden = state.get("orden", "")

    # ── Detectar tema desde el plan ───────────────────────────────────
    tema = _detectar_tema(state)

    # ── Actualizar core_memory con el tema actual ─────────────────────
    if tema:
        actualizar_core("tema_activo", tema)
        if isinstance(mem.get("core"), dict):
            mem["core"]["tema_activo"] = tema

    # ── Construir slots de contexto ───────────────────────────────────
    sesion_id = state.get("sesion_id", "")
    plan_pasos = state.get("plan_pasos") or []
    es_multitool = isinstance(plan_pasos, list) and len(plan_pasos) > 1
    contexto = construir_contexto_memoria(
        mem, tema=tema, sesion_id=sesion_id, es_multitool=es_multitool,
        orden=orden,
    )

    context_slots = {
        "tema":    tema,
        "orden":   orden,
        "contexto": contexto,
    }

    # El dump requiere consultas SQLite adicionales y solo sirve para depurar.
    # No debe pagar ese costo cada vez que se ejecuta una orden normal.
    from core.config.config_manager import get_config_manager
    if get_config_manager().get("DEBUG", False):
        dump = construir_context_dump(
            mem, tema=tema, sesion_id=sesion_id, es_multitool=es_multitool
        )
        print(f"\n{dump}")
    else:
        dump = ""

    return {
        "context_slots": context_slots,
        "context_dump":  dump,
        "tema_actual":   tema,
    }


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _detectar_tema(state: AetherState) -> str:
    """
    Extrae el tema/tool del primer paso del plan activo.
    Fallback: campo intent (legacy) o string vacío.
    """
    plan_pasos = state.get("plan_pasos") or []
    if plan_pasos and isinstance(plan_pasos[0], dict):
        tool = plan_pasos[0].get("tool", "")
        if tool:
            return str(tool)

    # Fallback legacy
    intent = state.get("intent", "")
    return str(intent) if intent else ""
