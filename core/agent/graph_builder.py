r"""
Builder del grafo LangGraph para Aether.

Grafo principal (CON TOOL PLANNING + CONTEXT MANAGER)
──────────────────────────────────────────────────────
START → planner → context_manager → plan_executor → ... → finalize → END

El context_manager se ejecuta SIEMPRE después del planner y antes de cualquier
inferencia. Decide qué información entra al modelo y genera el context_dump
para debug.
"""

from langgraph.graph import StateGraph, END

from core.agent.graph_state import AetherState
from core.agent.graph_nodes import (
    node_planner,
    node_plan_executor,
    node_plan_synthesizer,
    node_agent_loop,
    node_web,
    node_shell,
    node_launch,
    node_vision,
    node_computer_use,
    node_codigo,
    node_text,
    node_memory,
    node_finalize,
    node_error_diagnose,
    node_error_confirm,
    node_error_retry,
    node_error_fallback,
)
from core.agent.node_context_manager import node_context_manager


# ══════════════════════════════════════════════════════════════════════
# CONDICIONALES
# ══════════════════════════════════════════════════════════════════════

# Herramientas con un campo dedicado donde dejan datos crudos para Ornith
# (ver _CAMPO_DATOS_CRUDOS_POR_TOOL). La TUI también los usa para mostrar
# actividad en vivo (qué archivo/carpeta tocó cada paso).
_TOOLS_CON_RESUMEN = frozenset({"shell", "codigo", "file_write", "web", "vision", "mcp", "computer_use", "launch", "fs_read", "fs_write", "fs_mkdir", "fs_list"})

# Campo de "datos crudos" que cada tool deja en el estado para que Ornith
# los interprete. Si ese campo está vacío pero la tool YA fijó una
# final_response propia (una decisión determinista: pedido de aclaración,
# mensaje de error legible, cancelación, etc.), no tiene sentido volver a
# pasar por el LLM de síntesis: Ornith no tiene datos nuevos que agregar y
# puede reformular/alucinar sobre una respuesta que ya estaba bien.
_CAMPO_DATOS_CRUDOS_POR_TOOL = {
    "web":          "web_results",
    "shell":        "shell_output",
    "codigo":       "shell_output",
    "vision":       "vision_result",
    "mcp":          "mcp_result",
    "computer_use": "computer_use_result",
    "fs_read":      "fs_result",
    "fs_write":     "fs_result",
    "fs_mkdir":     "fs_result",
    "fs_list":      "fs_result",
}


def _tiene_datos_crudos_para_sintetizar(state: AetherState, tool: str) -> bool:
    campo = _CAMPO_DATOS_CRUDOS_POR_TOOL.get(tool)
    if not campo:
        return False
    val = state.get(campo)
    return isinstance(val, str) and val.strip() != ""


def _destino_post_plan(state: AetherState) -> str:
    plan_pasos = state.get("plan_pasos")
    if not isinstance(plan_pasos, list) or len(plan_pasos) == 0:
        return "finalize"

    plan_index = state.get("plan_index", 0)
    if not isinstance(plan_index, int) or plan_index < 0:
        plan_index = 0

    if plan_index < len(plan_pasos):
        return "plan_executor"
    if len(plan_pasos) <= 1:
        paso = plan_pasos[0] if isinstance(plan_pasos[0], dict) else {}
        tool = paso.get("tool")
        if tool in _TOOLS_CON_RESUMEN:
            ya_tiene_respuesta_final = bool((state.get("final_response") or "").strip())
            hay_datos_crudos = _tiene_datos_crudos_para_sintetizar(state, tool)
            if ya_tiene_respuesta_final and not hay_datos_crudos and tool not in {"launch", "fs_write", "fs_mkdir", "fs_list"}:
                return "finalize"
            return "plan_synthesizer"
        return "finalize"
    return "plan_synthesizer"


def _destino_post_context(state: AetherState) -> str:
    """
    Después de context_manager: node_planner ya decidió qué camino tomar.
    Los fast-paths deterministas (charla simple, corrección del usuario,
    resolución anafórica) siguen armando un plan_pasos de 1 paso → van a
    plan_executor como antes. Todo lo demás activa agent_activo=True y va
    al agent loop (razonamiento + tool calling nativo, sin plan armado de
    antemano).
    """
    if state.get("agent_activo", False):
        return "agent_loop"
    return "plan_executor"


def _route_plan_executor(state: AetherState) -> str:
    if state.get("error_activo", False):
        return "error_diagnose"
    return _destino_post_plan(state)


def _route_agent_loop(state: AetherState) -> str:
    if state.get("error_activo", False):
        return "error_diagnose"
    if state.get("agent_activo", False) and not state.get("done", False):
        return "agent_loop"
    return "finalize"


def _route_after_confirm(state: AetherState) -> str:
    if state.get("done", False):
        return END
    return "error_retry"


def _route_after_retry(state: AetherState) -> str:
    if state.get("error_activo", False):
        return "error_diagnose"
    return _destino_post_plan(state)


def _route_after_diagnose(state: AetherState) -> str:
    intento_actual = state.get("error_intento", 0)
    max_intentos   = state.get("error_max_intentos", 3)
    if intento_actual >= max_intentos:
        print(f"\n⚠️  [ERROR HANDLER]: Límite de {max_intentos} intentos alcanzado. Ejecutando fallback...")
        return "error_fallback"
    if state.get("fix_propuesto", "").strip():
        return "error_confirm"
    return "error_fallback"


def _route_after_fallback(state: AetherState) -> str:
    if state.get("done", False):
        return END
    if state.get("error_activo", False):
        return "error_confirm"
    return _destino_post_plan(state)


# ══════════════════════════════════════════════════════════════════════
# BUILDER
# ══════════════════════════════════════════════════════════════════════

def build_graph():
    builder = StateGraph(AetherState)

    # ── Nodos ────────────────────────────────────────────────────────
    builder.add_node("planner",          node_planner)
    builder.add_node("context_manager",  node_context_manager)   # ← NUEVO
    builder.add_node("plan_executor",    node_plan_executor)
    builder.add_node("agent_loop",       node_agent_loop)         # ← NUEVO
    builder.add_node("plan_synthesizer", node_plan_synthesizer)
    builder.add_node("finalize",         node_finalize)

    builder.add_node("error_diagnose", node_error_diagnose)
    builder.add_node("error_confirm",  node_error_confirm)
    builder.add_node("error_retry",    node_error_retry)
    builder.add_node("error_fallback", node_error_fallback)

    # ── Edges ────────────────────────────────────────────────────────

    # START → planner → context_manager → (plan_executor | agent_loop)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "context_manager")

    # context_manager decide entre el camino viejo (fast-paths deterministas
    # de node_planner, que arman un plan_pasos de 1 paso) y el agent loop
    # (todo lo demás: razonamiento + tool calling nativo, sin plan armado de
    # antemano). Ver _destino_post_context / node_planner.
    builder.add_conditional_edges(
        "context_manager",
        _destino_post_context,
        {
            "plan_executor": "plan_executor",
            "agent_loop":    "agent_loop",
        },
    )

    # plan_executor → loop / error / synthesizer / finalize
    builder.add_conditional_edges(
        "plan_executor",
        _route_plan_executor,
        {
            "plan_executor":    "plan_executor",
            "error_diagnose":   "error_diagnose",
            "plan_synthesizer": "plan_synthesizer",
            "finalize":         "finalize",
        },
    )

    # agent_loop es un self-loop: cada ronda razona + ejecuta las tool
    # calls que haya y vuelve a entrar mientras agent_activo=True. Termina
    # yendo directo a finalize (node_agent_loop ya deja final_response
    # armado él mismo -- no necesita pasar por plan_synthesizer).
    builder.add_conditional_edges(
        "agent_loop",
        _route_agent_loop,
        {
            "agent_loop":     "agent_loop",
            "error_diagnose": "error_diagnose",
            "finalize":       "finalize",
        },
    )

    builder.add_conditional_edges("error_diagnose", _route_after_diagnose, {
        "error_confirm":  "error_confirm",
        "error_fallback": "error_fallback",
    })

    builder.add_conditional_edges("error_confirm", _route_after_confirm, {
        END:           END,
        "error_retry": "error_retry",
    })

    builder.add_conditional_edges("error_retry", _route_after_retry, {
        "error_diagnose":   "error_diagnose",
        "plan_executor":    "plan_executor",
        "plan_synthesizer": "plan_synthesizer",
        "finalize":         "finalize",
    })

    builder.add_conditional_edges("error_fallback", _route_after_fallback, {
        END:                END,
        "error_confirm":    "error_confirm",
        "plan_executor":    "plan_executor",
        "plan_synthesizer": "plan_synthesizer",
        "finalize":         "finalize",
    })

    builder.add_edge("plan_synthesizer", "finalize")
    builder.add_edge("finalize",         END)

    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def reset_graph():
    global _graph
    _graph = None
