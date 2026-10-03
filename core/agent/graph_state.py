from typing import Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage

MAX_INTENTOS_DEFAULT = 3

MAX_INTENTOS_POR_CONTEXTO = {
    "launch": 5,
    "shell":  3,
    "codigo": 3,
}

class AetherState(TypedDict):
    # ── Entrada ──────────────────────────────────────────────────────
    orden:          str
    mem:            dict[str, Any]
    modo_autonomo:  bool

    # ── Control de flujo ─────────────────────────────────────────────
    done:           bool          # True → el grafo debe terminar

    # ── Tool Planning (legacy: fast-paths deterministas de node_planner) ─
    plan_activo:    bool          # True si hay un plan multi-tool en ejecución
    plan_pasos:     list[dict]    # [{"tool": "web", "query": "..."}, {"tool": "shell", ...}]
    plan_index:     int           # índice del paso actual
    plan_resultados: list[str]    # resultados de cada paso ejecutado

    # ── Agent Loop (tool calling nativo, sin plan armado de antemano) ────
    agent_activo:    bool          # True mientras el agent loop sigue corriendo
    agent_messages:  list          # transcript formato Ollama (system/user/assistant/tool)
    agent_pasos_log: list[dict]    # [{"tool":..., "args":..., "resultado":...}, ...] para debug/síntesis
    agent_images:    list[str]     # thumbnails b64 de adjuntos del turno → van
                                   # incrustados en el mensaje del usuario del
                                   # agent loop / text (visión nativa del modelo)

    # ── Mensajes LangChain ───────────────────────────────────────────
    messages:       list[BaseMessage]

    # ── Respuestas ───────────────────────────────────────────────────
    llm_response:   str | None    # respuesta cruda del LLM
    final_response: str | None    # respuesta final al usuario

    # ── Shell / ejecución ────────────────────────────────────────────
    shell_command:  str | None    # comando que se ejecutó
    shell_output:   str           # stdout+stderr del comando
    shell_error:    bool          # True si hubo error de ejecución

    # ── Web ──────────────────────────────────────────────────────────
    web_results:    str           # resultados de buscar_web()

    # ── MCP ──────────────────────────────────────────────────────────
    mcp_result:     str           # resultado crudo devuelto por node_mcp

    # ── Visión ───────────────────────────────────────────────────────
    vision_result:  str           # descripción retornada por ver_pantalla()

    # ── FileSystemTool (fs_write/fs_read/fs_mkdir/fs_list) ──────────────
    fs_result:      str           # dato crudo devuelto por los nodos fs_*
    _tool_args:     dict          # args estructurados de la tool call actual
                                   # (seteado por node_agent_loop; ver
                                   # _args_del_paso en graph_nodes.py)

    # ── Computer use (loop percepción-acción) ───────────────────────
    computer_use_log:    list[dict]  # pasos ejecutados [{paso, accion, error}, ...]
    computer_use_result: str         # resumen en texto para plan_synthesizer

    # ── Context Manager ──────────────────────────────────────────────
    sesion_id:        str           # identificador único de sesión (UUID)
    context_slots:    dict          # {"tema": "...", "contexto": "...", ...}
    context_dump:     str           # log legible de qué se inyectó
    tema_actual:      str           # tool/tema detectado (ej. "web", "shell", etc.)
    historial_filtrado: list        # turnos relevantes (para debug)

    # ── Código ───────────────────────────────────────────────────────
    _codigo_original: str         # código generado antes del primer error
    _archivo_codigo:  str         # ruta del archivo temporal (.py/.sh)

    # ── Error handler ────────────────────────────────────────────────
    error_activo:          bool
    error_contexto:        str    # "shell" | "launch" | "codigo"
    error_mensaje:         str
    error_cmd:             str    # campo legacy (usar shell_command)
    error_codigo_original: str    # campo legacy (usar _codigo_original)
    error_archivo:         str    # campo legacy (usar _archivo_codigo)
    error_intento:         int
    error_max_intentos:    int
    error_autorizado:      bool

    # Fix propuesto por node_error_diagnose
    fix_propuesto:   str          # comando o código corregido
    fix_diff:        str          # diff unificado (solo contexto "codigo")
    fix_fuente_web:  str          # URL fuente del fix

    # Aliases legacy mantenidos por compatibilidad con nodos anteriores
    error_fix_propuesto: str
    error_fix_diff:      str
    error_fix_fuente:    str


# ══════════════════════════════════════════════════════════════════════
# FACTORY DE ESTADO INICIAL
# ══════════════════════════════════════════════════════════════════════

def crear_estado_inicial(orden: str, mem: dict, modo_autonomo: bool = True,
                         imagenes: list[str] | None = None) -> "AetherState":
    """
    Construye un AetherState completo y coherente para invocar el grafo.

    Garantiza que TODOS los campos existan con defaults seguros (incluidos
    los campos de tool planning), evitando KeyError en nodos. La memoria se
    normaliza con normalizar_mem() para cumplir el esquema obligatorio.

    Es la única fuente de verdad para inicializar el estado del grafo; tanto
    el CLI/servicio como los tests deben usar esta factory en vez de armar
    el dict a mano.

    imagenes: thumbnails b64 de adjuntos de imagen (agente con visión
    inline). Vacío/None → comportamiento texto-only clásico.
    """
    # Import local para evitar import circular (memory_manager → no depende de state)
    from core.memory.memory_manager import normalizar_mem

    mem_norm = normalizar_mem(mem)

    return {
        # ── Entrada ──────────────────────────────────────────────────
        "orden":         orden,
        "mem":           mem_norm,
        "modo_autonomo": modo_autonomo,

        # ── Control de flujo ─────────────────────────────────────────
        "done":          False,

        # ── Tool Planning ────────────────────────────────────────────
        "plan_activo":     False,
        "plan_pasos":      [],
        "plan_index":      0,
        "plan_resultados": [],

        "agent_activo":    False,
        "agent_messages":  [],
        "agent_pasos_log": [],
        "agent_images":    list(imagenes or []),

        # ── Mensajes ─────────────────────────────────────────────────
        "messages":      [],

        # ── Respuestas ───────────────────────────────────────────────
        "llm_response":   None,
        "final_response": None,

        # ── Shell ────────────────────────────────────────────────────
        "shell_command": None,
        "shell_output":  "",
        "shell_error":   False,

        # ── Web / MCP / Visión / Código ──────────────────────────────
        "web_results":      "",
        "mcp_result":       "",
        "vision_result":    "",
        "fs_result":        "",
        "_tool_args":       {},
        "computer_use_log":    [],
        "computer_use_result": "",
        "_codigo_original": "",
        "_archivo_codigo":  "",

        # ── Context Manager ──────────────────────────────────────────
        "sesion_id":        mem_norm.get("sesion_id", ""),
        "context_slots":    {},
        "context_dump":     "",
        "tema_actual":      "",
        "historial_filtrado": [],

        # ── Error handler ────────────────────────────────────────────
        "error_activo":          False,
        "error_contexto":        "",
        "error_mensaje":         "",
        "error_cmd":             "",
        "error_codigo_original": "",
        "error_archivo":         "",
        "error_intento":         0,
        "error_max_intentos":    MAX_INTENTOS_DEFAULT,
        "error_autorizado":      False,
        "fix_propuesto":         "",
        "fix_diff":              "",
        "fix_fuente_web":        "",
        "error_fix_propuesto":   "",
        "error_fix_diff":        "",
        "error_fix_fuente":      "",
    }


def _claves_estado() -> set[str]:
    """Retorna el conjunto de claves esperadas en AetherState (para tests/validación)."""
    return set(AetherState.__annotations__.keys())