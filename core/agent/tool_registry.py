"""
Registro central de herramientas (tools) del grafo de Aether y validación de planes.

CONTRATO DE PLAN
────────────────
Un "plan" es una lista de "pasos". Cada paso es un dict con la forma:

    {
        "tool": "web",                 # obligatorio, debe existir en TOOL_REGISTRY
        "instruccion": "buscar ...",   # obligatorio, str no vacío (lo que debe hacer el paso)
        "args": {"query": "..."}        # opcional, dict con args específicos de la tool
    }

El executor (node_plan_executor) reutiliza los nodos reales del grafo: cada
tool mapea a una función de nodo (resuelta de forma lazy para evitar imports
circulares con graph_nodes).

DISEÑO
──────
- TOOL_REGISTRY es la única fuente de verdad de qué tools existen y qué
  función de nodo las ejecuta.
- validar_plan() NO importa graph_nodes: solo valida estructura y nombres,
  por lo que puede testearse de forma aislada y barata.
- get_node_func() hace el import lazy de graph_nodes cuando realmente se
  va a ejecutar un paso.
"""

from __future__ import annotations

from typing import Any, Callable


# Aliases legacy con guion (algunos prompts/modelos emiten fs-write, etc.)
_ALIASES_TOOL: dict[str, str] = {
    "fs-write": "fs_write",
    "fs-read": "fs_read",
    "fs-mkdir": "fs_mkdir",
    "fs-list": "fs_list",
}


# ══════════════════════════════════════════════════════════════════════
# REGISTRO DE TOOLS
# ══════════════════════════════════════════════════════════════════════
# "node" = nombre de la función de nodo en core.agent.graph_nodes
# "instruccion_requerida" = si el paso debe traer una instrucción no vacía
# "descripcion" = ayuda para el planner (prompt) y debugging
TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "text": {
        "node": "node_text",
        "instruccion_requerida": True,
        "descripcion": "Responder directamente o conversar usando el LLM.",
    },
    "web": {
        "node": "node_web",
        "instruccion_requerida": True,
        "descripcion": "Buscar información en internet y sintetizarla.",
    },
    "shell": {
        "node": "node_shell",
        "instruccion_requerida": True,
        "descripcion": "Generar y ejecutar un comando de sistema (shell).",
    },
    "launch": {
        "node": "node_launch",
        "instruccion_requerida": True,
        "descripcion": "Abrir/lanzar una aplicación (flatpak o binario en PATH).",
    },
    "vision": {
        "node": "node_vision",
        "instruccion_requerida": False,  # puede capturar pantalla sin instrucción explícita
        "descripcion": "Capturar la pantalla y analizar su contenido.",
    },
    "codigo": {
        "node": "node_codigo",
        "instruccion_requerida": True,
        "descripcion": "Generar código (python/bash/java), guardarlo en un archivo si el usuario da un nombre/ruta (ej. main.py, script.sh), y ejecutarlo.",
    },
    # ── Filesystem de primera clase (ver core/tools/filesystem_tool.py) ──
    # El agent loop los llama con args estructurados (path/content/files);
    # sin esto validar_tool_call() los rechaza como "desconocida".
    "fs_write": {
        "node": "node_fs_write",
        "instruccion_requerida": False,  # path+content van en args, no en texto
        "descripcion": "Escribir uno o varios archivos con ruta+contenido explícitos ({path, content} o {files: [{path, content}]}).",
    },
    "fs_read": {
        "node": "node_fs_read",
        "instruccion_requerida": False,
        "descripcion": "Leer un archivo de texto dado su path.",
    },
    "fs_mkdir": {
        "node": "node_fs_mkdir",
        "instruccion_requerida": False,
        "descripcion": "Crear un directorio (y sus padres) dado su path.",
    },
    "fs_list": {
        "node": "node_fs_list",
        "instruccion_requerida": False,
        "descripcion": "Listar el contenido (no recursivo) de un directorio.",
    },
    "memory": {
        "node": "node_memory",
        "instruccion_requerida": True,
        "descripcion": "Gestionar la memoria del agente (ver, borrar, recordar).",
    },
    "file_write": {
    "node": "node_file_write",
    "instruccion_requerida": False,  # puede inferir contenido de plan_resultados
    "descripcion": "Guardar el resultado de un paso anterior (o texto dado) en un archivo.",
     },
    "extract": {
        "node": "node_extract",
        "instruccion_requerida": False,  # opera sobre plan_resultados del paso previo
        "descripcion": (
            "Limpiar/extraer el contenido pedido por el usuario a partir del "
            "resultado crudo de un paso anterior (descarta metadatos de "
            "búsqueda, HTML sin decodificar y relleno conversacional)."
        ),
    },
    "mcp": {
        "node": "node_mcp",
        "instruccion_requerida": True,
        "descripcion": (
            "Invocar una tool de un servidor MCP externo conectado. "
            "El paso debe traer args={'server': <nombre>, 'name': <tool>, "
            "'arguments': {...}} identificando qué servidor y qué tool MCP usar."
        ),
    },
    "computer_use": {
        "node": "node_computer_use",
        "instruccion_requerida": True,
        "descripcion": (
            "Loop de percepción-acción: captura pantalla, decide una acción de "
            "mouse/teclado con el modelo de visión, la ejecuta, repite hasta "
            "cumplir el objetivo o alcanzar el límite de pasos."
        ),
    },
    # ── FileSystemTool (auditoría 2026-09-06, punto 3) ──────────────────
    # Operaciones explícitas de archivo/directorio con path+content
    # estructurados, para el agent loop. Ver core/tools/filesystem_tool.py
    # para el motivo (node_codigo solo guarda el primer bloque de código;
    # file_write legacy adivina la ruta por regex sobre texto libre).
    "fs_write": {
        "node": "node_fs_write",
        "instruccion_requerida": False,
        "descripcion": (
            "Escribir uno o varios archivos con ruta y contenido explícitos. "
            "Usar 'files' (lista de {path, content}) para proyectos de varios "
            "archivos: se escriben todos o ninguno (si uno falla, se revierten "
            "los ya escritos). Para un solo archivo alcanza con 'path'+'content'."
        ),
    },
    "fs_read": {
        "node": "node_fs_read",
        "instruccion_requerida": False,
        "descripcion": "Leer el contenido de un archivo de texto existente dado su path.",
    },
    "fs_mkdir": {
        "node": "node_fs_mkdir",
        "instruccion_requerida": False,
        "descripcion": "Crear un directorio (y sus padres si hacen falta) en el path dado.",
    },
    "fs_list": {
        "node": "node_fs_list",
        "instruccion_requerida": False,
        "descripcion": "Listar el contenido (archivos y subdirectorios) de un directorio dado su path.",
    },
}

# Conjunto de nombres de tools válidos (conveniencia)
TOOLS_VALIDAS: frozenset[str] = frozenset(TOOL_REGISTRY.keys())


# ══════════════════════════════════════════════════════════════════════
# SCHEMAS PARA TOOL CALLING NATIVO (AGENT LOOP)
# ══════════════════════════════════════════════════════════════════════
# JSON-schema de los argumentos de cada tool, en el formato que espera
# Ollama/OpenAI function calling ("parameters"). Es la contraparte
# estructurada de "descripcion": antes solo servía de texto para el
# planner viejo, ahora es lo que el modelo realmente usa para decidir CON
# QUÉ ARGUMENTOS llamar a una tool durante el agent loop.
TOOL_PARAMETROS: dict[str, dict] = {
    "text": {
        "type": "object",
        "properties": {
            "instruccion": {"type": "string", "description": "Qué responder o de qué conversar."},
        },
        "required": ["instruccion"],
    },
    "web": {
        "type": "object",
        "properties": {
            "instruccion": {"type": "string", "description": "Qué buscar en internet."},
        },
        "required": ["instruccion"],
    },
    "shell": {
        "type": "object",
        "properties": {
            "instruccion": {"type": "string", "description": "Qué comando de sistema ejecutar o lograr."},
        },
        "required": ["instruccion"],
    },
    "launch": {
        "type": "object",
        "properties": {
            "instruccion": {"type": "string", "description": "Qué aplicación abrir/lanzar."},
        },
        "required": ["instruccion"],
    },
    "vision": {
        "type": "object",
        "properties": {
            "instruccion": {"type": "string", "description": "Qué mirar/analizar en la pantalla (opcional)."},
        },
        "required": [],
    },
    "codigo": {
        "type": "object",
        "properties": {
            "instruccion": {"type": "string", "description": "Qué código generar y ejecutar."},
        },
        "required": ["instruccion"],
    },
    "memory": {
        "type": "object",
        "properties": {
            "instruccion": {"type": "string", "description": "Qué operación de memoria realizar (ver, borrar, recordar)."},
        },
        "required": ["instruccion"],
    },
    "file_write": {
        "type": "object",
        "properties": {
            "filename": {"type": "string", "description": "Nombre del archivo a escribir."},
            "instruccion": {"type": "string", "description": "Contenido a guardar, si no es el resultado del paso anterior (opcional)."},
        },
        "required": [],
    },
    "extract": {
        "type": "object",
        "properties": {
            "instruccion": {"type": "string", "description": "Qué extraer del resultado del paso anterior (opcional)."},
        },
        "required": [],
    },
    "mcp": {
        "type": "object",
        "properties": {
            "server": {"type": "string", "description": "Nombre del servidor MCP conectado a usar."},
            "name": {"type": "string", "description": "Nombre de la tool MCP a invocar en ese servidor."},
            "arguments": {"type": "object", "description": "Argumentos para la tool MCP (según su propio schema)."},
        },
        "required": ["server", "name"],
    },
    "computer_use": {
        "type": "object",
        "properties": {
            "instruccion": {"type": "string", "description": "Objetivo a lograr controlando mouse/teclado en pantalla."},
        },
        "required": ["instruccion"],
    },
    "fs_write": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Ruta del archivo a escribir (para un solo archivo)."},
            "content": {"type": "string", "description": "Contenido a escribir en 'path' (para un solo archivo)."},
            "files": {
                "type": "array",
                "description": "Lista de {path, content} para escribir varios archivos de forma atómica (proyectos multi-archivo). Si se usa 'files', no hace falta 'path'/'content' sueltos.",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
        "required": [],
    },
    "fs_read": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Ruta del archivo a leer."},
        },
        "required": ["path"],
    },
    "fs_mkdir": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Ruta del directorio a crear."},
        },
        "required": ["path"],
    },
    "fs_list": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Ruta del directorio a listar."},
        },
        "required": ["path"],
    },
}


def construir_tools_ollama() -> list[dict]:
    """
    Arma la lista de tools en formato nativo de Ollama (function calling)
    a partir de TOOL_REGISTRY + TOOL_PARAMETROS.

    Esto es lo que reemplaza al planner de un solo shot: en vez de pedirle
    al modelo un plan JSON completo de antemano, se le pasa el catálogo
    completo de tools (con su descripción y schema de argumentos) en cada
    turno del agent loop, y el modelo elige él mismo -- con razonamiento
    real, no keywords -- qué tool usar, con qué argumentos, o si ya puede
    responder directamente.
    """
    tools = []
    for nombre, meta in TOOL_REGISTRY.items():
        parametros = TOOL_PARAMETROS.get(nombre, {"type": "object", "properties": {}})
        tools.append({
            "type": "function",
            "function": {
                "name": nombre,
                "description": meta["descripcion"],
                "parameters": parametros,
            },
        })
    return tools


def validar_tool_call(tool: str, args: dict) -> tuple[bool, str]:
    """
    Sanity-check de UNA tool call propuesta por el modelo durante el agent
    loop (equivalente de validar_plan(), pero para una llamada individual
    de tool calling incremental en vez de un plan completo armado de
    antemano).

    Chequea lo barato y evidente -- la tool existe, args es un dict, trae
    instrucción si la tool la requiere, y (para mcp) trae server+name --
    antes de gastar una ejecución real con una llamada alucinada. No
    reemplaza la validación de tipos fina del JSON-schema; es un freno
    rápido, no un validador exhaustivo.

    Retorna (ok, motivo). motivo es "" si ok=True.
    """
    tool = _ALIASES_TOOL.get(tool, tool)
    if tool not in TOOL_REGISTRY:
        return False, f"tool '{tool}' no existe en el registro."
    if not isinstance(args, dict):
        return False, "'arguments' debe ser un objeto/dict."

    if TOOL_REGISTRY[tool]["instruccion_requerida"]:
        if not _instruccion_de_paso({"args": args}):
            return False, f"la tool '{tool}' requiere una instrucción no vacía."

    if tool == "mcp":
        if not args.get("server") or not args.get("name"):
            return False, "la tool 'mcp' requiere 'server' y 'name'."

    return True, ""


# ══════════════════════════════════════════════════════════════════════
# VALIDACIÓN DE PLANES
# ══════════════════════════════════════════════════════════════════════

def _instruccion_de_paso(paso: dict) -> str:
    """
    Extrae la instrucción de un paso, aceptando varias ubicaciones por
    compatibilidad: top-level 'instruccion', o dentro de 'args'
    (instruccion/query/command/app).
    """
    if not isinstance(paso, dict):
        return ""
    if isinstance(paso.get("instruccion"), str) and paso["instruccion"].strip():
        return paso["instruccion"].strip()
    args = paso.get("args")
    if isinstance(args, dict):
        for clave in ("instruccion", "query", "command", "app", "orden"):
            val = args.get(clave)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


def validar_plan(plan: Any) -> tuple[bool, list[str]]:
    """
    Valida la estructura de un plan antes de ejecutarlo.

    Reglas:
    - plan debe ser una lista no vacía
    - cada paso debe ser dict
    - cada paso debe tener 'tool' existente en TOOL_REGISTRY
    - si la tool requiere instrucción, el paso debe traer una no vacía
    - 'args' (si está) debe ser dict

    Retorna (ok, errores). ok=True solo si errores está vacío.
    """
    errores: list[str] = []

    if not isinstance(plan, list):
        return False, ["El plan debe ser una lista de pasos."]
    if len(plan) == 0:
        return False, ["El plan está vacío."]

    for i, paso in enumerate(plan):
        prefijo = f"paso {i + 1}"

        if not isinstance(paso, dict):
            errores.append(f"{prefijo}: no es un objeto/dict válido.")
            continue

        tool = paso.get("tool")
        if isinstance(tool, str):
            tool = _ALIASES_TOOL.get(tool, tool)
        if not tool:
            errores.append(f"{prefijo}: falta el campo 'tool'.")
            continue
        if tool not in TOOL_REGISTRY:
            errores.append(f"{prefijo}: tool '{tool}' no existe en el registro.")
            continue

        args = paso.get("args", {})
        if args is not None and not isinstance(args, dict):
            errores.append(f"{prefijo}: 'args' debe ser un dict.")

        if TOOL_REGISTRY[tool]["instruccion_requerida"]:
            if not _instruccion_de_paso(paso):
                errores.append(
                    f"{prefijo}: la tool '{tool}' requiere una instrucción no vacía."
                )

    return (len(errores) == 0), errores


# ══════════════════════════════════════════════════════════════════════
# RESOLUCIÓN LAZY DE NODOS
# ══════════════════════════════════════════════════════════════════════

def get_node_func(tool: str) -> Callable:
    """
    Resuelve la función de nodo asociada a una tool.

    Import lazy de core.agent.graph_nodes para evitar import circular
    (graph_nodes importa este módulo para el executor).

    El mapeo fs_* legacy también acepta alias con guion (fs-write, etc.)
    porque algunos prompts/modelos los emiten así.
    """
    normalizada = _ALIASES_TOOL.get(tool, tool)
    if normalizada not in TOOL_REGISTRY:
        raise KeyError(f"Tool desconocida: {tool}")

    import core.agent.graph_nodes as gn

    nombre_func = TOOL_REGISTRY[normalizada]["node"]
    func = getattr(gn, nombre_func, None)
    if func is None or not callable(func):
        raise AttributeError(
            f"La función de nodo '{nombre_func}' para la tool '{normalizada}' no existe."
        )
    return func
