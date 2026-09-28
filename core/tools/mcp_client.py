"""
Cliente MCP (Model Context Protocol) para Aether.

DISEÑO
──────
Todo el pipeline de Aether (graph_nodes.py, tool_registry.py) es síncrono.
El SDK oficial de MCP (paquete `mcp`) es 100% asyncio. En vez de mezclar
asyncio.run() por-llamada (que reconecta el server MCP en cada tool call,
con el overhead de levantar el subproceso stdio cada vez), este módulo:

  1. Levanta UN event loop de asyncio corriendo en un thread daemon aparte,
     vivo durante toda la sesión de Aether.
  2. Abre y mantiene conexiones persistentes a cada server MCP configurado
     (stdio_client + ClientSession), corriendo en ese loop.
  3. Expone una API SÍNCRONA (list_tools, call_tool, close) que despacha
     al loop vía asyncio.run_coroutine_threadsafe() y espera el resultado.

Esto es análogo a cómo la mayoría de los drivers de DB síncronos envuelven
un pool async por debajo: el resto del código de Aether no necesita saber
que hay asyncio corriendo adentro.

CONFIGURACIÓN
─────────────
Dos capas, mismo patrón que config.json / config.local.json:

  1. core/config/mcp_servers.json        → PLANTILLA versionada: servers
     conocidos con placeholders "{token}" (nunca tokens reales; el runtime
     jamás la escribe — el repo la distribuye como ejemplo editable).
  2. core/config/mcp_servers.local.json  → capa LOCAL ignorada por Git:
     tokens reales, servers custom, toggles de enabled y tombstones. TODO
     lo que escribe el runtime (Web UI /mcps, TUI /mcps) vive acá.

La vista que consume el manager es el MERGE por-server a nivel campo
(local pisa campo a campo; servers solo-locales se agregan; un server de
la plantilla con "__deleted__": true en local queda oculto):

    {
      "notion": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@notionhq/mcp-server"],
        "env": {"NOTION_API_KEY": "..."}   ← el token REAL va en el local
      }
    }

Por ahora solo se implementa transporte "stdio" (el más común para tools
locales). Si más adelante conectás un server remoto, se puede sumar
"streamable_http" siguiendo el mismo patrón (mcp.client.streamable_http).

USO TÍPICO (desde node_mcp)
────────────────────────────
    from core.tools.mcp_client import get_mcp_manager

    manager = get_mcp_manager()
    resultado = manager.call_tool("notion", "search", {"query": "..."})
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any


class MCPError(Exception):
    """Error genérico al operar con un server/tool MCP."""


class MCPServerNotFoundError(MCPError):
    """El server pedido no está en la config o no pudo conectarse."""


class MCPToolCallError(MCPError):
    """La llamada a la tool MCP falló (error del server o de la tool)."""


# ══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN — dos capas (mismo patrón que config.json/config.local.json)
# ══════════════════════════════════════════════════════════════════════
#   1. mcp_servers.json       → plantilla VERSIONADA: servers conocidos con
#      placeholders "{token}". El runtime NUNCA la escribe; el repo la
#      distribuye como ejemplo editable para que cada usuario ponga sus
#      tokens en la capa local.
#   2. mcp_servers.local.json → capa LOCAL ignorada por Git: tokens reales,
#      servers custom agregados desde la Web UI, toggles de enabled y
#      tombstones. TODO lo que escribe el runtime vive acá.
#
# Merge por-server a nivel CAMPO (como ConfigManager mezcla claves):
#   - server en ambas capas: local pisa campo a campo (env con el token
#     real, enabled, etc.); el resto de los campos los aporta la plantilla.
#   - server solo en local: se agrega (custom de la Web UI).
#   - server con "__deleted__": true en local: TOMBSTONE — la plantilla lo
#     define pero el usuario lo borró y NO aparece en la vista mergeada.

RUTA_BASE = Path(__file__).resolve().parent.parent / "config" / "mcp_servers.json"
RUTA_LOCAL = RUTA_BASE.with_name("mcp_servers.local.json")


def _ruta_config_default() -> Path:
    # Compat con llamadas viejas: la BASE es la plantilla versionada.
    return RUTA_BASE


def _leer_json_servers(ruta: Path) -> dict:
    """Lee un JSON de servers {nombre: cfg}; {} si falta o rompe (nunca lanza)."""
    if not ruta.exists():
        return {}
    try:
        with open(ruta, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  [MCP]: no se pudo leer {ruta}: {e}")
        return {}
    if not isinstance(data, dict):
        print(f"⚠️  [MCP]: {ruta} debe contener un objeto JSON {{server: config}}.")
        return {}
    return data


def cargar_local(ruta: Path | None = None) -> dict:
    """Capa local cruda: tokens reales, servers custom y tombstones."""
    return _leer_json_servers(ruta or RUTA_LOCAL)


def guardar_local(data: dict, ruta: Path | None = None) -> None:
    """Escritura atómica de la capa local. NUNCA toca la plantilla versionada."""
    destino = ruta or RUTA_LOCAL
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(destino)


def cargar_merge(ruta_base: Path | None = None,
                 ruta_local: Path | None = None) -> dict:
    """Vista combinada plantilla ← local (local gana campo a campo)."""
    base = _leer_json_servers(ruta_base or RUTA_BASE)
    local = _leer_json_servers(ruta_local or RUTA_LOCAL)

    merged: dict = {}
    for nombre, cfg in base.items():
        merged[nombre] = dict(cfg) if isinstance(cfg, dict) else cfg

    for nombre, cfg in local.items():
        if isinstance(cfg, dict) and cfg.get("__deleted__"):
            merged.pop(nombre, None)          # tombstone: borrado por el usuario
            continue
        if (nombre in merged and isinstance(cfg, dict)
                and isinstance(merged[nombre], dict)):
            merged[nombre] = {**merged[nombre], **cfg}   # override campo a campo
        else:
            merged[nombre] = cfg              # server custom (solo en local)
    return merged


def set_enabled(nombre: str, valor: bool) -> None:
    """Persiste un toggle de enabled en la capa LOCAL sin perder config.

    - Server de la plantilla: crea un override parcial {"enabled": valor};
      el resto de los campos los sigue aportando la plantilla.
    - Server custom (solo local): preserva su config completa y solo pisa
      "enabled" (nunca la reemplaza por un dict sin command/args/env).
    """
    local = cargar_local()
    cfg = local.get(nombre)
    if isinstance(cfg, dict) and not cfg.get("__deleted__"):
        local[nombre] = {**cfg, "enabled": bool(valor)}
    else:
        local[nombre] = {"enabled": bool(valor)}
    guardar_local(local)


def _cargar_config_servers(ruta: Path | None = None) -> dict[str, dict]:
    """Config que consume el manager: merge plantilla ← local.

    `ruta` (compat con MCPClientManager(config_path=...)) permite apuntar la
    PLANTILLA a otra ruta; la capa local se mergea igual.
    """
    return cargar_merge(ruta_base=ruta or RUTA_BASE)


# ══════════════════════════════════════════════════════════════════════
# MANAGER — event loop en thread aparte + conexiones persistentes
# ══════════════════════════════════════════════════════════════════════

class MCPClientManager:
    """
    Gestiona conexiones MCP persistentes y expone una API síncrona.

    Una sola instancia debe vivir durante toda la sesión de Aether
    (ver get_mcp_manager() más abajo para el singleton de conveniencia).
    """

    def __init__(self, config_path: Path | None = None):
        self._config = _cargar_config_servers(config_path)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._queue: asyncio.Queue | None = None  # cola de (coro, future) para el dispatcher

        # Se acceden SOLO desde dentro de la task del dispatcher (por eso
        # son "privados a esa task": todo enter/exit de context managers
        # async debe ocurrir en la MISMA task, o anyio revienta con
        # "cancel scope in a different task than it was entered in").
        self._sessions: dict[str, Any] = {}       # server -> ClientSession
        self._tools_cache: dict[str, list] = {}   # server -> [Tool, ...]
        self._exit_stack: AsyncExitStack | None = None

        self._iniciar_loop_en_thread()

    # ── Arranque del loop + task dispatcher persistente ──────────────
    #
    # Todas las operaciones (conectar, listar tools, llamar tools, cerrar)
    # se ejecutan dentro de UNA SOLA task de asyncio que vive mientras
    # dure el manager. Si en cambio se despachara cada llamada con
    # run_coroutine_threadsafe() directo (una task nueva por llamada),
    # los context managers de anyio (stdio_client/ClientSession) quedan
    # abiertos en una task y se intentan cerrar en otra -> error.

    def _iniciar_loop_en_thread(self) -> None:
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._queue = asyncio.Queue()
            self._exit_stack = AsyncExitStack()
            loop.create_task(self._dispatcher())
            self._ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=_run, name="mcp-client-loop", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        if self._loop is None:
            raise MCPError("No se pudo iniciar el event loop de MCP.")

    async def _dispatcher(self) -> None:
        """
        Única task que ejecuta TODAS las coroutines del manager (conectar,
        listar, llamar tools, cerrar). Garantiza que los cancel scopes de
        anyio se abran y cierren siempre en la misma task.

        Al recibir la señal de shutdown (None), es esta misma task la que
        detiene el loop -- si en cambio close() detuviera el loop por su
        cuenta con otro call_soon_threadsafe(), quedaría una carrera entre
        "el dispatcher procesa el None y termina" y "el loop se detiene",
        dejando a veces esta task pending al momento de destruirse.
        """
        while True:
            item = await self._queue.get()
            if item is None:  # señal de shutdown
                break
            coro, fut = item
            try:
                resultado = await coro
                if not fut.cancelled():
                    fut.set_result(resultado)
            except Exception as e:
                if not fut.cancelled():
                    fut.set_exception(e)
        self._loop.stop()

    def _run_coro(self, coro, timeout: float = 30.0):
        """
        Encola una coroutine para que la ejecute el dispatcher (siempre
        la misma task) y espera el resultado de forma síncrona.
        """
        if self._loop is None or self._queue is None:
            raise MCPError("El event loop de MCP no está inicializado.")
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (coro, fut))
        return fut.result(timeout=timeout)

    # ── Conexión lazy a servers ──────────────────────────────────────

    async def _conectar_server(self, server: str):
        """Conecta (si no está conectado) y devuelve la ClientSession del server."""
        if server in self._sessions:
            return self._sessions[server]

        cfg = self._config.get(server)
        if cfg is None:
            raise MCPServerNotFoundError(
                f"Server MCP '{server}' no está en la configuración "
                f"({_ruta_config_default()})."
            )

        transporte = cfg.get("transport", "stdio")
        if transporte != "stdio":
            raise MCPError(
                f"Transporte '{transporte}' no soportado todavía para el "
                f"server '{server}' (solo 'stdio' por ahora)."
            )

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=cfg["command"],
            args=cfg.get("args", []),
            env=cfg.get("env"),
        )

        read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        self._sessions[server] = session
        print(f"🔌 [MCP]: conectado a server '{server}'.")
        return session

    # ── API async interna ────────────────────────────────────────────

    async def _list_tools_async(self, server: str) -> list:
        if server in self._tools_cache:
            return self._tools_cache[server]
        session = await self._conectar_server(server)
        resp = await session.list_tools()
        self._tools_cache[server] = resp.tools
        return resp.tools

    async def _call_tool_async(self, server: str, name: str, arguments: dict) -> str:
        session = await self._conectar_server(server)
        try:
            resultado = await session.call_tool(name, arguments or {})
        except Exception as e:
            raise MCPToolCallError(
                f"Fallo al llamar '{name}' en server '{server}': {e}"
            ) from e

        return _normalizar_resultado(resultado)

    # ── API pública SÍNCRONA (esto es lo que usa node_mcp) ───────────

    def list_servers(self) -> list[str]:
        """Servers configurados (conectados o no todavía)."""
        return list(self._config.keys())

    def list_tools(self, server: str) -> list[dict]:
        """
        Lista las tools disponibles en un server (conecta si hace falta).
        Devuelve [{"name":..., "description":..., "input_schema":...}, ...],
        útil para inyectar en el prompt del planner Y para validar/resolver
        argumentos requeridos antes de llamar la tool (ver node_mcp).

        NOTA: el SDK de MCP expone el JSON Schema de entrada como
        `inputSchema` (camelCase, tal cual el spec MCP). Antes este método
        lo descartaba por completo, lo que dejaba a toda la lógica de
        validación de argumentos requeridos (_inferir_args_mcp,
        obtener_catalogo_mcp_condensado) operando siempre sobre {} sin que
        nadie lo notara.
        """
        tools = self._run_coro(self._list_tools_async(server))
        return [
            {
                "name": t.name,
                "description": getattr(t, "description", "") or "",
                "input_schema": getattr(t, "inputSchema", None) or {},
            }
            for t in tools
        ]

    def get_tool_schema(self, server: str, name: str) -> dict:
        """
        Devuelve el input_schema (JSON Schema) de una tool puntual, o {}
        si no se encuentra. Usa list_tools (cacheado) por debajo.
        """
        for t in self.list_tools(server):
            if t.get("name") == name:
                return t.get("input_schema") or {}
        return {}

    def list_all_tools(self) -> dict[str, list[dict]]:
        """Catálogo completo: {server: [tools...]} para todos los servers configurados."""
        catalogo = {}
        for server in self.list_servers():
            try:
                catalogo[server] = self.list_tools(server)
            except Exception as e:
                print(f"⚠️  [MCP]: no se pudo listar tools de '{server}': {e}")
                catalogo[server] = []
        return catalogo

    def call_tool(self, server: str, name: str, arguments: dict | None = None) -> str:
        """
        Llama una tool MCP y devuelve el resultado normalizado a string.
        Lanza MCPServerNotFoundError / MCPToolCallError en caso de fallo.
        """
        return self._run_coro(self._call_tool_async(server, name, arguments or {}))

    def close(self) -> None:
        """Cierra todas las conexiones y detiene el loop. Llamar al salir de Aether."""
        if self._loop is None:
            return

        async def _cerrar():
            if self._exit_stack is not None:
                await self._exit_stack.aclose()

        try:
            self._run_coro(_cerrar(), timeout=10)
        except Exception as e:
            print(f"⚠️  [MCP]: error cerrando conexiones: {e}")
        finally:
            # Señal de shutdown para el dispatcher. Es el propio dispatcher
            # (ver _dispatcher) el que llama loop.stop() al procesarla, para
            # evitar la carrera de detener el loop antes de que la task
            # termine de salir de su while.
            if self._queue is not None:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
            if self._thread is not None:
                self._thread.join(timeout=5)


def _normalizar_resultado(resultado) -> str:
    """
    Aplana el resultado de session.call_tool() (lista de content blocks:
    TextContent, ImageContent, etc.) a un string plano para mcp_result.
    """
    partes = []
    contenido = getattr(resultado, "content", None) or []
    for bloque in contenido:
        texto = getattr(bloque, "text", None)
        if texto:
            partes.append(texto)
        else:
            tipo = getattr(bloque, "type", "desconocido")
            partes.append(f"[contenido no textual: {tipo}]")

    if getattr(resultado, "isError", False):
        return f"[ERROR MCP] {' '.join(partes) or 'la tool devolvió error sin detalle'}"

    return "\n".join(partes) if partes else ""


# ══════════════════════════════════════════════════════════════════════
# SINGLETON DE CONVENIENCIA
# ══════════════════════════════════════════════════════════════════════
# node_mcp necesita una sola instancia viva durante toda la sesión (no una
# por-llamada), para que las conexiones stdio persistan entre tool calls.

_manager_singleton: MCPClientManager | None = None
_singleton_lock = threading.Lock()


def get_mcp_manager() -> MCPClientManager:
    global _manager_singleton
    if _manager_singleton is None:
        with _singleton_lock:
            if _manager_singleton is None:
                _manager_singleton = MCPClientManager()
    return _manager_singleton
