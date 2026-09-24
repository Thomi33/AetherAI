"""
mcp_routes.py — Gestión de servers MCP (core/config/mcp_servers.json).

Es el comando /mcps de la TUI, más la posibilidad de AGREGAR un MCP custom
desde la Web UI (nombre, transport, command, args, env) y de ver las tools
que expone cada server.

Formato del archivo (es el que consume core/tools/mcp_client.py):

    {
      "nombre": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@org/mcp-server"],
        "env": {"CLAVE": "valor"},
        "enabled": true
      }
    }

SEGURIDAD: `env` puede guardar tokens (GITHUB_PERSONAL_ACCESS_TOKEN, etc.).
El backend expone la API con CORS abierto, así que GET NUNCA devuelve los
valores de env en claro: devuelve los valores enmascarados ("********") más
un preview de 4 chars. Al escribir, un valor igual al mask conserva el valor
real guardado (keychain transparente).
"""

import asyncio
import json
import re

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

_RE_NOMBRE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MASK = "********"


def _ruta_config():
    """Misma ruta que usa core/tools/mcp_client.py: core/config/mcp_servers.json
    (resuelta desde el paquete, no desde este archivo, para no depender de
    cuántos niveles de carpeta tenga backend/api/routes/)."""
    from pathlib import Path
    import core.config
    return Path(core.config.__file__).resolve().parent / "mcp_servers.json"


def _leer() -> dict:
    ruta = _ruta_config()
    if not ruta.exists():
        return {}
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"No se pudo leer mcp_servers.json: {e}") from e


def _escribir(data: dict) -> None:
    _ruta_config().write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _preview(valor: str) -> str:
    """Enmascarado consistente: '••••••1234' (últimos 4, nunca un corte a
    mitad de palabra tipo '…KEN}' que deja ver un fragmento sin contexto)."""
    valor = str(valor)
    return f"••••••{valor[-4:]}" if len(valor) > 4 else "••••••"


def _serializar(nombre: str, cfg: dict) -> dict:
    """Server → dict JSON-safe con env enmascarado."""
    env = cfg.get("env") or {}
    return {
        "name": nombre,
        "transport": cfg.get("transport", "stdio"),
        "command": cfg.get("command", ""),
        "args": cfg.get("args") or [],
        "env": {
            k: {"valor": _MASK, "preview": _preview(v)}
            for k, v in env.items() if v is not None
        },
        "enabled": bool(cfg.get("enabled", True)),
    }


def _resolver_env(env_nuevo: dict | None, env_viejo: dict) -> dict:
    """Mezcla env nuevo con el guardado: valores enmascarados = conservar."""
    if env_nuevo is None:
        return dict(env_viejo)
    out: dict = {}
    for k, v in env_nuevo.items():
        if v == _MASK and k in env_viejo:
            out[k] = env_viejo[k]
        else:
            out[k] = v
    return out


@router.get("/mcps")
async def listar():
    try:
        data = _leer()
    except ValueError as e:
        return {"ok": False, "servers": [], "error": str(e)}
    servers = [_serializar(n, c) for n, c in data.items() if isinstance(c, dict)]
    return {"ok": True, "servers": servers}


class McpServerBody(BaseModel):
    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = []
    env: dict[str, str] | None = None
    enabled: bool = True


def _validar(body: McpServerBody) -> str | None:
    if not _RE_NOMBRE.match(body.name):
        return "Nombre inválido (letras, números, '-', '_' y '.')"
    if body.transport != "stdio":
        return "Por ahora solo se soporta transport 'stdio' " \
               "(ver core/tools/mcp_client.py)"
    if not (body.command or "").strip():
        return "El command es obligatorio (ej. 'npx')"
    return None


def _cfg_desde_body(body: McpServerBody, env_base: dict) -> dict:
    return {
        "transport": body.transport,
        "command": body.command.strip(),
        "args": [str(a) for a in body.args],
        "env": _resolver_env(body.env, env_base) or None,
        "enabled": bool(body.enabled),
    }


@router.post("/mcps")
async def agregar(body: McpServerBody):
    """Agrega un server MCP custom. 409 si el nombre ya existe (usar PATCH)."""
    err = _validar(body)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    try:
        data = _leer()
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    if body.name in data:
        return JSONResponse(
            {"ok": False, "error": f"Ya existe '{body.name}'"}, status_code=409)
    data[body.name] = _cfg_desde_body(body, {})
    try:
        _escribir(data)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "server": _serializar(body.name, data[body.name])}


@router.patch("/mcps/{nombre}")
async def editar(nombre: str, body: McpServerBody):
    """Edita un server (incluye el toggle enabled del /mcps de la TUI)."""
    err = _validar(body)
    if err:
        return JSONResponse({"ok": False, "error": err}, status_code=400)
    try:
        data = _leer()
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    if nombre not in data:
        return JSONResponse(
            {"ok": False, "error": f"No existe '{nombre}'"}, status_code=404)
    viejo = data[nombre] if isinstance(data[nombre], dict) else {}
    nueva = _cfg_desde_body(body, viejo.get("env") or {})
    if body.name != nombre:
        del data[nombre]
    data[body.name] = nueva
    try:
        _escribir(data)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True, "server": _serializar(body.name, nueva)}


@router.delete("/mcps/{nombre}")
async def borrar(nombre: str):
    try:
        data = _leer()
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    if nombre not in data:
        return JSONResponse(
            {"ok": False, "error": f"No existe '{nombre}'"}, status_code=404)
    del data[nombre]
    try:
        _escribir(data)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    return {"ok": True}


@router.get("/mcps/{nombre}/tools")
async def tools(nombre: str):
    """Tools que expone un server (conecta en vivo; puede tardar la 1ra vez).

    Corre en un thread porque el MCPClientManager es síncrono por diseño
    (despacha a un event loop propio, ver core/tools/mcp_client.py).
    """
    try:
        data = _leer()
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
    cfg = data.get(nombre)
    if not isinstance(cfg, dict):
        return JSONResponse(
            {"ok": False, "error": f"No existe '{nombre}'"}, status_code=404)
    if not cfg.get("enabled", True):
        return {"ok": False, "tools": [],
                "error": f"'{nombre}' está deshabilitado (activalo primero)"}

    def _listar():
        from core.tools.mcp_client import get_mcp_manager
        return get_mcp_manager().list_tools(nombre)

    try:
        tools_lista = await asyncio.wait_for(
            asyncio.to_thread(_listar), timeout=45
        )
        return {"ok": True, "server": nombre, "tools": tools_lista}
    except asyncio.TimeoutError:
        return {"ok": False, "tools": [],
                "error": "Timeout conectando al server (45s)"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "tools": [], "error": str(e)}
