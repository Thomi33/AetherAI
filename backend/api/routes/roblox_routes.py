"""
roblox_routes.py — Control del runtime autónomo de Roblox desde la Web UI.

Es el comando /play-roblox de la TUI (tui/app.py::_manejar_play_roblox):
- POST /api/roblox {"action": "start", "provider": "google"|"ollama"|"hybrid"}
- POST /api/roblox {"action": "stop"}
- GET  /api/roblox/status

El runtime corre en un subproceso (core/tools/roblox_bridge.py::RobloxRuntime).
Igual que en la TUI, el guarda el runtime en un singleton de proceso y la
config se pasa como snapshot (env vars), sin tocar config.json.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

_RUNTIME = None  # RobloxRuntime del proceso backend (como _roblox_runtime de la TUI)

PROVIDERS = ("google", "ollama", "hybrid")


def _status() -> dict:
    running = _RUNTIME is not None and _RUNTIME.running
    return {
        "running": running,
        "provider": _RUNTIME.vision_provider if _RUNTIME else None,
        "pid": (_RUNTIME.process.pid
                if running and _RUNTIME.process is not None else None),
    }


@router.get("/roblox/status")
async def status():
    return {"ok": True, **_status()}


class RobloxBody(BaseModel):
    action: str = "start"          # start | stop | status
    provider: str | None = None    # google | ollama | hybrid (default: hybrid)


@router.post("/roblox")
async def control(body: RobloxBody):
    """Espejo de _manejar_play_roblox de la TUI."""
    global _RUNTIME
    from core.tools.roblox_bridge import RobloxRuntime

    accion = (body.action or "start").lower()
    try:
        if accion == "stop":
            if _RUNTIME is None:
                ok, msg = False, "El runtime Roblox no está activo."
            else:
                ok, msg = _RUNTIME.stop()
        elif accion == "status":
            running = _RUNTIME is not None and _RUNTIME.running
            ok, msg = True, ("Roblox autónomo activo."
                             if running else "Roblox inactivo.")
        elif accion == "start" or accion in PROVIDERS:
            # start con provider opcional (o provider como acción directa,
            # como hace la TUI: /play-roblox google)
            provider = accion if accion in PROVIDERS else body.provider
            if provider not in PROVIDERS:
                provider = None          # None → default del bridge (hybrid)
            if _RUNTIME is not None and _RUNTIME.running:
                ok, msg = False, "El runtime Roblox ya está activo."
            else:
                _RUNTIME = (RobloxRuntime(vision_provider=provider)
                            if provider else RobloxRuntime())
                ok, msg = _start_con_config(_RUNTIME)
        else:
            return {"ok": False,
                    "message": "Uso: action = start | stop | status | google |"
                               " ollama | hybrid"}
    except Exception as e:  # noqa: BLE001 — la UI muestra el mensaje
        return {"ok": False, "message": f"{type(e).__name__}: {e}",
                **_status()}

    return {"ok": ok, "message": msg, **_status()}


def _start_con_config(runtime) -> tuple[bool, str]:
    """start() con snapshot de config (MODELO/OLLAMA_HOST vía env vars)."""
    from core.config.config_manager import get_config_manager

    try:
        snapshot = get_config_manager().get_all()
    except Exception:  # noqa: BLE001 — sin snapshot arranca con defaults
        snapshot = None
    return runtime.start(config=snapshot)
