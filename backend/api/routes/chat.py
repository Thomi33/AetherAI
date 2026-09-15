"""
chat.py — Rutas de chat del backend revivido.

- POST /api/chat         → respuesta completa (bloqueante, compat)
- POST /api/chat/stream  → SSE con tokens en vivo (motor LangGraph real)
- POST /api/stop         → cancela la inferencia en curso
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
import itertools

from backend.core.aether_service import (
    AetherService,
    TokenEvent,
    StdoutLineEvent,
    NodeUpdateEvent,
    DoneEvent,
    ErrorEvent,
)
from core.utils.response_cleaner import limpiar_respuesta_chat
from tui.status_messages import real_state_for_node

router = APIRouter()

# Claves del estado del grafo (graph_state.py) que la Web UI necesita para
# reconstruir el panel de plan/actividad que tiene la TUI (plan_panel.py).
# Se sanearán a JSON-safe antes de mandarlas por SSE.
_CLAVES_DELTA_FRONT = (
    "plan_pasos",
    "plan_index",
    "plan_activo",
    "tool_actual",
    "herramienta",
    "agent_pasos_log",
    "fs_result",
    "shell_output",
    "context_compactado",
    "error_activo",
)

_LIMITE_CAMPO = 1500  # truncado por campo para no inflar el SSE


def _sanear_campo(v, profundidad: int = 0):
    """Convierte un valor del delta a algo JSON-serializable y acotado."""
    if profundidad > 3 or v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return v[:_LIMITE_CAMPO] + ("…" if len(v) > _LIMITE_CAMPO else "")
    if isinstance(v, dict):
        out = {}
        for k, val in list(v.items())[:40]:
            try:
                out[str(k)] = _sanear_campo(val, profundidad + 1)
            except Exception:  # noqa: BLE001 — nunca romper el stream
                out[str(k)] = "<?>"
        return out
    if isinstance(v, (list, tuple)):
        return [_sanear_campo(x, profundidad + 1) for x in list(v)[:20]]
    return str(v)[:_LIMITE_CAMPO]


def _delta_front(delta) -> dict:
    """Extrae del delta del grafo solo lo que la Web UI consume (JSON-safe)."""
    if not isinstance(delta, dict):
        return {}
    return {
        clave: _sanear_campo(delta[clave])
        for clave in _CLAVES_DELTA_FRONT
        if clave in delta and delta[clave] not in (None, "", [], {}, False)
    }


def _extraer_mensaje(request: dict) -> str:
    return (request.get("message") or request.get("content") or "").strip()


@router.post("/chat")
async def chat(request: dict):
    message = _extraer_mensaje(request)
    if not message:
        return JSONResponse({"detail": "El campo 'message' es obligatorio"}, status_code=400)

    result = AetherService.process_message(message)
    return result


@router.post("/chat/stream")
async def chat_stream(raw_request: Request):
    body = await raw_request.json()
    message = _extraer_mensaje(body)
    if not message:
        return JSONResponse({"detail": "El campo 'message' es obligatorio"}, status_code=400)

    def _consumir():
        """El generador es síncrono: el grafo corre en su propio hilo."""
        for evento in AetherService.iter_eventos(message):
            if isinstance(evento, TokenEvent):
                payload = {"type": "token", "data": evento.fragmento}
            elif isinstance(evento, NodeUpdateEvent):
                payload = {
                    "type": "node",
                    "node": evento.nodo,
                    # Estado humano (mismo mapa que la TUI, status_messages.py)
                    # para el chip de actividad de la web.
                    "estado": real_state_for_node(evento.nodo, evento.delta),
                    # Delta saneado: plan_pasos/plan_index/tool_actual/
                    # agent_pasos_log/etc. → panel de plan/actividad web.
                    "delta": _delta_front(evento.delta),
                }
            elif isinstance(evento, StdoutLineEvent):
                payload = {"type": "log", "data": evento.texto[:2000]}
            elif isinstance(evento, DoneEvent):
                payload = {
                    "type": "done",
                    # Misma limpieza que la TUI (limpiar_respuesta_chat):
                    # sin [SHELL]...[/SHELL] ni JSON de tool-call residual.
                    "response": limpiar_respuesta_chat(evento.respuesta),
                }
            elif isinstance(evento, ErrorEvent):
                payload = {"type": "error", "message": evento.mensaje}
            else:
                continue
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    async def stream_response():
        yield f"data: {json.dumps({'type': 'start', 'busy': AetherService.ocupado()})}\n\n"

        gen = _consumir()
        loop = asyncio.get_running_loop()
        _VACIO = itertools.chain()  # sentinel del run_in_executor

        pending = None  # futuro del executor que corre next(gen) — uno solo
        try:
            while True:
                if pending is None:
                    pending = loop.run_in_executor(None, next, gen, _VACIO)
                try:
                    # Shield: un timeout NO cancela el executor (sigue solo).
                    chunk = await asyncio.wait_for(asyncio.shield(pending), timeout=15)
                    pending = None
                except asyncio.TimeoutError:
                    if await raw_request.is_disconnected():
                        # Cliente se fue: dejar de consumir. El lock lo
                        # libera el hilo del grafo al terminar (fix del bug
                        # "busy eterno" — ver aether_service._correr_grafo_en_hilo).
                        break
                    yield ": keep-alive\n\n"  # mantiene proxies/conexión vivos
                    continue
                if chunk is _VACIO:
                    break
                yield chunk
        finally:
            # Si el cliente cortó (Stop/cierre de pestaña), intentar cerrar
            # el generador síncrono; si está "already executing" en el
            # executor, no pasa nada: el lock lo libera el hilo igual.
            try:
                gen.close()
            except Exception:  # noqa: BLE001
                pass

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/stop")
async def stop():
    """Cancela la inferencia en curso (misma señal que usa la TUI)."""
    from core.agent.streaming import request_cancel
    request_cancel()
    return {"ok": True, "message": "Señal de cancelación enviada"}
