"""
ws_routes.py — Socket de chat (/ws/chat): la Web UI (o cualquier cliente)
se conecta y listo. Un solo socket, múltiples mensajes por conexión.

Protocolo JSON en ambos sentidos:

Cliente → servidor:
    {"message": "hola"}          → ejecuta el grafo (streaming de eventos)
    {"type": "stop"}             → cancela la inferencia en curso
    {"type": "ping"}             → heartbeat, responde {"type": "pong"}

Servidor → cliente (mismo contrato de eventos que el SSE /api/chat/stream):
    {"type": "hello", "version": ..., "busy": ...}   (al conectar)
    {"type": "start", "busy": ...}
    {"type": "token", "data": "..."}
    {"type": "node", "node": "...", "estado": "...", "delta": {...}}
    {"type": "log", "data": "..."}
    {"type": "done", "response": "..."}
    {"type": "error", "message": "..."}
    {"type": "pong"}

Es un transporte aditivo: no toca el SSE existente. El lock de generación
lo maneja AetherService (lo libera el hilo del grafo al terminar), así que
una desconexión a mitad de stream nunca deja el backend "ocupado".
"""

import asyncio
import itertools

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.aether_service import (
    AetherService,
    TokenEvent,
    StdoutLineEvent,
    NodeUpdateEvent,
    DoneEvent,
    ErrorEvent,
)
from backend.api.routes.chat import _delta_front, _preparar_orden
from core.utils.response_cleaner import limpiar_respuesta_chat
from tui.status_messages import real_state_for_node

router = APIRouter()


def _iter_payloads(mensaje: str):
    """Generador síncrono: eventos del grafo → dicts JSON (espejo del SSE)."""
    for evento in AetherService.iter_eventos(mensaje):
        if isinstance(evento, TokenEvent):
            yield {"type": "token", "data": evento.fragmento}
        elif isinstance(evento, NodeUpdateEvent):
            yield {
                "type": "node",
                "node": evento.nodo,
                "estado": real_state_for_node(evento.nodo, evento.delta),
                "delta": _delta_front(evento.delta),
            }
        elif isinstance(evento, StdoutLineEvent):
            yield {"type": "log", "data": evento.texto[:2000]}
        elif isinstance(evento, DoneEvent):
            yield {"type": "done", "response": limpiar_respuesta_chat(evento.respuesta)}
        elif isinstance(evento, ErrorEvent):
            yield {"type": "error", "message": evento.mensaje}


async def _stream_mensaje(ws: WebSocket, mensaje: str) -> None:
    """Corre UNA orden por el grafo y streamea los eventos por el socket."""
    await ws.send_json({"type": "start", "busy": AetherService.ocupado()})
    gen = _iter_payloads(mensaje)
    loop = asyncio.get_running_loop()
    _VACIO = itertools.chain()  # sentinel del run_in_executor
    try:
        while True:
            payload = await loop.run_in_executor(None, next, gen, _VACIO)
            if payload is _VACIO:
                break
            await ws.send_json(payload)
    finally:
        try:
            gen.close()
        except Exception:  # noqa: BLE001
            pass


@router.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    await ws.send_json(
        {
            "type": "hello",
            "name": "Aether",
            "busy": AetherService.ocupado(),
            "hint": 'Enviá {"message": "..."} — mismo contrato que /api/chat/stream.',
        }
    )
    try:
        while True:
            data = await ws.receive_json()
            tipo = data.get("type")
            if tipo == "ping":
                await ws.send_json({"type": "pong"})
                continue
            if tipo == "stop":
                from core.agent.streaming import request_cancel

                request_cancel()
                await ws.send_json({"type": "stopped"})
                continue
            # Visión/IO fuera del event loop (ver chat.py /chat/stream).
            mensaje, _metas = await asyncio.get_running_loop().run_in_executor(
                None, _preparar_orden, data)
            if not mensaje.strip():
                await ws.send_json(
                    {"type": "error", "message": "Mandá {'message': 'texto'}"}
                )
                continue
            await _stream_mensaje(ws, mensaje)
    except WebSocketDisconnect:
        # Cliente se fue: el lock lo libera el hilo del grafo al terminar.
        return
