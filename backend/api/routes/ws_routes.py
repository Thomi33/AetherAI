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
    {"type": "reasoning", "data": "..."}
    {"type": "node", "node": "...", "estado": "...", "delta": {...}}
    {"type": "log", "data": "..."}
    {"type": "done", "response": "...", "reasoning": "..."}
    {"type": "error", "message": "..."}
    {"type": "pong"}

Es un transporte aditivo: no toca el SSE existente. El lock de generación
lo maneja AetherService (lo libera el hilo del grafo al terminar), así que
una desconexión a mitad de stream nunca deja el backend "ocupado".
"""

import asyncio
import itertools

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.aether_service import AetherService
# Serialización de eventos COMPARTIDA con el SSE /api/chat/stream: una sola
# vía (_payload_de_evento) mantiene ambos transportes a paridad — la cadena
# isinstance duplicada del SSE era la que perdía el ReasoningEvent.
from backend.api.routes.chat import _payload_de_evento, _preparar_orden

router = APIRouter()


def _iter_payloads(mensaje: str, imagenes: list[str] | None = None):
    """Generador síncrono: eventos del grafo → dicts JSON (espejo del SSE).

    Serializa con _payload_de_evento (chat.py): ReasoningEvent por su canal
    propio y done con respuesta limpia + razonamiento completo — idéntico
    contrato que /api/chat/stream."""
    for evento in AetherService.iter_eventos(mensaje, imagenes):
        payload = _payload_de_evento(evento)
        if payload is not None:
            yield payload


async def _stream_mensaje(ws: WebSocket, mensaje: str,
                          imagenes: list[str] | None = None) -> None:
    """Corre UNA orden por el grafo y streamea los eventos por el socket."""
    await ws.send_json({"type": "start", "busy": AetherService.ocupado()})
    gen = _iter_payloads(mensaje, imagenes)
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
            mensaje, _metas, imagenes = await asyncio.get_running_loop().run_in_executor(
                None, _preparar_orden, data)
            if not mensaje.strip():
                await ws.send_json(
                    {"type": "error", "message": "Mandá {'message': 'texto'}"}
                )
                continue
            await _stream_mensaje(ws, mensaje, imagenes)
    except WebSocketDisconnect:
        # Cliente se fue: el lock lo libera el hilo del grafo al terminar.
        return
