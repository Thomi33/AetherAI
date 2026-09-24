"""
chat.py — Rutas de chat del backend revivido.

- POST /api/chat         → respuesta completa (bloqueante, compat)
- POST /api/chat/stream  → SSE con tokens en vivo (motor LangGraph real)
- POST /api/stop         → cancela la inferencia en curso
- POST /api/attachments/upload  → upload multipart (imágenes, PDF, DOCX, audio, video)
- POST /api/attachments/transcribe  → transcripción audio/video con progress SSE
"""

from fastapi import APIRouter, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
import itertools
from pathlib import Path

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
    "_ornith_reasoning",   # razonamiento del modelo ([Pensando] en la Web UI)
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


_LIMITE_RAZONAMIENTO = 12000  # el razonamiento se muestra completo en la UI


def _delta_front(delta) -> dict:
    """Extrae del delta del grafo solo lo que la Web UI consume (JSON-safe)."""
    if not isinstance(delta, dict):
        return {}
    out = {}
    for clave in _CLAVES_DELTA_FRONT:
        if clave not in delta or delta[clave] in (None, "", [], {}, False):
            continue
        if clave == "_ornith_reasoning":
            # Va al chip "Pensando…" de la Web UI: límite más generoso que
            # el truncado general de 1500 chars (el razonamiento es largo).
            razonamiento = str(delta[clave])
            out[clave] = (razonamiento[:_LIMITE_RAZONAMIENTO]
                          + ("…" if len(razonamiento) > _LIMITE_RAZONAMIENTO
                             else ""))
        else:
            out[clave] = _sanear_campo(delta[clave])
    return out


def _extraer_mensaje(request: dict) -> str:
    return (request.get("message") or request.get("content") or "").strip()


def _preparar_orden(body: dict) -> tuple[str, list[dict]]:
    """
    Mensaje del usuario + adjuntos (imágenes/archivos) → orden final para el
    grafo. Los adjuntos pueden venir de dos formas (se mezclan):

    - {name, mime, data}  — base64 crudo: se guardan en disco ahora.
    - {name, mime, path}  — ya subidos vía POST /attachments/upload: se
      referencian por path validado, SIN volver a guardarlos ni a mandar
      su contenido por el body (evita el bug de "adjunto vacío" y el
      doble envío de base64 pesado).

    Devuelve (orden, metas).
    """
    mensaje = _extraer_mensaje(body)
    adjuntos = body.get("attachments") or []
    if not isinstance(adjuntos, list) or not adjuntos:
        return mensaje, []
    from backend.core.attachments import (
        adjuntos_ya_guardados, componer_orden, guardar_adjuntos,
    )
    pendientes = [a for a in adjuntos if a.get("data")]
    pre_subidos = [a for a in adjuntos if not a.get("data") and a.get("path")]
    metas_previas = adjuntos_ya_guardados(pre_subidos)
    metas_nuevas, errores = guardar_adjuntos(pendientes)
    metas = metas_previas + metas_nuevas
    return componer_orden(mensaje, metas, errores), metas


@router.post("/chat")
async def chat(request: dict):
    # La preparación puede describir imágenes con el modelo de visión
    # (segundos): nunca dentro del event loop.
    orden, _metas = await asyncio.get_running_loop().run_in_executor(
        None, _preparar_orden, request)
    if not orden.strip():
        return JSONResponse(
            {"detail": "Enviá 'message' o al menos un adjunto válido"},
            status_code=400)

    result = AetherService.process_message(orden)
    return result


@router.post("/chat/stream")
async def chat_stream(raw_request: Request):
    body = await raw_request.json()
    # Igual que /chat: visión/IO fuera del event loop.
    orden, _metas = await asyncio.get_running_loop().run_in_executor(
        None, _preparar_orden, body)
    if not orden.strip():
        return JSONResponse(
            {"detail": "Enviá 'message' o al menos un adjunto válido"},
            status_code=400)

    def _consumir():
        """El generador es síncrono: el grafo corre en su propio hilo."""
        for evento in AetherService.iter_eventos(orden):
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


@router.post("/attachments/upload")
async def upload_attachments(
    files: list[UploadFile] = File(...),
    message: str = Form(""),
):
    """Upload multipart de archivos (imágenes, PDF, DOCX, audio, video).
    Devuelve metadatos de los archivos guardados + texto extraído/transcrito.
    """
    from backend.core.attachments import guardar_adjuntos, componer_orden

    attachments = []
    for f in files[:8]:  # tope 8
        data = await f.read()
        import base64
        attachments.append({
            "name": f.filename,
            "mime": f.content_type or "application/octet-stream",
            "data": base64.b64encode(data).decode(),
        })

    def _procesar():
        metas_, errores_ = guardar_adjuntos(attachments)
        # describir=False: la visión corre UNA vez, al enviar el mensaje por
        # /chat|/chat/stream. Describir acá duplicaba la llamada al modelo
        # de visión (segundos por imagen) con un resultado que se descartaba.
        orden_ = componer_orden(message, metas_, errores_, describir=False)
        return metas_, errores_, orden_

    metas, errores, orden = await asyncio.get_running_loop().run_in_executor(
        None, _procesar)
    
    return {
        "ok": True,
        "orden": orden,
        "metas": [
            {
                "name": m["name"],
                "mime": m["mime"],
                "kind": m["kind"],
                "path": m.get("path"),
                "size": m.get("size"),
                "texto_extraido": m.get("texto_extraido"),
            }
            for m in metas
        ],
        "errores": errores,
    }


@router.post("/attachments/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("base"),
):
    """Transcribe audio/video usando faster-whisper con progress SSE.
    Modelo: tiny, base, small, medium, large.
    """
    from backend.core.attachments import _transcribir_audio
    from backend.core.attachments import _directorio_adjuntos
    import base64
    import tempfile
    
    # Guardar archivo temporal
    import tempfile
    sufijo = Path(file.filename or "audio.webm").suffix or ".webm"
    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "error": "Audio vacío"}, status_code=400)
    with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    
    try:
        # Usar modelo especificado
        from backend.core.attachments import _transcribir_audio
        from core.config.config_manager import get_config_manager
        
        cfg = get_config_manager()
        original_model = cfg.get("WHISPER_MODEL", "base")
        cfg.set("WHISPER_MODEL", model, validate=False)
        
        try:
            texto = await asyncio.get_event_loop().run_in_executor(None, _transcribir_audio, tmp_path)
        finally:
            cfg.set("WHISPER_MODEL", original_model, validate=False)
    finally:
        tmp_path.unlink(missing_ok=True)
    
    return {"ok": True, "text": texto, "model": model}


@router.post("/attachments/transcribe/stream")
async def transcribe_audio_stream(
    file: UploadFile = File(...),
    model: str = Form("base"),
):
    """Transcribe audio/video con progress SSE (tokens de progreso)."""
    # TODO: implementar streaming de progreso real con faster-whisper
    # Por ahora, delega al endpoint simple
    return await transcribe_audio(file, model)

