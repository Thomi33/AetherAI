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
    _log_event,
    _CURRENT_STATE,
    _CANCEL_EVENT,
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


def _payload_de_evento(event) -> dict | None:
    """Convierte eventos internos a payloads SSE/WS normalizados.
    Devuelve None para eventos desconocidos (se omiten en el stream)."""
    from backend.core.aether_service import (
        TokenEvent, ReasoningEvent, NodeUpdateEvent, StdoutLineEvent,
        DoneEvent, ErrorEvent,
    )
    if isinstance(event, TokenEvent):
        return {"type": "token", "data": event.fragmento}
    if isinstance(event, ReasoningEvent):
        return {"type": "reasoning", "data": event.fragmento}
    if isinstance(event, NodeUpdateEvent):
        return {
            "type": "node",
            "node": event.nodo,
            "estado": getattr(event, "estado", ""),
            "delta": _delta_front(event.delta),
        }
    if isinstance(event, StdoutLineEvent):
        return {"type": "log", "data": event.texto}
    if isinstance(event, DoneEvent):
        return {
            "type": "done",
            "response": event.respuesta,
            "reasoning": event.reasoning or "",
        }
    if isinstance(event, ErrorEvent):
        return {"type": "error", "message": event.mensaje}
    return None  # tipo desconocido → se omite


def _extraer_mensaje(request: dict) -> str:
    return (request.get("message") or request.get("content") or "").strip()


def _preparar_orden(body: dict) -> tuple[str, list[dict], list[str]]:
    """
    Mensaje del usuario + adjuntos (imágenes/archivos) → orden final para el
    grafo. Los adjuntos pueden venir de dos formas (se mezclan):

    - {name, mime, data}  — base64 crudo: se guardan en disco ahora.
    - {name, mime, path}  — ya subidos vía POST /attachments/upload: se
      referencian por path validado, SIN volver a guardarlos ni a mandar
      su contenido por el body (evita el bug de "adjunto vacío" y el
      doble envío de base64 pesado).

    Devuelve (orden, metas, imagenes_b64).
    """
    mensaje = _extraer_mensaje(body)
    adjuntos = body.get("attachments") or []
    if not isinstance(adjuntos, list) or not adjuntos:
        return mensaje, [], []
    from backend.core.attachments import (
        adjuntos_ya_guardados, componer_orden, guardar_adjuntos,
        generar_thumbnail_b64, modelo_admite_imagenes,
    )
    from core.config.settings import MODELO
    pendientes = [a for a in adjuntos if a.get("data")]
    pre_subidos = [a for a in adjuntos if not a.get("data") and a.get("path")]
    metas_previas = adjuntos_ya_guardados(pre_subidos)
    metas_nuevas, errores = guardar_adjuntos(pendientes)
    metas = metas_previas + metas_nuevas

    # Verificar si el modelo soporta visión nativa
    soporta_vision = modelo_admite_imagenes(MODELO)

    # Generar thumbnails para imágenes SOLO si el modelo soporta visión
    imagenes_b64 = []
    if soporta_vision:
        for m in metas:
            if m.get("kind") == "image" and m.get("path"):
                thumb = generar_thumbnail_b64(m["path"])
                if thumb:
                    imagenes_b64.append(thumb)

    # inline=True saltea la descripción por visión (modelo la ve directo)
    inline = bool(imagenes_b64)
    orden = componer_orden(mensaje, metas, errores, describir=not inline, inline=inline)
    return orden, metas, imagenes_b64


@router.post("/chat")
async def chat(request: dict):
    # La preparación puede describir imágenes con el modelo de visión
    # (segundos): nunca dentro del event loop.
    orden, _metas, _imagenes = await asyncio.get_running_loop().run_in_executor(
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
    orden, _metas, _imagenes = await asyncio.get_running_loop().run_in_executor(
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
        # Include model reasoning capability in start event so frontend knows
        # whether to show reasoning block (only for models with native reasoning)
        from core.config.settings import MODELO
        from backend.core.attachments import modelo_admite_imagenes
        # Check if model supports reasoning (native thinking models)
        # Currently models with vision typically also support reasoning
        # but we use a more specific check: models that emit ReasoningEvent
        # For now, assume models with "gemma" or "qwen" or "deepseek" in name support reasoning
        model_name = MODELO.lower()
        has_reasoning = any(x in model_name for x in ("gemma", "qwen", "deepseek", "r1", "reasoning"))
        yield f"data: {json.dumps({'type': 'start', 'busy': AetherService.ocupado(), 'reasoning': has_reasoning})}\n\n"

        gen = _consumir()
        loop = asyncio.get_running_loop()
        _VACIO = itertools.chain()  # sentinel del run_in_executor
        inf_id_global = [None]  # referencia mutable para logging

        def _client_gone() -> bool:
            """Chequea desconexión del cliente sin bloquear."""
            try:
                return loop.run_in_executor(
                    None, asyncio.run, asyncio.wait_for(
                        raw_request.is_disconnected(), timeout=0.01))
            except Exception:
                return raw_request.is_disconnected() is not False

        pending = None  # futuro del executor que corre next(gen) — uno solo
        try:
            while True:
                # Chequeo de desconexión al inicio de cada iteración del while
                # (antes de llamar al generador síncrono que llamaría a next()).
                # Si el cliente cerró: no abortamos la inferencia activa, solo
                # dejamos de consumir y el hilo del grafo libera el lock.
                try:
                    disconnected = await asyncio.wait_for(
                        raw_request.is_disconnected(), timeout=0.01)
                except asyncio.TimeoutError:
                    disconnected = False
                if disconnected:
                    _log_event("BACKEND_FRONTEND_CONNECTION_ERROR", None,
                               "cliente desconectado (detectado antes de next(gen))")
                    break

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
                        _log_event("BACKEND_FRONTEND_CONNECTION_ERROR", None,
                                   "cliente desconectado durante stream")
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


@router.get("/attachments/file")
async def get_attachment_file(
    path: str | None = None,
    name: str | None = None,
    thumb: int = 0,
):
    """Sirve un archivo adjunto guardado (o su thumbnail) para la Web UI.

    Parámetros:
    - path: ruta absoluta del archivo en disco (prioridad 1)
    - name: nombre del archivo original (busca en el directorio de adjuntos, prioridad 2)
    - thumb: 1 = devolver thumbnail JPEG, 0 = archivo original

    Responde 404 si el archivo no existe o está fuera del directorio permitido.
    """
    from backend.core.attachments import _directorio_adjuntos, generar_thumbnail_b64
    from fastapi.responses import Response
    from pathlib import Path

    base_dir = _directorio_adjuntos()

    # Resolver path del archivo
    target_path: Path | None = None
    if path:
        target_path = Path(path)
    elif name:
        # Buscar archivo que termine con el nombre dado (maneja prefijo timestamp)
        try:
            for f in base_dir.iterdir():
                if f.is_file() and f.name.endswith("_" + name) or f.name == name:
                    target_path = f
                    break
        except OSError:
            pass
        if target_path is None:
            return JSONResponse({"detail": "Archivo no encontrado"}, status_code=404)
    else:
        return JSONResponse({"detail": "Falta 'path' o 'name'"}, status_code=404)

    # Seguridad: path debe estar dentro del directorio de adjuntos
    try:
        target_path = target_path.resolve()
        if not target_path.is_relative_to(base_dir.resolve()):
            return JSONResponse({"detail": "Acceso denegado"}, status_code=404)
    except (ValueError, RuntimeError):
        return JSONResponse({"detail": "Path inválido"}, status_code=404)

    if not target_path.exists():
        return JSONResponse({"detail": "Archivo no encontrado"}, status_code=404)

    if thumb:
        # Generar/devolver thumbnail
        thumb_b64 = generar_thumbnail_b64(target_path)
        if thumb_b64:
            import base64
            return Response(content=base64.b64decode(thumb_b64), media_type="image/jpeg")
        # Si no se pudo generar thumbnail, servir original
        thumb = 0

    # Servir archivo original
    return Response(
        content=target_path.read_bytes(),
        media_type="application/octet-stream",
    )

