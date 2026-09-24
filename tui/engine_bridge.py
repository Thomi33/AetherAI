"""
engine_bridge.py — Puente entre la TUI (Textual) y el motor LangGraph de Aether.

OPCIÓN A (mínima invasión): no tocamos graph_nodes.py. En vez de eso:

1. Replicamos las 3 líneas de inicialización que hacía AetherService
   (inicializar_db, cargar_memoria, actualizar_flatpaks) pero sin el lock de
   threading ni el resto del aparataje pensado para FastAPI multi-request.

2. Invocamos el grafo con `.stream(estado, stream_mode="updates")` en vez de
   `.invoke()`. Esto nos da, nodo por nodo, el dict parcial que cada uno
   retorna — exactamente lo que graph_state.py define (plan_pasos, plan_index,
   error_activo, etc.) sin tener que parsear nada.

3. Como los nodos además hacen print() y _llm_chat hace print(token, ...) para
   el streaming, redirigimos sys.stdout a un buffer mientras el grafo corre.
   Un hilo aparte va leyendo ese buffer línea por línea (o char por char para
   el streaming de tokens) y lo empuja a una cola (queue.Queue) que la TUI
   consume de forma asíncrona.

Esto es deliberadamente un puente "de afuera": si mañana migramos a la Opción
B (callbacks de estado reales en graph_nodes.py), este archivo es el único
que hay que reescribir — la TUI (app.py) no se entera del cambio porque
consume la misma interfaz pública: iter_eventos(orden).
"""

from __future__ import annotations

import io
import sys
import re
import queue
import threading
import contextlib
import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

import requests

PROJECT_ROOT_HINT = "Ajustá sys.path en run.py, no aquí."


# ══════════════════════════════════════════════════════════════════════
# EVENTOS QUE LA TUI CONSUME
# ══════════════════════════════════════════════════════════════════════

@dataclass
class NodeUpdateEvent:
    """Un nodo del grafo terminó. Viene de stream_mode='updates'."""
    nodo: str
    delta: dict[str, Any]


@dataclass
class StdoutLineEvent:
    """Una línea completa de stdout capturada (print() de algún nodo, con \\n)."""
    texto: str


@dataclass
class TokenEvent:
    """
    Fragmento de streaming de texto sin salto de línea (los tokens que
    _llm_chat va imprimiendo con print(t, end='', flush=True)). No es una
    línea completa — la TUI debe ir concatenándolos en el chat en vivo.
    """
    fragmento: str


@dataclass
class DoneEvent:
    """El grafo terminó. Trae la respuesta final."""
    respuesta: str


@dataclass
class ErrorEvent:
    """Excepción no controlada durante la ejecución del grafo."""
    mensaje: str


Evento = NodeUpdateEvent | StdoutLineEvent | TokenEvent | DoneEvent | ErrorEvent


# ══════════════════════════════════════════════════════════════════════
# ESTADO DEL MOTOR (memoria en RAM, igual que AetherService pero sin lock)
# ══════════════════════════════════════════════════════════════════════

@dataclass
class _MotorState:
    inicializado: bool = False
    mem: dict = field(default_factory=dict)


_motor = _MotorState()
logger = logging.getLogger(__name__)

_RE_ANSI = re.compile(r"\x1b\[[0-9;]*[mGKHhJsu]")



def inicializar_motor() -> None:
    """
    Equivalente a AetherService.initialize(), sin threading.Lock:
    la TUI es un solo proceso interactivo, no hay requests concurrentes
    que proteger.

    Usa asegurar_esquema + normalizar_mem para garantizar que "conversacion"
    y otras claves siempre existan.
    """
    if _motor.inicializado:
        return

    from core.memory.memory_manager import asegurar_esquema, cargar_memoria, normalizar_mem
    from core.tools.flatpak_manager import actualizar_flatpaks

    asegurar_esquema()
    raw_mem = cargar_memoria()
    _motor.mem = normalizar_mem(raw_mem)

    # Sesión única por ejecución de la TUI
    import uuid
    if not _motor.mem.get("sesion_id"):
        _motor.mem["sesion_id"] = uuid.uuid4().hex

    actualizar_flatpaks(_motor.mem, salida_lista="")
    _motor.inicializado = True


def calentar_modelo() -> None:
    """Envía un prompt mínimo para cargar el modelo local de Ollama.

    No crea un turno de conversación ni escribe en la memoria: solo deja el
    modelo listo para la primera orden real de la TUI.
    """
    from core.config.config_manager import get_config_manager

    config = get_config_manager()
    host = str(config.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    model = str(config.get("MODELO", "ornith-1.5:9b"))
    options = dict(config.get("OLLAMA_GEN_OPTIONS", {}))
    options.update({"num_ctx": config.get("NUM_CTX", 8192), "num_predict": 1})

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": "Responde únicamente: READY",
                "stream": False,
                "keep_alive": config.get("OLLAMA_KEEP_ALIVE", -1),
                "options": options,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("No se pudo calentar Ollama (%s): %s", host, exc)
        return

    logger.info("Modelo Ollama listo: %s", model)


def obtener_historial_para_mostrar(n: int = 50) -> list[dict]:
    """
    Trae los últimos N turnos guardados en el store JSON para mostrarlos al
    abrir la TUI. No filtra por sesion_id: cada corrida de la TUI genera
    una sesión nueva (ver arriba), así que filtrar por la sesión actual
    siempre daría una lista vacía. Esto es solo para continuidad visual
    del historial, no para el contexto que recibe el LLM (eso lo maneja
    node_context_manager con su propio filtrado por sesión/tema).
    """
    from core.memory.memory_manager import obtener_ultimos_turnos
    try:
        return obtener_ultimos_turnos(n)
    except Exception as e:  # noqa: BLE001
        print(f"[TUI] Error trayendo historial para mostrar: {e}")
        return []


class _QueueWriter(io.TextIOBase):
    """
    file-like object que reemplaza sys.stdout temporalmente.

    Después de mover el streaming de tokens a streaming.emit_token(),
    todo lo que llega acá es SIEMPRE log/debug (print() de graph_nodes.py:
    [MCP], [PLAN EXECUTOR], [FALLBACK], etc.). Ya no hay tokens del LLM
    compitiendo por este buffer, así que no hace falta detectar prefijos
    ni mantener estado de "estamos en streaming".

    - '\\n' → línea completa de log → StdoutLineEvent
    - '\\r' → línea de sobreescritura (countdown ⏳) → se descarta
    """

    def __init__(self, q: "queue.Queue[Evento]"):
        self._q = q
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        if "\n" not in self._buffer and "\r" not in self._buffer:
            return len(s)

        partes = re.split(r"(\n|\r)", self._buffer)
        self._buffer = partes.pop()  # leftover sin separador (o "" si terminó justo en uno)

        for i in range(0, len(partes), 2):
            segmento, sep = partes[i], partes[i + 1]
            if sep == "\r":
                continue  # countdown / overwrite → descartar
            texto = segmento.rstrip("\r\n")
            # Truncar líneas larguísimas (rutas/JSON crudo) para no degradar la UI.
            if len(texto) > 500:
                texto = texto[:500] + "…"
            if texto.strip():
                # Quitar códigos ANSI (colores del shell, p.ej.
                # \x1b[1;33m...\x1b[0m) para que no rompan el
                # parser de markup de Textual al llegar al DebugPanel.
                texto = _RE_ANSI.sub("", texto)
                self._q.put(StdoutLineEvent(texto=texto))
        return len(s)

    def flush(self) -> None:
        pass

    def vaciar_residual(self) -> None:
        resto = _RE_ANSI.sub("", self._buffer)
        if len(resto) > 500:
            resto = resto[:500] + "…"
        if resto.strip():
            self._q.put(StdoutLineEvent(texto=resto))
        self._buffer = ""


def _correr_grafo_en_hilo(orden: str, q: "queue.Queue[Evento]") -> None:
    writer = _QueueWriter(q)
    try:
        from core.agent.graph_builder import get_graph
        from core.agent.graph_state import crear_estado_inicial
        from core.memory.memory_manager import registrar_turno
        from core.agent.streaming import InferenceCancelled, set_token_sink, clear_token_sink

        set_token_sink(lambda frag: q.put(TokenEvent(fragmento=frag)))
        registrar_turno(_motor.mem, "usuario", orden)

        grafo = get_graph()
        estado = crear_estado_inicial(orden, _motor.mem, modo_autonomo=True)
        ultimo_estado: dict[str, Any] = dict(estado)

        with contextlib.redirect_stdout(writer):
            for update in grafo.stream(estado, stream_mode="updates"):
                for nodo, delta in update.items():
                    if isinstance(delta, dict):
                        ultimo_estado.update(delta)
                    q.put(NodeUpdateEvent(nodo=nodo, delta=delta or {}))
            writer.vaciar_residual()

        respuesta = ultimo_estado.get("final_response") or "Operación completada."
        # Comparte el scheduler con API/CLI: coalesce de llamadas, una sola
        # consolidación LLM a la vez y refresco del resumen RAM al completarse.
        from core.memory.consolidator import programar_consolidacion
        programar_consolidacion(_motor.mem)
        q.put(DoneEvent(respuesta=respuesta))

    except InferenceCancelled:
        writer.vaciar_residual()
        q.put(ErrorEvent(mensaje="Inferencia cancelada por el usuario."))
    except Exception as e:  # noqa: BLE001 — cualquier falla acá, incluidos imports, debe llegar a la UI
        writer.vaciar_residual()
        q.put(ErrorEvent(mensaje=f"{type(e).__name__}: {e}"))
    finally:
        try:
            from core.agent.streaming import clear_token_sink
            clear_token_sink()
        except Exception:
            pass


def iter_eventos(orden: str) -> Iterator[Evento]:
    """
    Generador síncrono que la TUI consume desde un worker de Textual
    (@work(thread=True) o run_worker con thread=True).

    Internamente lanza un hilo que corre el grafo y va empujando eventos
    a una queue.Queue; este generador los va sacando y entregando uno a
    uno (bloqueante, pero se ejecuta en un worker thread, así que no
    congela la UI).
    """
    from core.agent.streaming import reset_cancel
    reset_cancel()
    q: "queue.Queue[Evento]" = queue.Queue()
    hilo = threading.Thread(target=_correr_grafo_en_hilo, args=(orden, q), daemon=True)
    hilo.start()

    while True:
        evento = q.get()
        yield evento
        if isinstance(evento, (DoneEvent, ErrorEvent)):
            break


def cancelar_inferencia() -> None:
    """Solicita detener la inferencia activa en el siguiente token recibido."""
    from core.agent.streaming import request_cancel
    request_cancel()
