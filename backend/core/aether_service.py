"""
AetherService: Adaptador singleton para integración de Aether con FastAPI.
Utiliza la arquitectura refactorizada en core/ (motor LangGraph único).

BACKEND REVIVIDO:
- Inicializa UNA SOLA VEZ (DB + memoria RAM)
- Protege SQLite con threading.Lock
- process_message(): llamada bloqueante (compat)
- iter_eventos(): STREAMING real del grafo (tokens en vivo + eventos de
  nodos + logs de stdout), igual que la TUI consume vía tui/engine_bridge.py
- Lock de generación: una sola inferencia a la vez (el token sink es global)
"""

import logging
import threading
import sys
import queue
import contextlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterator

logger = logging.getLogger("aether_service")

# =====================================================================
# SINGLETON + LOCKS PARA THREAD-SAFETY
# =====================================================================
_AETHER_LOCK = threading.Lock()
_AETHER_INITIALIZED = False
_AETHER_MEMORY: Dict[str, Any] = None

# ── Sistema de ciclo de vida de inferencia ────────────────────────────
# Estado activo: evento de cancelación compartido entre el endpoint SSE
# (que detecta desconexión del cliente) y el hilo del grafo (que debe cortar
# antes de iniciar el siguiente paso; ahora se checkea en cada paso).
_CANCEL_EVENT = threading.Event()
# Lock de generación: una sola inferencia a la vez (el token sink es global)
_GEN_LOCK = threading.Lock()
# Estado actual (QUEUED|RUNNING|COMPLETED|FAILED|CANCELLED)
_CURRENT_STATE: dict = {
    "estado": "IDLE",
    "inf_id": None,
    "inicio": 0.0,
    "motivo": "",
    "error": None,
}

# Log estructurado de eventos de ciclo de vida (máx 50 eventos)
_EVENT_LOG: list[dict] = []

# Cola FIFO de prompts pendientes (cuando hay una inferencia activa).
# El endpoint _enqueue_prompt pone; el worker consumer la drena con timeout
# corto para no bloquear inútilmente.
_PROMPT_QUEUE: "queue.Queue[tuple[str, dict]]" = queue.Queue()


def _log_event(tipo: str, inf_id: str | None = None,
               motivo: str = "", detalle: dict | None = None) -> None:
    """Registra un evento de ciclo de vida en _EVENT_LOG."""
    _EVENT_LOG.append({
        "tipo": tipo,
        "inf_id": inf_id,
        "motivo": motivo,
        "detalle": detalle or {},
        "ts": time.time(),
    })
    if len(_EVENT_LOG) > 50:
        del _EVENT_LOG[:-50]
    logger.info(f"[{tipo}]" + (f" id={inf_id}" if inf_id else "")
                + (f": {motivo}" if motivo else ""))


def _emit_inference_start(inf_id: str) -> None:
    _CURRENT_STATE["estado"] = "RUNNING"
    _CURRENT_STATE["inf_id"] = inf_id
    _CURRENT_STATE["inicio"] = time.time()
    _CURRENT_STATE["motivo"] = ""
    _log_event("INFERENCE_STARTED", inf_id)


def _emit_inference_queued(inf_id: str, motivo: str = "") -> None:
    _CURRENT_STATE["estado"] = "QUEUED"
    _CURRENT_STATE["inf_id"] = inf_id
    _CURRENT_STATE["motivo"] = motivo
    _log_event("INFERENCE_QUEUED", inf_id, motivo)


def _emit_inference_completed(inf_id: str) -> None:
    _CURRENT_STATE["estado"] = "COMPLETED"
    _CURRENT_STATE["inf_id"] = inf_id
    _log_event("INFERENCE_COMPLETED", inf_id)


def _emit_inference_failed(inf_id: str, motivo: str = "") -> None:
    _CURRENT_STATE["estado"] = "FAILED"
    _CURRENT_STATE["inf_id"] = inf_id
    _CURRENT_STATE["motivo"] = motivo
    _CURRENT_STATE["error"] = motivo
    _log_event("INFERENCE_FAILED", inf_id, motivo)


def _emit_inference_cancelled(inf_id: str, motivo: str = "") -> None:
    _CURRENT_STATE["estado"] = "CANCELLED"
    _CURRENT_STATE["inf_id"] = inf_id
    _CURRENT_STATE["motivo"] = motivo
    _log_event("INFERENCE_CANCELLED", inf_id, motivo)


_RE_ANSI = re.compile(r"\x1b\[[0-9;]*[mGKHhJsu]")

# =====================================================================
# EVENTOS QUE EL BACKEND EMITE (idéntico contrato que tui/engine_bridge)
# =====================================================================
@dataclass
class NodeUpdateEvent:
    """Un nodo del grafo terminó (stream_mode='updates')."""
    nodo: str
    delta: dict


@dataclass
class StdoutLineEvent:
    """Línea de stdout capturada (print() de algún nodo)."""
    texto: str


@dataclass
class TokenEvent:
    """Fragmento de streaming de texto (tokens en vivo)."""
    fragmento: str


@dataclass
class ReasoningEvent:
    """Fragmento de streaming de RAZONAMIENTO del modelo.

    Canal SEPARADO de la respuesta: nunca se mezcla con TokenEvent (content
    limpio), ni con StdoutLineEvent (logs). La Web UI lo usa para el bloque
    de razonamiento expandible bajo la cabecera de estado.
    """
    fragmento: str


@dataclass
class DoneEvent:
    """El grafo terminó. Trae la respuesta final y el razonamiento completo
    del turno (por robustez, por si algún fragmento de streaming se perdió)."""
    respuesta: str
    reasoning: str = ""


@dataclass
class ErrorEvent:
    """Excepción no controlada durante la ejecución del grafo."""
    mensaje: str


Evento = (NodeUpdateEvent | StdoutLineEvent | TokenEvent | ReasoningEvent
          | DoneEvent | ErrorEvent)


class _QueueWriter:
    """Captura print()/stdout del grafo y lo manda a la cola como líneas."""

    def __init__(self, q: "queue.Queue[Evento]"):
        self._q = q
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        while "\n" in self._buffer:
            linea, self._buffer = self._buffer.split("\n", 1)
            texto = _RE_ANSI.sub("", linea).rstrip()
            if texto:
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


class AetherService:
    """Servicio singleton para ejecutar el agente Aether desde FastAPI."""

    @staticmethod
    def _setup_paths():
        """Agrega ruta del proyecto a sys.path."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

    @staticmethod
    def initialize():
        """Inicializa el agente Aether UNA SOLA VEZ."""
        global _AETHER_INITIALIZED, _AETHER_MEMORY

        if _AETHER_INITIALIZED:
            logger.debug("Aether ya inicializado, reutilizando instancia")
            return

        with _AETHER_LOCK:
            if _AETHER_INITIALIZED:
                return

            try:
                AetherService._setup_paths()

                from core.memory.memory_manager import inicializar_db, cargar_memoria

                logger.debug("Inicializando BD de Aether...")
                inicializar_db()

                logger.debug("Cargando memoria desde BD...")
                _AETHER_MEMORY = cargar_memoria()

                # Índice de programas (best-effort: si falla no bloquea el arranque)
                try:
                    from core.tools.flatpak_manager import actualizar_flatpaks
                    logger.debug("Indexando programas del sistema...")
                    actualizar_flatpaks(_AETHER_MEMORY, salida_lista="")
                except Exception as e:
                    logger.warning(f"Indexación de programas omitida: {e}")

                _AETHER_INITIALIZED = True
                logger.info("Aether inicializado correctamente")

            except Exception as e:
                logger.error(f"Error durante inicialización de Aether: {e}", exc_info=True)
                _AETHER_INITIALIZED = False
                raise

    @staticmethod
    def _asegurar_inicializado():
        if not _AETHER_INITIALIZED:
            AetherService.initialize()

    @staticmethod
    def ocupado() -> bool:
        """True si hay una inferencia en curso (el token sink es global)."""
        return _GEN_LOCK.locked()

    @staticmethod
    def actualizar_resumen_ram(texto: str) -> None:
        """Espeja el resumen nuevo en la memoria RAM del motor (igual que
        _motor_actualizar_resumen de la TUI): el próximo turno ya lo ve sin
        esperar a recargar memoria desde la DB."""
        if _AETHER_MEMORY is not None:
            _AETHER_MEMORY["resumen"] = texto

    @staticmethod
    def process_message(user_message: str,
                        imagenes: list[str] | None = None) -> Dict[str, Any]:
        """
        Procesa un mensaje del usuario y devuelve respuesta del agente
        (bloqueante, sin streaming). Para la Web UI usar iter_eventos().

        imagenes: thumbnails b64 de adjuntos (visión inline del modelo
        principal, ver backend/core/attachments.py).
        """
        try:
            AetherService._asegurar_inicializado()
        except Exception as e:
            logger.error(f"Fallo al inicializar Aether: {e}")
            return {"response": "Aether no pudo inicializarse", "agent_status": "error"}

        try:
            from core.services.graph_service import procesar_orden_grafo

            with _AETHER_LOCK:
                logger.debug(f"Procesando mensaje: {user_message[:50]}...")
                respuesta = procesar_orden_grafo(
                    user_message, _AETHER_MEMORY, modo_autonomo=True,
                    imagenes=imagenes)

            return {"response": respuesta, "agent_status": "ready"}

        except Exception as e:
            logger.error(f"Error procesando mensaje: {e}", exc_info=True)
            return {"response": "Aether encontró un error interno", "agent_status": "error"}

    @staticmethod
    def iter_eventos(orden: str,
                     imagenes: list[str] | None = None) -> Iterator[Evento]:
        """
        Generador con STREAMING real del grafo (mismo contrato que la TUI):
        TokenEvent / StdoutLineEvent / NodeUpdateEvent / DoneEvent / ErrorEvent.

        Lanza el grafo en un hilo y los eventos llegan vía queue.Queue.
        Solo una inferencia a la vez (token sink global compartido con la TUI).

        imagenes: thumbnails b64 de adjuntos de imagen (visión inline).
        """
        from core.agent.streaming import reset_cancel
        reset_cancel()

        inf_id = f"inf-{int(time.time())}-{id(orden) % 10000:04d}"

        # ── CICLO DE VIDA: QUEUED → RUNNING/FAILED/CANCELLED ──────────
        if not _GEN_LOCK.acquire(blocking=False):
            _emit_inference_queued(inf_id, "inferencia activa; en cola FIFO")
            _log_event("INFERENCE_QUEUED", inf_id, "encolada (lock ocupado)")
            yield ErrorEvent(
                mensaje="Ya hay una inferencia en curso. Esperá que termine o enviá /stop."
            )
            return

        _emit_inference_start(inf_id)
        try:
            AetherService._asegurar_inicializado()
        except Exception as e:
            _emit_inference_failed(inf_id, f"init failed: {e}")
            _GEN_LOCK.release()
            yield ErrorEvent(mensaje=f"Aether no pudo inicializarse: {type(e).__name__}: {e}")
            return

        q: "queue.Queue[Evento]" = queue.Queue()
        hilo = threading.Thread(
            target=_correr_grafo_en_hilo, args=(orden, q, imagenes, inf_id),
            daemon=True,
            name=f"grafo-{inf_id}",
        )
        hilo.start()
        _log_event("INFERENCE_STARTED_THREAD", inf_id, "hilo del grafo lanzado")

        # NOTA (fix bug "busy eterno"): el _GEN_LOCK NO se libera acá.
        # Si el cliente SSE se desconecta a mitad de stream, este generador
        # puede quedar huérfano indefinidamente (Starlette no lo cierra de
        # forma confiable y gen.close() puede fallar en un generador que
        # está siendo iterado desde un executor). El lock lo libera el HILO
        # del grafo (_correr_grafo_en_hilo), que SIEMPRE termina.
        try:
            while True:
                evento = q.get()
                yield evento
                if isinstance(evento, (DoneEvent, ErrorEvent)):
                    break
        except GeneratorExit:
            # El cliente SSE cerró la pestaña/tags o se desconectó; el
            # generador NO puede devuelvar más datos. Dejamos que muera.
            # El hilo del grafo sigue correndo y su finally libera _GEN_LOCK
            # cuando termina (sea COMPLETED, FAILED o CANCELLED).
            _log_event("INFERENCE_CLIENT_GONE", inf_id,
                       "generador SSE cerrado por el cliente")
            # Lo dejamos propagar: Python cierra el generador al terminar
            raise

    @staticmethod
    def get_status() -> Dict[str, Any]:
        """Obtiene estado del agente (modelo y host leídos de config.json)."""
        try:
            from core.config.config_manager import get_config_manager
            cfg = get_config_manager()

            base = {
                "name": "Aether",
                "model": cfg.get("MODELO", "?"),
                "ollama": cfg.get("OLLAMA_HOST", "?"),
                "busy": AetherService.ocupado(),
                "version": "2.1.0-revived",
                # Preferencias compartidas con la TUI (/effort y /agents).
                "effort": cfg.get("EFFORT", "medium"),
                "agent": cfg.get("AGENTE", "build"),
            }
            if not _AETHER_INITIALIZED:
                return {**base, "status": "uninitialized", "agent_status": "offline"}

            memoria_size = len(_AETHER_MEMORY.get("conversacion", [])) if _AETHER_MEMORY else 0
            return {
                **base,
                "status": "ready",
                "memory_size": memoria_size,
                "agent_status": "online",
            }
        except Exception as e:
            logger.error(f"Error obteniendo status: {e}")
            return {"name": "Aether", "status": "error", "agent_status": "offline"}


# =====================================================================
# EJECUCIÓN DEL GRAFO EN HILO (espejo de tui/engine_bridge.py)
# =====================================================================
def _correr_grafo_en_hilo(orden: str, q: "queue.Queue[Evento]",
                           imagenes: list[str] | None = None,
                           inf_id: str | None = None) -> None:
    from core.agent.streaming import set_cancel_thread_event
    inf_id = inf_id or f"inf-{int(time.time())}-{id(orden) % 10000:04d}"
    cancelled = False
    # Registrar el evento de cancelación de este hilo: streaming.request_cancel()
    # hilo_context-cancel_event, y el check ISP (ver abajo) lo lee.
    set_cancel_thread_event(_CANCEL_EVENT)
    writer = _QueueWriter(q)
    try:
        from core.agent.graph_builder import get_graph
        from core.agent.graph_state import crear_estado_inicial
        from core.memory.memory_manager import registrar_turno
        from core.agent.streaming import (
            InferenceCancelled,
            set_token_sink,
            set_reasoning_sink,
            clear_token_sink,
            clear_reasoning_sink,
        )
        from core.memory.consolidator import programar_consolidacion

        set_token_sink(lambda frag: q.put(TokenEvent(fragmento=frag)))

        # Canal de razonamiento: separado de tokens y de stdout (logs).
        # Además de streamear cada fragmento, se acumula el razonamiento del
        # turno para el DoneEvent (por si el cliente pierde fragmentos).
        razonamiento_turno: list[str] = []

        def _sink_reasoning(frag: str) -> None:
            razonamiento_turno.append(frag)
            q.put(ReasoningEvent(fragmento=frag))

        set_reasoning_sink(_sink_reasoning)

        # NO registrar el turno del usuario aquí: si lo hacemos antes de correr
        # el grafo, el context_manager lo verá en la memoria y lo incluirá en
        # el contexto de conversación. El modelo vería el mensaje del usuario
        # tanto en el historial como en el prompt actual, generando eco.
        # El turno del usuario se registra al final (ver bloque finally).

        grafo = get_graph()
        estado = crear_estado_inicial(orden, _AETHER_MEMORY, modo_autonomo=True,
                                      imagenes=imagenes)
        ultimo_estado: dict[str, Any] = dict(estado)

        with contextlib.redirect_stdout(writer):
            from core.agent.streaming import is_cancelled, InferenceCancelled
            for update in grafo.stream(estado, stream_mode="updates"):
                for nodo, delta in update.items():
                    if isinstance(delta, dict):
                        ultimo_estado.update(delta)
                    q.put(NodeUpdateEvent(nodo=nodo, delta=delta or {}))
                    # Chequeo cooperativo de cancelación entre pasos: si el
                    # frontend cerró la conexión (Stop, cierre de pestaña,
                    # pérdida de WebSocket), abortamos AHORA antes del
                    # siguiente paso del grafo. Sin esto el hilo queda corriendo
                    # como huérfano en background hasta terminar naturalmente,
                    # y nadie consume sus eventos. Core.agent.streaming
                    # request_cancel() marca el evento compartido almacenado
                    # acá (set_cancel_thread_event), y el contexto externo del
                    # servidor SSE/WS chequea is_cancelled().
                    if is_cancelled():
                        raise InferenceCancelled()
            writer.vaciar_residual()

        respuesta = ultimo_estado.get("final_response") or "Operación completada."
        programar_consolidacion(_AETHER_MEMORY)
        q.put(DoneEvent(respuesta=respuesta, reasoning="".join(razonamiento_turno)))

    except InferenceCancelled:
        cancelled = True
        writer.vaciar_residual()
        q.put(ErrorEvent(mensaje="Inferencia cancelada por el usuario."))
    except Exception as e:  # noqa: BLE001 — toda falla debe llegar a la UI
        writer.vaciar_residual()
        q.put(ErrorEvent(mensaje=f"{type(e).__name__}: {e}"))
    finally:
        try:
            from core.agent.streaming import clear_token_sink, clear_reasoning_sink
            clear_token_sink()
            clear_reasoning_sink()
        except Exception:
            pass
        # Registrar el turno del usuario AHORA (después de que el grafo terminó).
        # Así el context_manager no lo ve en el historial de la inferencia actual,
        # evitando que el modelo eco el mensaje del usuario.
        try:
            from core.memory.memory_manager import registrar_turno
            registrar_turno(_AETHER_MEMORY, "usuario", orden)
        except Exception:
            pass
        # Estado final: COMPLETED / FAILED / CANCELLED según cómo terminó.
        try:
            if cancelled:
                _emit_inference_cancelled(inf_id, "cancelada por el usuario/stop")
            elif _CURRENT_STATE.get("error"):
                _emit_inference_failed(inf_id, _CURRENT_STATE["error"])
            else:
                _emit_inference_completed(inf_id)
        except Exception:
            pass
        _CURRENT_STATE.pop("error", None)
        _CANCEL_EVENT.clear()
        # Emitir QUEUE_CONTINUING para que los logs muestren que el worker
        # global sigue adelante después de esta inferencia (aunque sea FAILED).
        _log_event("QUEUE_CONTINUING", inf_id,
                   f"_PROMPT_QUEUE.qsize()={_PROMPT_QUEUE.qsize()}")
        # Una sola inferencia a la vez: el lock se libera ACÁ, en el hilo,
        # que siempre termina (grafo completo, cancelado o con error).
        # Antes se liberaba en el generador consumidor (iter_eventos) y una
        # desconexión abrupta del cliente SSE lo dejaba tomado para siempre:
        # la Web UI quedaba en "ocupado" y todo chat nuevo rechazado.
        try:
            _GEN_LOCK.release()
        except RuntimeError:
            pass  # ya estaba libre (defensivo)
