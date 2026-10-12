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
from datetime import datetime
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

# ── Sesión live de la inferencia (re-attach de la Web UI tras refresh) ──
# Buffer append-only con TODOS los eventos de la inferencia ACTIVA. Una tab
# recargada (F5 / reinicio del navegador) puede re-engancharse a mitad de
# turno y re-emitirlos desde seq 0: la UI se restaura tal como estaba
# (tokens en vivo, plan, razonamiento). ADITIVO: el consumidor original
# (SSE /api/chat/stream o /ws/chat) sigue recibiendo por su cola privada
# exactamente igual que antes (_publish alimenta ambos destinos).
_LIVE_SESSION: dict | None = None
_LIVE_LOCK = threading.Lock()
# Una sesión terminada se conserva 24 h (el navegador puede volver HORAS
# después y la tab recargada igual recupera la respuesta completa del
# turno; la sesión vieja se reemplaza sola cuando arranca la próxima
# inferencia, así que el TTL es solo higiene de RAM).
_LIVE_TTL_SEG = 24 * 60 * 60


def _modelo_con_reasoning() -> bool:
    """Razonamiento nativo del MODELO (ÚNICA fuente de verdad: la usan
    el 'start' del SSE /chat/stream y la sesión live del turno).

    DELEGADO en core.config.capabilities.modelo_con_reasoning (matriz
    de capacidades): primero las capacidades REALES de Ollama
    (/api/show → capabilities: "thinking", verificado contra Ollama
    v0.40.0); el heurístico por nombre queda SOLO como fallback cuando
    Ollama no reporta capabilities (caída / versión vieja / modelo no
    instalado). Firma y contrato sin cambios (bool).
    """
    try:
        from core.config.capabilities import modelo_con_reasoning
        return bool(modelo_con_reasoning())
    except Exception:  # noqa: BLE001 — nunca frenar un turno por esto
        return False


def _live_reset(inf_id: str, orden: str, razonamiento: bool) -> None:
    """Crea la sesión live de la inferencia que arranca (reemplaza la previa)."""
    global _LIVE_SESSION
    with _LIVE_LOCK:
        _LIVE_SESSION = {
            "inf_id": inf_id,
            "preview": (orden or "")[:120] + ("…" if len(orden or "") > 120 else ""),
            "inicio": time.time(),
            "seq": 0,
            "eventos": [],       # [(seq, Evento)] append-only
            "done": False,
            "fin_ts": 0.0,
            "estado": "RUNNING",
            "reasoning": bool(razonamiento),
        }


def _live_fin(estado: str, inf_id: str | None = None) -> None:
    """Marca la sesión live como terminada (COMPLETED/FAILED/CANCELLED)."""
    with _LIVE_LOCK:
        s = _LIVE_SESSION
        if s is None or s["done"]:
            return
        if inf_id is not None and s["inf_id"] != inf_id:
            return  # la sesión ya es de otro turno (defensivo)
        s["done"] = True
        s["estado"] = estado
        s["fin_ts"] = time.time()


def _live_purge_si_vencida() -> None:
    """Purga la sesión TERMINADA si venció el TTL. El llamador debe
    sostener _LIVE_LOCK (no la toma acá: reentrancia)."""
    global _LIVE_SESSION
    s = _LIVE_SESSION
    if (s is not None and s.get("done")
            and time.time() - s.get("fin_ts", 0.0) > _LIVE_TTL_SEG):
        _LIVE_SESSION = None


def _publish(q: "queue.Queue", evento) -> None:
    """Única vía de emisión del hilo del grafo: cola del consumidor ACTUAL
    (comportamiento de siempre) + buffer secuenciado de la sesión live
    (copia para clientes que se reconecten después del refresh)."""
    q.put(evento)
    with _LIVE_LOCK:
        s = _LIVE_SESSION
        if s is not None and not s["done"]:
            s["seq"] += 1
            s["eventos"].append((s["seq"], evento))


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

    def __init__(self, q: "queue.Queue[Evento]", publicar=None):
        self._q = q
        # publicar(evento): vía de emisión compartida (cola del consumidor +
        # sesión live). Por defecto solo la cola (compat con otros usos).
        self._publicar = publicar or (lambda ev: q.put(ev))
        self._buffer = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buffer += s
        while "\n" in self._buffer:
            linea, self._buffer = self._buffer.split("\n", 1)
            texto = _RE_ANSI.sub("", linea).rstrip()
            if texto:
                self._publicar(StdoutLineEvent(texto=texto))
        return len(s)

    def flush(self) -> None:
        pass

    def vaciar_residual(self) -> None:
        resto = _RE_ANSI.sub("", self._buffer)
        if len(resto) > 500:
            resto = resto[:500] + "…"
        if resto.strip():
            self._publicar(StdoutLineEvent(texto=resto))
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
                # Sesión única por arranque del servicio (igual que la TUI):
                # sin sesion_id, los turnos del backend quedan sin sesión y
                # no aparecen en /sesiones ni en el historial por tema/sesión.
                if not _AETHER_MEMORY.get("sesion_id"):
                    import uuid
                    _AETHER_MEMORY["sesion_id"] = uuid.uuid4().hex

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

        # FIX reloj (Plan B): hora REAL de llegada de la orden al runtime.
        # Viaja por el estado del grafo (AetherState.fecha_envio) hasta
        # node_finalize, que estampa el turno del usuario con ESTA fecha —
        # antes quedaba con la hora del finalize (minutos después en
        # inferencias largas: el bug del "reloj detenido en 7:43 pm").
        fecha_envio = datetime.now().isoformat()

        # ── CICLO DE VIDA: QUEUED → RUNNING/FAILED/CANCELLED ──────────
        if not _GEN_LOCK.acquire(blocking=False):
            _emit_inference_queued(inf_id, "inferencia activa; en cola FIFO")
            _log_event("INFERENCE_QUEUED", inf_id, "encolada (lock ocupado)")
            yield ErrorEvent(
                mensaje="Ya hay una inferencia en curso. Esperá que termine o enviá /stop."
            )
            return

        _emit_inference_start(inf_id)
        # Sesión live del turno (re-attach de la Web UI tras refresh): buffer
        # append-only de los mismos eventos que van a la cola del consumidor.
        _live_reset(inf_id, orden, _modelo_con_reasoning())
        try:
            AetherService._asegurar_inicializado()
        except Exception as e:
            _emit_inference_failed(inf_id, f"init failed: {e}")
            _live_fin("FAILED", inf_id)
            _GEN_LOCK.release()
            yield ErrorEvent(mensaje=f"Aether no pudo inicializarse: {type(e).__name__}: {e}")
            return

        q: "queue.Queue[Evento]" = queue.Queue()
        hilo = threading.Thread(
            target=_correr_grafo_en_hilo,
            args=(orden, q, imagenes, inf_id, fecha_envio),
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
                # Ciclo de vida de la inferencia (re-attach de la Web UI tras
                # refresh; el detalle del turno vive en /api/chat/live/status).
                "inf_estado": _CURRENT_STATE.get("estado"),
                "inf_id": _CURRENT_STATE.get("inf_id"),
                "inf_inicio": _CURRENT_STATE.get("inicio"),
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


    @staticmethod
    def live_snapshot() -> Dict[str, Any]:
        """Estado de la sesión live del turno (re-attach de la Web UI).

        active=True → inferencia en curso; done=True → terminada pero aún
        bufferizada (TTL de 24 h: la sesión vieja se reemplaza sola con la
        próxima inferencia). El frontend lo consulta al boot tras un
        refresh/reinicio del navegador para decidir si re-attachearse.
        """
        with _LIVE_LOCK:
            _live_purge_si_vencida()
            s = _LIVE_SESSION
            if s is None:
                return {"active": False, "done": False, "seq": 0, "eventos": 0}
            return {
                "active": not s["done"],
                "done": s["done"],
                "inf_id": s["inf_id"],
                "estado": s["estado"],
                "inicio": s["inicio"],
                "fin_ts": s["fin_ts"],
                "seq": s["seq"],
                "eventos": len(s["eventos"]),
                "preview": s["preview"],
                "reasoning": s["reasoning"],
            }

    @staticmethod
    def iter_live_eventos(since: int = 0) -> Iterator[tuple[int, Any]]:
        """Re-emite la sesión live desde seq > since, en vivo hasta done.

        Generador síncrono (se consume vía run_in_executor como el grafo):
        primero drena el buffer existente (replay del turno hasta el
        momento del refresh) y después sigue los eventos nuevos (poll corto
        de 100 ms) hasta que la sesión termina. Mantiene su propia
        referencia al dict: la purga por TTL no puede quitárselo de arriba.
        """
        with _LIVE_LOCK:
            _live_purge_si_vencida()
            s = _LIVE_SESSION
        if s is None:
            return
        indice = 0
        while True:
            total = len(s["eventos"])
            while indice < total:
                seq, evento = s["eventos"][indice]
                indice += 1
                if seq > since:
                    yield seq, evento
            # El evento terminal (Done/Error) SIEMPRE está en el buffer antes
            # de done=True (lo publica el hilo antes de su finally), así que
            # el consumidor jamás se queda esperando el final que ya pasó.
            if s["done"] and indice >= total:
                return
            time.sleep(0.1)


# =====================================================================
# EJECUCIÓN DEL GRAFO EN HILO (espejo de tui/engine_bridge.py)
# =====================================================================
def _correr_grafo_en_hilo(orden: str, q: "queue.Queue[Evento]",
                           imagenes: list[str] | None = None,
                           inf_id: str | None = None,
                           fecha_envio: str = "") -> None:
    from core.agent.streaming import set_cancel_thread_event
    inf_id = inf_id or f"inf-{int(time.time())}-{id(orden) % 10000:04d}"
    cancelled = False
    # Registrar el evento de cancelación de este hilo: streaming.request_cancel()
    # hilo_context-cancel_event, y el check ISP (ver abajo) lo lee.
    set_cancel_thread_event(_CANCEL_EVENT)
    # Emisión del hilo: cola del consumidor actual + sesión live (re-attach).
    writer = _QueueWriter(q, lambda ev: _publish(q, ev))
    ultimo_estado: dict[str, Any] = {}
    llego_finalize = False
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
        from core.memory.memory_manager import programar_consolidacion

        set_token_sink(lambda frag: _publish(q, TokenEvent(fragmento=frag)))

        # Canal de razonamiento: separado de tokens y de stdout (logs).
        # Además de streamear cada fragmento, se acumula el razonamiento del
        # turno para el DoneEvent (por si el cliente pierde fragmentos).
        razonamiento_turno: list[str] = []

        def _sink_reasoning(frag: str) -> None:
            razonamiento_turno.append(frag)
            _publish(q, ReasoningEvent(fragmento=frag))

        set_reasoning_sink(_sink_reasoning)

        # NO registrar el turno del usuario acá: si lo hacemos antes de correr
        # el grafo, el context_manager lo verá en la memoria y lo incluirá en
        # el contexto de conversación (eco). Ahora lo registra node_finalize
        # —antes que la respuesta y con tema/sesión— al completarse la
        # inferencia; el finally solo cubre los caminos sin finalize.

        grafo = get_graph()
        estado = crear_estado_inicial(orden, _AETHER_MEMORY, modo_autonomo=True,
                                      imagenes=imagenes, fecha_envio=fecha_envio)
        ultimo_estado = dict(estado)

        with contextlib.redirect_stdout(writer):
            from core.agent.streaming import is_cancelled, InferenceCancelled
            for update in grafo.stream(estado, stream_mode="updates"):
                for nodo, delta in update.items():
                    if isinstance(delta, dict):
                        ultimo_estado.update(delta)
                    if nodo == "finalize":
                        llego_finalize = True
                    _publish(q, NodeUpdateEvent(nodo=nodo, delta=delta or {}))
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

        from core.services.graph_service import respuesta_final_o_informe
        respuesta = respuesta_final_o_informe(ultimo_estado)
        programar_consolidacion(_AETHER_MEMORY)
        _publish(q, DoneEvent(respuesta=respuesta, reasoning="".join(razonamiento_turno)))

    except InferenceCancelled:
        cancelled = True
        writer.vaciar_residual()
        _publish(q, ErrorEvent(mensaje="Inferencia cancelada por el usuario."))
    except Exception as e:  # noqa: BLE001 — toda falla debe llegar a la UI
        writer.vaciar_residual()
        _publish(q, ErrorEvent(mensaje=f"{type(e).__name__}: {e}"))
    finally:
        try:
            from core.agent.streaming import clear_token_sink, clear_reasoning_sink
            clear_token_sink()
            clear_reasoning_sink()
        except Exception:
            pass
        # Fallback del turno del usuario: SOLO si el grafo no llegó a finalize
        # (cancelación, error, END directo). Si llegó, node_finalize ya lo
        # registró —antes que la respuesta y con tema/sesión—; no duplicar.
        if not llego_finalize:
            try:
                from core.memory.memory_manager import registrar_turno
                registrar_turno(
                    _AETHER_MEMORY, "usuario", orden,
                    sesion_id=_AETHER_MEMORY.get("sesion_id", ""),
                    tema=ultimo_estado.get("tema_actual", ""),
                    fecha=fecha_envio or None,
                )
            except Exception:
                pass
        # Estado final: COMPLETED / FAILED / CANCELLED según cómo terminó.
        try:
            if cancelled:
                _emit_inference_cancelled(inf_id, "cancelada por el usuario/stop")
                _live_fin("CANCELLED", inf_id)
            elif _CURRENT_STATE.get("error"):
                _emit_inference_failed(inf_id, _CURRENT_STATE["error"])
                _live_fin("FAILED", inf_id)
            else:
                _emit_inference_completed(inf_id)
                _live_fin("COMPLETED", inf_id)
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
