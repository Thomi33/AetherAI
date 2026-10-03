"""
streaming.py — Canales dedicados para streaming del LLM.

- token sink: fragmentos de la RESPUESTA (content limpio, sin razonamiento).
- reasoning sink: fragmentos de RAZONAMIENTO (thinking) — canal separado,
  nunca se mezclan: quien consume el token sink (TUI, Web UI, SSE) recibe
  únicamente respuesta, y el razonamiento viaja por su propio evento.

La separación la hace el backend (_ThinkingStreamSplitter en graph_nodes
+ el campo nativo `thinking` de Ollama): los clientes NO parsean texto.
"""

from __future__ import annotations
import threading
from typing import Callable, Optional

_sink: Optional[Callable[[str], None]] = None
_reasoning_sink: Optional[Callable[[str], None]] = None
_cancel_event = threading.Event()


class InferenceCancelled(BaseException):
    """Control-flow exception that must not be swallowed by node fallbacks."""


def set_token_sink(callback: Callable[[str], None]) -> None:
    global _sink
    _sink = callback


def clear_token_sink() -> None:
    global _sink
    _sink = None


def set_reasoning_sink(callback: Callable[[str], None]) -> None:
    """Registra el consumidor de fragmentos de razonamiento (canal propio)."""
    global _reasoning_sink
    _reasoning_sink = callback


def clear_reasoning_sink() -> None:
    global _reasoning_sink
    _reasoning_sink = None


def reset_cancel() -> None:
    _cancel_event.clear()
    try:
        import core.agent.graph_nodes as g
        g._streaming_cancel_event = None
    except Exception:
        pass


def request_cancel() -> None:
    _cancel_event.set()


def is_cancelled() -> bool:
    # Primero el thread-local (si graph_nodes._streaming_cancel_event fue
    # registrado), porque el hilo del grafo lo sobreescribe. El global
    # queda como red de seguridad. Si el thread-local está seteado
    # (set_cancel_thread_event), tiene prioridad y el global no conta
    # (el generador SSE usa el set_cancel_thread_event para wiring).
    try:
        import core.agent.graph_nodes as g
        if getattr(g, "_streaming_cancel_event", None) is not None:
            if g._streaming_cancel_event.is_set():
                return True
            # Si thread-local está actuando como proxy del global, seguir
            # mirando el global (doble red de seguridad).
            pass
    except Exception:
        pass
    return _cancel_event.is_set()


def emit_token(fragmento: str) -> None:
    if _sink is not None and fragmento:
        _sink(fragmento)


def emit_reasoning(fragmento: str) -> None:
    """Emite un fragmento de razonamiento por su canal dedicado.

    Sin sink registrado es no-op (la TUI hoy no lo consume). NUNCA cae en
    stdout: por diseño el razonamiento no aparece en los logs.
    """
    if _reasoning_sink is not None and fragmento:
        _reasoning_sink(fragmento)


def set_cancel_thread_event(event: threading.Event) -> None:
    """Registra el evento de cancelación de ESTE hilo del grafo.

    streaming.request_cancel() marca `_cancel_event` global (_GEN_LOCK).
    graph_nodes.node_agent_loop llama is_cancelled() QUE mira
    graph_nodes._streaming_cancel_event (que esta función configura),
    no el global de streaming. Sin este puente, el endpoint SSE que hace
    request_cancel() no logra abortar el grafo corriendo.
    """
    import core.agent.graph_nodes as g
    g._streaming_cancel_event = event