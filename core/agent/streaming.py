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


def request_cancel() -> None:
    _cancel_event.set()


def is_cancelled() -> bool:
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