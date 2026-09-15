"""
response_cleaner.py — Limpieza de la respuesta final del agente.

Compartida entre la TUI (tui/app.py) y el backend web
(backend/api/routes/chat.py) para que ambas interfaces muestren el MISMO
texto final, sin artefactos internos del tool-calling.

Historial: esta lógica vivía embebida en tui/app.py; se extrajo acá al
conectar la Web UI al mismo contrato de salida que la TUI.
"""

from __future__ import annotations

import re

# Códigos ANSI de color/cursor que el streaming puede arrastrar.
_RE_STREAM_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def limpiar_respuesta_chat(respuesta: str) -> str:
    """Quita protocolo interno de tool-calling de la respuesta final del LLM.

    El agente necesita emitir [SHELL]...[/SHELL] y JSON de tool call como
    pasos INTERMEDIOS (los ejecuta node_shell), pero la respuesta FINAL que
    va al chat debería ser lenguaje natural limpio. A veces el modelo deja
    escapar esos artefactos hasta la síntesis final; esto los filtra como
    defensa sin tocar el flujo de ejecución.
    """
    s = _RE_STREAM_ANSI.sub("", str(respuesta or ""))
    # Bloques de protocolo [SHELL]...[/SHELL] que sobrevivieron a la síntesis
    s = re.sub(r"\[SHELL\].*?\[/SHELL\]", "", s, flags=re.DOTALL | re.IGNORECASE)
    # JSON de tool call suelto {"name":"shell","arguments":{...}}
    s = re.sub(
        r'\{[^{}]*"name"\s*:\s*"shell"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\}',
        "", s, flags=re.DOTALL,
    )
    # Prefijos internos de una sola línea ([MCP]..., [PLAN EXECUTOR]...)
    s = re.sub(r"(?m)^\[(MCP|PLAN EXECUTOR|INFO|DEBUG|SHELL DETECTADO)[^\n]*\n?", "", s)
    # Colapsar líneas en blanco múltiples dejadas por la limpieza
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return s or "(sin respuesta textual)"
