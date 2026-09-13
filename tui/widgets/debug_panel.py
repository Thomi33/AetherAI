"""
Widget Debug Panel para mostrar logs de ejecución en la TUI.
Muestra mensajes de herramientas, errores y otros eventos del motor.
"""

from __future__ import annotations

import re

from textual.widgets import Static
from rich.text import Text

# ANSI escape sequences que el motor inyecta en los logs (p.ej.
# "\x1b[1;33mcomando\x1b[0m") y que podrían confundir el parser de
# markup de Textual si el widget estuviera en markup=True. Como el panel
# se renderiza con markup=False (ver abajo), esto es solo una defensa extra.
_RE_ANSI = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def _strip_ansi(texto: str) -> str:
    """Quita códigos ANSI (color, cursor, etc.) del texto."""
    return _RE_ANSI.sub('', texto)


def _escapar_corchetes(texto: str) -> str:
    """Escapa TODOS los corchetes `[` `]` para renderizado literal.

    `rich.markup.escape` solo escapa tags que empiecen en minúscula
    (`[a-z#/@]...`), dejando `[SHELL]`, `[INFO]`, `[✓]`, `[bold]`, etc.
    sin tocar. Este helper escapa cualquier corchete para que el texto
    se muestre literal en el panel (aunque se renderice sin markup).
    """
    return (texto
            .replace(chr(92), chr(92) + chr(92))
            .replace('[', chr(92) + '[')
            .replace(']', chr(92) + ']'))  # noqa: E501


class DebugPanel(Static):
    """Panel que muestra los logs de debug y ejecución.

    Se inicializa con markup=False para que los logs del motor (protocolo
    [SHELL]...[/SHELL], estados [✓], [INFO], JSON de tool call, etc.)
    nunca provoquen MarkupError en el render.
    """

    def __init__(self, **kwargs):
        kwargs['markup'] = False  # [FIX] markup=False evita MarkupError en logs que
                                   # contienen corchetes literales ([SHELL], [INFO], [✓])
                                   # o códigos ANSI del shell. Sin esto, Static.update(texto)
                                   # intenta parsear el texto como markup y falla.
        super().__init__(**kwargs)
        self._debug_logs: list[str] = []
        self._max_lines = 8  # debe coincidir con la altura visible del panel

    def agregar_log(self, mensaje=None, tipo='info'):  # noqa: A003 — API pública legacy
        """Agrega una línea al panel de debug."""
        if mensaje is not None:
            texto = _strip_ansi(str(mensaje))
            texto = _escapar_corchetes(texto)
            prefix = f'[{tipo.upper()}]' if tipo else ''
            texto = f'{prefix} {texto}'
            self._debug_logs.append(texto)
            self._repintar()

    def _repintar(self):
        """Reconstruye el contenido visual del panel."""
        if len(self._debug_logs) > self._max_lines:
            self._debug_logs = self._debug_logs[-self._max_lines:]
        texto = '\n'.join(self._debug_logs)
        # Static tiene markup=False (ver __init__), pero pasar un Text
        # construido manualmente es defensa extra: Static.render() lo pasa
        # por Content.from_rich_text y salta el parser de markup.
        # (Text NO acepta kwarg `markup` — solo texto plano.)
        self.update(Text(texto))

    def reset(self):
        self._debug_logs.clear()
        self._repintar()
