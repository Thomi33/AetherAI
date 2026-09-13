"""Prueba de regresión para el fix de MarkupError en DebugPanel.

Los strings que hoy rompen (ANSI, [SHELL], [INFO], JSON de tool call,
[✓] final, etc.) deben poder repaintarse sin levantar MarkupError.
"""

from __future__ import annotations

import re
from textual.content import Content
from textual.markup import escape as markup_escape

# Mismos utilidades que usa debug_panel.py para sanitizar logs
_RE_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _limpiar_ansi(texto: str) -> str:
    return _RE_ANSI.sub("", texto)


def _sanitizar_log(texto: str) -> str:
    """Mismo pipeline que DebugPanel.agregar_log (markup=False no aplica acá,
    pero el escape + limpieza ANSI basta para que Content.from_markup no falle)."""
    texto = _limpiar_ansi(texto)
    return markup_escape(texto)


NIVEL_PELIGROSO = {
    "ANSI + shell con redirect/grep":
        "\x1b[1;33mflatpak list --columns=application,name 2>/dev/null | grep -i sober\x1b[0m",
    "ANSI sin cerrar (como en el traceback)":
        "\x1b[1;33mflatpak list --columns=application,name 2>/dev/null | grep -i sober",
    "[SHELL]…[/SHELL] protocolo":
        "[SHELL] flatpak list --columns=application,name [/SHELL]",
    "[SHELL] sin cerrar + args largos":
        "[SHELL] flatpak list --columns=application,name 2>/dev/null | grep -i sober[/SHELL]",
    "prefijo [INFO] + contenido ANSI":
        "[INFO] ⚠️  [SHELL DETECTADO]: \x1b[1;33mflatpak list --columns=application,name 2>/dev/null | grep -i sober\x1b[0m",
    "[✓] Operación completada (REAL_STATES / node_finalize)":
        "[✓] Operación completada",
    "JSON de tool calling (fuga al chat)":
        '{"name":"shell","arguments":{"instruccion":"flatpak list --columns=application,name 2>/dev/null | grep -i sober"}}',
    "Prefijo de chat [bold] intentado (hoy se ve literal)":
        "[bold]ornith-1.5:9b[/bold]",
    "[•] estados reales (planner, context_manager, etc.)":
        "[•] Ejecutando herramienta: shell",
    "Chain tool call + pipeline shell":
        "[SHELL] cat -n /home/thomi/mi_proyecto_crew/tui/app.py | head -n 40 && flatpak list --columns=application,name 2>/dev/null | grep -i sober[/SHELL]",
}


def test_markup_despues_de_sanitizar_no_levanta_MarkupError():
    """Cada nivel peligroso, sanitizado, debe poder convertirse a Content
    sin MarkupError (que es lo que provocaba el crash en _repintar)."""
    for nombre, peligroso in NIVEL_PELIGROSO.items():
        sanitizado = _sanitizar_log(peligroso)
        # Marcamos como `markup=True` explícito para forzar el parseo que
        # antes crashaba en visualize(..., markup=True).
        try:
            Content.from_markup(sanitizado)
        except Exception as e:
            msg = f"FAIL [{nombre}]: {type(e).__name__}: {e}"
            print(msg)
            raise AssertionError(msg) from e
        else:
            print(f"OK [{nombre}]: {sanitizado[:60]!r}...")


def test_escape_textual_vs_rich():
    """textual.markup.escape escapa cualquier [tag] (mayúsculas incluidas),
    a diferencia de rich.markup.escape que solo escapa minúsculas."""
    from rich.markup import escape as rich_escape

    pecho = "[SHELL] comando[/SHELL]"
    assert rich_escape(pecho) == pecho, "rich.escape NO escapa [SHELL] (mayúsculas)"
    assert markup_escape(pecho) != pecho, "textual.escape SÍ escapa [SHELL]"
    assert "\\" in markup_escape(pecho)


def test_ansi_se_quita_completamente():
    crudo = "\x1b[1;33mflatpak list 2>/dev/null\x1b[0m"
    assert _limpiar_ansi(crudo) == "flatpak list 2>/dev/null"


if __name__ == "__main__":
    test_escape_textual_vs_rich()
    test_ansi_se_quita_completamente()
    test_markup_despues_de_sanitizar_no_levanta_MarkupError()
    print("\nTODO OK")
