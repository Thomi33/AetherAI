"""Prueba de regresión para el fix de MarkupError en DebugPanel.

Los strings que hoy rompen (ANSI, [SHELL], [INFO], JSON de tool call,
[✓] final, etc.) deben poder repaintarse sin levantar MarkupError.

IMPORTANTE: estos tests ejercitan el pipeline REAL de producción
(`tui.widgets.debug_panel._strip_ansi` + `_escapar_corchetes`), no una
copia local. La versión anterior de este archivo testeaba una réplica
del pipeline que quedó desincronizada del código real (falso positivo de
cobertura) y asumía comportamiento de `rich.markup.escape` que no se
cumple (rich SÍ escapa tags de cierre como `[/SHELL]` sin importar el
case). Eso hacía fallar `test_escape_textual_vs_rich` aunque el código
de producción era correcto.
"""

from __future__ import annotations

import pathlib
import sys

# Permite ejecutar este archivo directamente (`python tests/test_tui_markup_fix.py`)
# además de vía pytest (donde conftest.py ya agrega la raíz al path).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from textual.content import Content

from tui.widgets.debug_panel import _escapar_corchetes, _strip_ansi


def _sanitizar_log(texto: str) -> str:
    """Pipeline real de producción: DebugPanel.agregar_log aplica
    _strip_ansi() y luego _escapar_corchetes() antes de renderizar
    con markup=False (Static recibe un rich.text.Text plano)."""
    return _escapar_corchetes(_strip_ansi(texto))


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


def test_escapar_corchetes_escapa_todos_los_corchetes():
    """El helper de producción debe escapar TODO corchete, no solo los tags
    en minúscula que reconoce rich.markup.escape. Este es el requisito real:
    `[SHELL]`, `[/SHELL]`, `[INFO]`, `[✓]` deben quedar literales."""
    entrada = "[SHELL] comando[/SHELL] [INFO] [✓] sin-cerrar["
    salida = _escapar_corchetes(entrada)
    assert salida == (
        "\\[SHELL\\] comando\\[/SHELL\\] \\[INFO\\] \\[✓\\] sin-cerrar\\["
    )
    # No queda ningún corchete sin escapar.
    sin_invalidos = salida.replace("\\[", "").replace("\\]", "")
    assert "[" not in sin_invalidos and "]" not in sin_invalidos


def test_escapar_corchetes_duplica_backslashes_primero():
    """Los backslashes existentes se duplican ANTES de escapar corchetes,
    para no confundir escapes originales con los insertados."""
    assert _escapar_corchetes("\\") == "\\\\"
    assert _escapar_corchetes("\\[x]") == "\\\\\\[x\\]"


def test_ansi_se_quita_completamente():
    crudo = "\x1b[1;33mflatpak list 2>/dev/null\x1b[0m"
    assert _strip_ansi(crudo) == "flatpak list 2>/dev/null"


def test_pipeline_con_entradas_limite():
    """Edge cases: string vacío, solo espacios, solo ANSI, unicode."""
    assert _sanitizar_log("") == ""
    assert _sanitizar_log("   ") == "   "
    assert _sanitizar_log("\x1b[0m\x1b[1;33m") == ""
    # Unicode no ASCII no se altera.
    assert _sanitizar_log("órbítá ñ ✓") == "órbítá ñ ✓"


if __name__ == "__main__":
    test_escapar_corchetes_escapa_todos_los_corchetes()
    test_escapar_corchetes_duplica_backslashes_primero()
    test_ansi_se_quita_completamente()
    test_pipeline_con_entradas_limite()
    test_markup_despues_de_sanitizar_no_levanta_MarkupError()
    print("\nTODO OK")
