"""
slash_completer.py — Autocompletado de comandos al escribir '/'.

Aparece flotando sobre el input cuando el texto empieza con '/'.
Se navega con flechas arriba/abajo y se confirma con Tab o Enter.
Desaparece con Escape o cuando el texto deja de empezar con '/'.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label, ListItem, ListView
from textual.reactive import reactive

SLASH_COMMANDS = [
    ("/agents",    "Switch agent"),
    ("/debug",     "View debug info"),
    ("/effort",    "Set effort level"),
    ("/get",       "Get config value"),
    ("/help",      "Show help"),
    ("/historial", "Ver sesión archivada"),
    ("/mcps",      "Toggle MCPs"),
    ("/memory",    "Editar resumen de memoria"),
    ("/models",    "Switch model"),
    ("/play-roblox", "Iniciar Roblox; opcionalmente usar google u ollama"),
    ("/prompt",    "System prompt sin tocar código (extra/override/sintesis)"),
    ("/new",       "New session"),
    ("/sesiones",  "List sessions"),
    ("/set",       "Set config value"),
    ("/themes",    "Switch theme"),
]

CSS = """
SlashCompleter {
    dock: bottom;
    layer: overlay;
    width: 50;
    height: auto;
    max-height: 14;
    background: $surface;
    border: solid $primary;
    display: none;
    offset: 0 -1;
}

SlashCompleter ListView {
    border: none;
    background: $surface;
    height: auto;
    padding: 0;
}

SlashCompleter ListItem {
    padding: 0 1;
    background: $surface;
    color: $text-muted;
}

SlashCompleter ListItem.--highlight {
    background: $primary-darken-2;
    color: $text;
}
"""


class SlashCompleter(Widget):
    """Widget de autocompletado flotante para slash commands."""

    DEFAULT_CSS = CSS

    visible_completions: reactive[list[tuple[str, str]]] = reactive([], layout=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._matches: list[tuple[str, str]] = []

    def compose(self) -> ComposeResult:
        yield ListView(id="sc-list")

    async def update_query(self, text: str) -> None:
        """
        Llamar desde app.py cada vez que cambia el input.

        FIX (crash slash commands): en Textual 8.x, ListView.clear() retorna
        AwaitRemove y ListView.append() retorna AwaitMount — son ASYNC. Sin
        `await`, el árbol de widgets queda a medio montar y la app crashea al
        escribir '/' + cualquier opción. Ahora se await tanto clear como
        append.
        """
        if not text.startswith("/"):
            self.display = False
            return

        query = text.lower()
        # Dedup por comando: ids derivan de cmd (sc-<cmd>) y duplicados
        # rompen el DOM con DuplicateIds en Textual 8.x.
        seen: set[str] = set()
        self._matches = [
            (cmd, desc)
            for cmd, desc in SLASH_COMMANDS
            if cmd.startswith(query) and not (cmd in seen or seen.add(cmd))
        ]

        lv = self.query_one("#sc-list", ListView)
        await lv.clear()

        if not self._matches:
            self.display = False
            return

        for cmd, desc in self._matches:
            await lv.append(ListItem(
                Label(f"[bold]{cmd}[/bold]  [dim]{desc}[/dim]"),
                id=f"sc-{cmd.lstrip('/')}",
            ))
        lv.index = 0
        self.display = True

    def get_selected(self) -> str | None:
        """Devuelve el comando seleccionado actualmente (sin confirmar)."""
        lv = self.query_one("#sc-list", ListView)
        if lv.highlighted_child:
            item_id = lv.highlighted_child.id or ""
            return "/" + item_id.replace("sc-", "", 1)
        return None

    def confirm(self) -> str | None:
        """Confirma la selección y oculta el widget. Retorna el comando."""
        cmd = self.get_selected()
        self.display = False
        return cmd

    def move_up(self) -> None:
        lv = self.query_one("#sc-list", ListView)
        if lv.index is not None and lv.index > 0:
            lv.index -= 1

    def move_down(self) -> None:
        lv = self.query_one("#sc-list", ListView)
        if lv.index is not None and lv.index < len(self._matches) - 1:
            lv.index += 1

    def hide(self) -> None:
        self.display = False
