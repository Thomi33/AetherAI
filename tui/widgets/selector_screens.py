"""
selector_screens.py — Pantallas modales al estilo OpenCode para la TUI de Aether.

Provee:
  - ModelSelectorScreen   : lista modelos de Ollama en vivo (sin reiniciar)
  - AgentSelectorScreen   : selector de agente (build / plan / etc.)
  - ThemeSelectorScreen   : selector de tema visual
  - McpSelectorScreen     : toggle de servidores MCP
  - EffortSelectorScreen  : nivel de esfuerzo del modelo (low/medium/high/max)
  - CommandPaletteScreen  : paleta de comandos rápidos (Ctrl+P)
  - DirAuthScreen         : gate de autorización del directorio de trabajo

Cada Screen sigue el mismo patrón:
  1. Input de búsqueda filtrante en tiempo real.
  2. Lista de opciones navegable con flechas / Enter.
  3. Escape para cerrar sin cambiar nada.
  4. Ítem activo marcado con bullet naranja (●).
"""

from __future__ import annotations

import re
import subprocess
from typing import Callable

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Button, Input, Label, ListItem, ListView, Static, TextArea
from textual.containers import Vertical, Horizontal


def _safe_id(item: str) -> str:
    """Sanitiza un string para usarlo como id válido de Textual
    (solo letras, números, guiones y underscores; no puede empezar con número)."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "-", item)
    if safe and safe[0].isdigit():
        safe = f"m-{safe}"
    return safe


# ─── Paleta de colores compartida (coherente con status_bar / plan_panel) ───
_CSS_BASE = """
Screen {
    align: center middle;
    background: rgba(0,0,0,0.6);
}

#modal-box {
    width: 50;
    background: $surface;
    border: solid $primary;
    padding: 0 1;
}

#modal-title {
    text-style: bold;
    padding: 0 0 0 0;
    color: $text;
    border-bottom: solid $primary-darken-2;
    height: 1;
    margin-bottom: 1;
}

#modal-search {
    border: none;
    background: $surface;
    padding: 0;
    margin: 0 0 1 0;
    height: 1;
    color: $text;
}

#modal-search:focus {
    border: none;
}

#modal-list {
    border: none;
    background: $surface;
    height: auto;
    max-height: 20;
    padding: 0;
}

#modal-list > ListItem {
    padding: 0 1;
    background: $surface;
    color: $text-muted;
}

#modal-list > ListItem.--highlight {
    background: $primary-darken-2;
    color: $text;
}

#modal-list > ListItem.active-item > Label {
    color: $warning;
}

#modal-footer {
    height: 1;
    margin-top: 1;
    border-top: solid $primary-darken-2;
    color: $text-muted;
    padding: 0;
}
"""


def _fetch_ollama_models() -> list[str]:
    """Llama a 'ollama list' y devuelve los nombres de modelos disponibles."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.strip().splitlines()
        models = []
        for line in lines[1:]:  # saltar cabecera
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models or ["ornith:9b"]
    except Exception:
        return ["ornith:9b"]


# ─────────────────────────────────── BASE SCREEN ──────────────────────────────────────

class _BaseSelectorScreen(Screen):
    """Base para todas las pantallas de selección."""

    DEFAULT_CSS = _CSS_BASE

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cerrar", show=False),
        Binding("enter", "confirm_selection", "Confirmar", show=False),
    ]

    def __init__(
        self,
        title: str,
        items: list[str],
        current: str | None = None,
        footer_hint: str = "enter confirmar  esc cerrar",
        on_select: Callable[[str], None] | None = None,
    ):
        super().__init__()
        self._title = title
        self._all_items = items
        self._filtered = list(items)
        self._current = current
        self._footer_hint = footer_hint
        self._on_select = on_select

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static(self._title, id="modal-title")
            yield Input(placeholder="Search", id="modal-search")
            lv = ListView(id="modal-list")
            yield lv
            yield Static(self._footer_hint, id="modal-footer")

    def on_mount(self) -> None:
        self.run_worker(self._rebuild_list(self._all_items), exclusive=True)
        self.query_one("#modal-search", Input).focus()

    async def _rebuild_list(self, items: list[str]) -> None:
        lv = self.query_one("#modal-list", ListView)
        await lv.clear()
        for idx, item in enumerate(items):
            label_text = f"● {item}" if item == self._current else f"  {item}"
            # Prefijo con el índice para garantizar unicidad incluso si dos items
            # sanitizan al mismo string (ej. 'a:b' y 'a.b' -> 'a-b').
            li = ListItem(Label(label_text), id=f"item-{idx}-{_safe_id(item)}")
            if item == self._current:
                li.add_class("active-item")
            await lv.append(li)   # FIX: append es async en Textual 8.x (AwaitMount)
        if items:
            lv.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "modal-search":
            query = event.value.lower()
            filtered = [i for i in self._all_items if query in i.lower()]
            self._filtered = filtered
            self.run_worker(self._rebuild_list(filtered), exclusive=True)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def action_confirm_selection(self) -> None:
        lv = self.query_one("#modal-list", ListView)
        if lv.highlighted_child is not None:
            idx = lv.index
            if 0 <= idx < len(self._filtered):
                selected = self._filtered[idx]
                if self._on_select:
                    self._on_select(selected)
                self.dismiss(selected)
        else:
            self.dismiss(None)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        idx = self.query_one("#modal-list", ListView).index
        if 0 <= idx < len(self._filtered):
            selected = self._filtered[idx]
            if self._on_select:
                self._on_select(selected)
            self.dismiss(selected)


# ─────────────────────────────────── MODEL SELECTOR ──────────────────────────────────

class ModelSelectorScreen(_BaseSelectorScreen):
    """Selecciona un modelo de Ollama disponible en el sistema (sin reiniciar)."""

    def __init__(self, current_model: str, on_select: Callable[[str], None] | None = None):
        models = _fetch_ollama_models()
        super().__init__(
            title="Select model",
            items=models,
            current=current_model,
            footer_hint="enter confirmar  esc cerrar  (modelos de ollama list)",
            on_select=on_select,
        )


# ─────────────────────────────────── AGENT SELECTOR ──────────────────────────────────

AGENTS = [
    ("build", "native"),
    ("plan",  "native"),
]


class AgentSelectorScreen(Screen):
    """Selector de agente al estilo OpenCode."""

    DEFAULT_CSS = _CSS_BASE

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cerrar", show=False),
        Binding("enter", "confirm_selection", "Confirmar", show=False),
    ]

    def __init__(self, current_agent: str = "build", on_select: Callable[[str], None] | None = None):
        super().__init__()
        self._current = current_agent
        self._on_select = on_select
        self._items = AGENTS

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static("Select agent", id="modal-title")
            yield Input(placeholder="Search", id="modal-search")
            lv = ListView(id="modal-list")
            yield lv
            yield Static("enter confirmar  esc cerrar", id="modal-footer")

    def on_mount(self) -> None:
        self.run_worker(self._rebuild(), exclusive=True)
        self.query_one("#modal-search", Input).focus()

    async def _rebuild(self, query: str = "") -> None:
        lv = self.query_one("#modal-list", ListView)
        await lv.clear()
        for name, kind in self._items:
            if query and query not in name:
                continue
            bullet = "● " if name == self._current else "  "
            li = ListItem(
                Label(f"{bullet}[bold]{name}[/bold] [dim]{kind}[/dim]"),
                id=f"agent-{name}"
            )
            if name == self._current:
                li.add_class("active-item")
            await lv.append(li)   # FIX: append es async en Textual 8.x (AwaitMount)
        if lv.children:
            lv.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "modal-search":
            self.run_worker(self._rebuild(event.value.lower()), exclusive=True)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def action_confirm_selection(self) -> None:
        lv = self.query_one("#modal-list", ListView)
        if lv.highlighted_child:
            item_id = lv.highlighted_child.id or ""
            name = item_id.replace("agent-", "")
            if self._on_select:
                self._on_select(name)
            self.dismiss(name)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        name = item_id.replace("agent-", "")
        if self._on_select:
            self._on_select(name)
        self.dismiss(name)


# ─────────────────────────────────── THEME SELECTOR ──────────────────────────────────

THEMES = [
    "cobalt2", "cursor", "dracula", "everforest", "flexoki", "github",
    "gruvbox", "kanagawa", "lucent-orng", "material", "matrix",
    "mercury", "monokai", "nightowl", "nord", "one-dark", "opencode",
    "orng", "osaka-jade", "palenight", "rosepine", "solarized",
    "synthwave84", "system", "tokyonight", "vercel", "vesper", "zenburn",
    # Temas nativos de Textual
    "textual-dark", "textual-light",
]


class ThemeSelectorScreen(_BaseSelectorScreen):
    """Selector de tema visual al estilo OpenCode."""

    def __init__(self, current_theme: str = "system", on_select: Callable[[str], None] | None = None):
        super().__init__(
            title="Themes",
            items=THEMES,
            current=current_theme,
            footer_hint="enter aplicar  esc cerrar",
            on_select=on_select,
        )


# ─────────────────────────────────── MCP SELECTOR ──────────────────────────────────

class McpSelectorScreen(Screen):
    """Toggle de servidores MCP (vista mergeada: plantilla + capa local)."""

    DEFAULT_CSS = _CSS_BASE + """
    .mcp-enabled  { color: $success; }
    .mcp-disabled { color: $text-muted; }
    """

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cerrar", show=False),
        Binding("space", "toggle_current", "Toggle", show=False),
        Binding("enter", "confirm", "Confirmar", show=False),
    ]

    def __init__(self, on_done: Callable[[dict], None] | None = None):
        super().__init__()
        self._on_done = on_done
        self._states: dict[str, bool] = {}  # name → enabled
        self._names: list[str] = []
        self._filtered: list[str] = []
        self._load_servers()

    def _load_servers(self) -> None:
        """Carga la vista mergeada (plantilla versionada ← overrides locales
        con tokens reales/servers custom). El merge SIEMPRE devuelve el
        formato {nombre: cfg}, así que los formatos legados de lista o
        {"servers": [...]} ya no hacen falta acá."""
        try:
            from core.tools.mcp_client import cargar_merge
            data = cargar_merge()
        except Exception:
            data = {}
        if isinstance(data, dict):
            for name, cfg in data.items():
                self._states[name] = (
                    cfg.get("enabled", True) if isinstance(cfg, dict) else True
                )
                self._names.append(name)
        self._filtered = list(self._names)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static("MCPs", id="modal-title")
            yield Input(placeholder="Search", id="modal-search")
            yield ListView(id="modal-list")
            yield Static("space toggle  enter confirmar  esc cerrar", id="modal-footer")

    def on_mount(self) -> None:
        self.run_worker(self._rebuild(), exclusive=True)
        self.query_one("#modal-search", Input).focus()

    async def _rebuild(self, query: str = "") -> None:
        lv = self.query_one("#modal-list", ListView)
        await lv.clear()
        if not self._names:
            await lv.append(ListItem(Label("[dim]No results found[/dim]"), id="mcp-empty"))
            return
        filtered = [n for n in self._names if not query or query in n.lower()]
        self._filtered = filtered
        for name in filtered:
            enabled = self._states.get(name, True)
            icon = "●" if enabled else "○"
            cls = "mcp-enabled" if enabled else "mcp-disabled"
            li = ListItem(Label(f"{icon} {name}", classes=cls), id=f"mcp-{_safe_id(name)}")
            await lv.append(li)   # FIX: append es async en Textual 8.x (AwaitMount)
        if lv.children:
            lv.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "modal-search":
            self.run_worker(self._rebuild(event.value.lower()), exclusive=True)

    def action_toggle_current(self) -> None:
        lv = self.query_one("#modal-list", ListView)
        if lv.highlighted_child and 0 <= lv.index < len(self._filtered):
            name = self._filtered[lv.index]
            if name in self._states:
                self._states[name] = not self._states[name]
                self.run_worker(
                    self._rebuild(self.query_one("#modal-search", Input).value.lower()),
                    exclusive=True,
                )

    def action_confirm(self) -> None:
        if self._on_done:
            self._on_done(dict(self._states))
        self.dismiss(dict(self._states))

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


# ─────────────────────────────────── EFFORT SELECTOR ─────────────────────────────────

EFFORT_LEVELS = [
    ("low",    "Respuestas rápidas, mínima reflexión"),
    ("medium", "Equilibrio velocidad / calidad  [recomendado]"),
    ("high",   "Razonamiento extendido, más tokens de pensamiento"),
    ("max",    "Sin límite de thinking — tareas complejas"),
]


class EffortSelectorScreen(Screen):
    """Selector de nivel de esfuerzo del modelo."""

    DEFAULT_CSS = _CSS_BASE

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cerrar", show=False),
        Binding("enter", "confirm", "Confirmar", show=False),
    ]

    def __init__(self, current: str = "medium", on_select: Callable[[str], None] | None = None):
        super().__init__()
        self._current = current
        self._on_select = on_select
        self._filtered = list(EFFORT_LEVELS)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static("Effort level", id="modal-title")
            yield Input(placeholder="Search", id="modal-search")
            yield ListView(id="modal-list")
            yield Static("enter confirmar  esc cerrar", id="modal-footer")

    def on_mount(self) -> None:
        self.run_worker(self._rebuild(), exclusive=True)
        self.query_one("#modal-search", Input).focus()

    async def _rebuild(self, query: str = "") -> None:
        lv = self.query_one("#modal-list", ListView)
        await lv.clear()
        self._filtered = [(n, d) for n, d in EFFORT_LEVELS if not query or query in n]
        for name, desc in self._filtered:
            bullet = "● " if name == self._current else "  "
            li = ListItem(
                Label(f"{bullet}[bold]{name}[/bold] [dim]{desc}[/dim]"),
                id=f"effort-{name}"
            )
            if name == self._current:
                li.add_class("active-item")
            await lv.append(li)   # FIX: append es async en Textual 8.x (AwaitMount)
        if lv.children:
            lv.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "modal-search":
            self.run_worker(self._rebuild(event.value.lower()), exclusive=True)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def action_confirm(self) -> None:
        lv = self.query_one("#modal-list", ListView)
        if lv.highlighted_child:
            item_id = lv.highlighted_child.id or ""
            name = item_id.replace("effort-", "")
            if self._on_select:
                self._on_select(name)
            self.dismiss(name)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        name = item_id.replace("effort-", "")
        if self._on_select:
            self._on_select(name)
        self.dismiss(name)


# ─────────────────────────────────── COMMAND PALETTE ─────────────────────────────────

COMMANDS = [
    ("/agents",    "Switch agent"),
    ("/connect",   "Connect provider"),
    ("/debug",     "View debug info"),
    ("/diff",      "Open diff viewer"),
    ("/effort",    "Set effort level"),
    ("/exit",      "Exit the app"),
    ("/get",       "Get config value"),
    ("/help",      "Show help"),
    ("/historial", "Ver sesión archivada"),
    ("/mcps",      "Toggle MCPs"),
    ("/memory",    "Editar resumen de memoria"),
    ("/models",    "Switch model"),
    ("/new",       "New session"),
    ("/sesiones",  "List sessions"),
    ("/set",       "Set config value"),
    ("/themes",    "Switch theme"),
]


# ─────────────────────────────────── MEMORY EDITOR ────────────────────────────────────

class MemoryEditorScreen(Screen):
    """
    Editor del resumen acumulativo de memoria (rolling summary).

    A diferencia de los selectores de arriba, esto no elige de una lista:
    muestra el texto completo del resumen actual en un TextArea editable.
    Ctrl+S guarda directo en resumen_memoria (sin pasar por el LLM
    consolidador — es edición manual del usuario), Escape cierra sin guardar.

    Pensálo como abrir el archivo de memoria en un editor de texto: lo que
    Aether tiene guardado sobre vos, visible y editable a mano, en vez de un
    dump de mensajes crudos que nadie lee.
    """

    DEFAULT_CSS = """
    MemoryEditorScreen {
        align: center middle;
        background: rgba(0,0,0,0.6);
    }

    #memory-box {
        width: 90%;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 0 1;
    }

    #memory-title {
        text-style: bold;
        color: $text;
        border-bottom: solid $primary-darken-2;
        height: 1;
        margin-bottom: 1;
    }

    #memory-textarea {
        height: 1fr;
        border: solid $primary-darken-2;
    }

    #memory-footer {
        height: 1;
        margin-top: 1;
        border-top: solid $primary-darken-2;
        color: $text-muted;
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cerrar sin guardar", show=False),
        Binding("ctrl+s", "guardar", "Guardar", show=False),
    ]

    def __init__(self, texto_actual: str, on_save: Callable[[str], None] | None = None):
        super().__init__()
        self._texto_actual = texto_actual
        self._on_save = on_save

    def compose(self) -> ComposeResult:
        with Vertical(id="memory-box"):
            yield Static("Memoria de Aether — resumen acumulativo", id="memory-title")
            yield TextArea(self._texto_actual, id="memory-textarea")
            yield Static("ctrl+s guardar   esc cerrar sin guardar", id="memory-footer")

    def on_mount(self) -> None:
        self.query_one("#memory-textarea", TextArea).focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def action_guardar(self) -> None:
        nuevo_texto = self.query_one("#memory-textarea", TextArea).text
        if self._on_save:
            self._on_save(nuevo_texto)
        self.dismiss(nuevo_texto)


class CommandPaletteScreen(Screen):
    """Paleta de comandos (Ctrl+P) al estilo OpenCode."""

    DEFAULT_CSS = _CSS_BASE

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cerrar", show=False),
        Binding("enter", "confirm", "Confirmar", show=False),
    ]

    def __init__(self, on_select: Callable[[str], None] | None = None):
        super().__init__()
        self._on_select = on_select
        self._filtered = list(COMMANDS)

    def compose(self) -> ComposeResult:
        with Vertical(id="modal-box"):
            yield Static("Commands", id="modal-title")
            yield Input(placeholder="Search", id="modal-search")
            yield ListView(id="modal-list")
            yield Static("enter ejecutar  esc cerrar", id="modal-footer")

    def on_mount(self) -> None:
        self.run_worker(self._rebuild(), exclusive=True)
        self.query_one("#modal-search", Input).focus()

    async def _rebuild(self, query: str = "") -> None:
        lv = self.query_one("#modal-list", ListView)
        await lv.clear()
        self._filtered = [(cmd, desc) for cmd, desc in COMMANDS if not query or query in cmd]
        for cmd, desc in self._filtered:
            li = ListItem(
                Label(f"[bold]{cmd}[/bold]  [dim]{desc}[/dim]"),
                id=f"cmd-{cmd.lstrip('/')}",
            )
            await lv.append(li)   # FIX: append es async en Textual 8.x (AwaitMount)
        if lv.children:
            lv.index = 0

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "modal-search":
            self.run_worker(self._rebuild(event.value.lower()), exclusive=True)

    def action_dismiss_none(self) -> None:
        self.dismiss(None)

    def action_confirm(self) -> None:
        lv = self.query_one("#modal-list", ListView)
        if lv.highlighted_child:
            item_id = lv.highlighted_child.id or ""
            cmd = "/" + item_id.replace("cmd-", "", 1)
            if self._on_select:
                self._on_select(cmd)
            self.dismiss(cmd)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        item_id = event.item.id or ""
        cmd = "/" + item_id.replace("cmd-", "", 1)
        if self._on_select:
            self._on_select(cmd)
        self.dismiss(cmd)


# ───────────────────────────────── PASTE PREVIEW ─────────────────────────────────

class PastePreviewScreen(Screen):
    """
    Muestra el contenido completo de un pegado colapsado por PasteInput
    (ver widgets/paste_input.py). Solo lectura — no es un editor, es una
    forma de ver "qué hay adentro" del placeholder "[pasted N characters]"
    antes de mandar el mensaje.
    """

    DEFAULT_CSS = """
    PastePreviewScreen {
        align: center middle;
        background: rgba(0,0,0,0.6);
    }

    #paste-box {
        width: 90%;
        height: 80%;
        background: $surface;
        border: solid $primary;
        padding: 0 1;
    }

    #paste-title {
        text-style: bold;
        color: $text;
        border-bottom: solid $primary-darken-2;
        height: 1;
        margin-bottom: 1;
    }

    #paste-textarea {
        height: 1fr;
        border: solid $primary-darken-2;
    }

    #paste-footer {
        height: 1;
        margin-top: 1;
        border-top: solid $primary-darken-2;
        color: $text-muted;
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss_none", "Cerrar", show=False),
        Binding("enter", "dismiss_none", "Cerrar", show=False),
    ]

    def __init__(self, texto: str):
        super().__init__()
        self._texto = texto

    def compose(self) -> ComposeResult:
        with Vertical(id="paste-box"):
            yield Static(f"Pegado completo — {len(self._texto)} caracteres", id="paste-title")
            yield TextArea(self._texto, id="paste-textarea", read_only=True, show_line_numbers=True)
            yield Static("esc / enter cerrar", id="paste-footer")

    def on_mount(self) -> None:
        self.query_one("#paste-textarea", TextArea).focus()

    def action_dismiss_none(self) -> None:
        self.dismiss(None)


# ───────────────────────────────────────────────── DIR AUTH GATE ─────────────────────────────────────────────

class DirAuthScreen(Screen):
    """
    Gate de autorización de directorio para la TUI.

    Se muestra SOLO la primera vez que Aether arranca en una carpeta
    (fue_evaluada(ruta) es False): presenta al agente con el mismo texto
    que los entrypoints de terminal (PRESENTACION en
    core/config/dir_authorization.py) y persiste la decisión del Creador
    con autorizar()/denegar(). No decide nada por su cuenta: traduce el
    flujo de input() de la terminal a botones de la TUI.

    dismiss() retorna True si autorizó, False si denegó (el caller en
    tui/app.py cierra la aplicación cuando la decisión es negativa).
    """

    DEFAULT_CSS = """
    DirAuthScreen {
        align: center middle;
        background: rgba(0,0,0,0.6);
    }

    #dirauth-box {
        width: 68;
        height: auto;
        max-height: 92%;
        background: $surface;
        border: solid $warning;
        padding: 1 2;
    }

    #dirauth-title {
        text-style: bold;
        color: $text;
        border-bottom: solid $primary-darken-2;
        height: 1;
        margin-bottom: 1;
    }

    #dirauth-text {
        height: auto;
        max-height: 12;
        overflow-y: auto;
        color: $text;
        margin-bottom: 1;
    }

    #dirauth-buttons {
        height: auto;
        align-horizontal: center;
    }

    #dirauth-buttons Button {
        margin: 0 2;
    }

    #dirauth-footer {
        height: 1;
        margin-top: 1;
        border-top: solid $primary-darken-2;
        color: $text-muted;
        padding: 0;
    }
    """

    BINDINGS = [
        Binding("s", "autorizar", "Autorizar", show=False),
        Binding("n", "denegar", "Denegar", show=False),
        Binding("escape", "denegar", "Denegar", show=False),
    ]

    def __init__(self, ruta):
        super().__init__()
        self._ruta = str(ruta)

    def compose(self) -> ComposeResult:
        from core.config.dir_authorization import PRESENTACION
        with Vertical(id="dirauth-box"):
            yield Static("Aether — autorización de directorio", id="dirauth-title")
            yield Static(PRESENTACION.format(ruta=self._ruta), id="dirauth-text")
            with Horizontal(id="dirauth-buttons"):
                yield Button("✅ Autorizar (s)", variant="success", id="dirauth-btn-si")
                yield Button("❌ No autorizar (n)", variant="error", id="dirauth-btn-no")
            yield Static("s autorizar   n / esc denegar", id="dirauth-footer")

    def on_mount(self) -> None:
        self.query_one("#dirauth-btn-si", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # Mismo patrón que AetherApp.on_button_pressed (tui/app.py): decidir
        # por id del botón, sin depender de helpers de versión de Textual.
        if event.button.id == "dirauth-btn-si":
            self.action_autorizar()
        elif event.button.id == "dirauth-btn-no":
            self.action_denegar()

    def action_autorizar(self) -> None:
        from core.config.dir_authorization import autorizar
        autorizar(self._ruta)
        self.dismiss(True)

    def action_denegar(self) -> None:
        from core.config.dir_authorization import denegar
        denegar(self._ruta)
        self.dismiss(False)
