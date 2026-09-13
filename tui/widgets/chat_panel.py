"""
chat_panel.py — Widget para mostrar conversaciones entre usuario y Aether.

Usa un RichLog interno (scrollable, con markup) en vez de Static.update(),
que sobreescribía todo el contenido en cada mensaje nuevo. Esto habilita:
  1. Scroll hacia arriba dentro de la sesión actual (nativo de RichLog).
  2. Carga en bloque del historial de sesiones pasadas al abrir la TUI,
     vía cargar_historial().
  3. Alternar entre la sesión en vivo y sesiones archivadas (comando
     '/sesiones' + '/historial <n>' / '/historial actual' en app.py),
     vía mostrar_sesion() / volver_a_sesion_actual().

La sesión en vivo se guarda aparte en self._vivo (buffer de líneas ya
renderizadas) para poder restaurarla tal cual al volver de ver una sesión
archivada, sin perder nada de lo que pasó mientras el usuario miraba historial
viejo.

Además hay un Static transitorio (`#chat_streaming`) debajo del RichLog donde
se ve la respuesta del asistente EN VIVO, token por token, bajo el prefijo
"Aether:". Al llegar el DoneEvent se limpia y el mensaje final ya saneado
queda escrito en el RichLog.
"""

from textual.app import ComposeResult
from textual.widgets import RichLog, Static


class ChatPanel(Static):
    """Widget que muestra la conversación del chat."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._chat_log: RichLog | None = None
        self._streaming: Static | None = None  # respuesta en vivo, bajo "Aether:"
        self._vivo: list[str] = []       # buffer de la sesión en vivo (para restaurar)
        self._viendo_archivo = False     # True mientras se muestra una sesión vieja

    def compose(self) -> ComposeResult:
        self._chat_log = RichLog(
            id="chat_richlog",
            markup=True,
            wrap=True,
            highlight=False,
            auto_scroll=True,
        )
        yield self._chat_log
        self._streaming = Static("", id="chat_streaming")
        yield self._streaming

    @property
    def log(self):
        return self._chat_log

    @log.setter
    def log(self, value):
        self._chat_log = value

    def agregar_mensaje(self, mensaje=None, tipo="user", from_history=False,
                        es_markup_intencional=False):
        """Agrega un mensaje al chat. Se llama desde app.py.

        ``es_markup_intencional``: si True, el mensaje contiene markup de
        Rich/Textual deliberado (p.ej. ``[bold]...[/bold]``) generado por
        la propia TUI, no por el LLM — no se escapa, solo se stripean
        códigos ANSI por seguridad. Si False (valor por defecto), el
        mensaje se considera texto libre del LLM/usuario y se escapan
        **todos** los corchetes ``[`` ``]`` para evitar MarkupError en
        el RichLog(markup=True) subyacente.
        """
        if mensaje is None:
            return

        from .debug_panel import _strip_ansi, _escapar_corchetes
        texto_limpio = _strip_ansi(str(mensaje))
        prefijos = {
            "user": "[bold cyan]Tú:[/bold cyan]\n",
            "assistant": "[bold green]Aether:[/bold green]\n",
        }
        prefix = prefijos.get(tipo, "")
        if es_markup_intencional:
            texto = f"{prefix}{texto_limpio}"
        else:
            texto = f"{prefix}{_escapar_corchetes(texto_limpio)}"

        if not from_history:
            # Todo lo que no viene de una carga de historial es actividad
            # en vivo de esta sesión — se guarda siempre, se muestre o no.
            self._vivo.append(texto)
            if self._viendo_archivo:
                # No interrumpimos la vista de la sesión archivada; el
                # mensaje ya quedó en _vivo y aparece al volver con
                # '/historial actual'.
                return

        if self._chat_log:
            if not from_history:
                self._chat_log.auto_scroll = (
                    self._chat_log.scroll_y >= self._chat_log.max_scroll_y - 2
                )
            self._chat_log.write(texto)
        else:
            # Fallback defensivo si compose() todavía no corrió.
            self.update(f"\n{texto}")

    def mostrar_streaming(self, texto_acumulado: str) -> None:
        """Actualiza la respuesta en vivo bajo el prefijo "Aether:".

        Se llama con el texto acumulado en cada TokenEvent. El cuerpo se
        sanea igual que un mensaje normal (ANSI fuera, corchetes escapados)
        porque el Static usa markup=True; el prefijo es intencional.
        Mientras se mira una sesión archivada no se muestra nada para no
        mezclar la vista (al volver, el próximo token repinta todo).
        """
        if self._viendo_archivo or self._streaming is None:
            return
        from .debug_panel import _strip_ansi, _escapar_corchetes
        cuerpo = _escapar_corchetes(_strip_ansi(str(texto_acumulado)))
        self._streaming.update(f"[bold green]Aether:[/bold green]\n{cuerpo}")
        try:
            self._streaming.scroll_end(animate=False)
        except Exception:  # noqa: BLE001 — scroll best-effort
            pass

    def ocultar_streaming(self) -> None:
        """Limpia el widget de streaming (al completar, fallar o reenviar)."""
        if self._streaming is not None:
            self._streaming.update("")

    def cargar_historial(self, turnos: list[dict]) -> None:
        """
        Escribe en bloque el historial pasado (últimos N turnos globales,
        ver engine_bridge.obtener_historial_para_mostrar) al montar la TUI.
        No pasa a formar parte de _vivo: es contexto visual, no la sesión
        actual.
        """
        if not self._chat_log or not turnos:
            return

        self._chat_log.auto_scroll = True
        for turno in turnos:
            self._escribir_turno(turno)
        self._chat_log.write("[dim]── sesión anterior · fin del historial cargado ──[/dim]")

    def mostrar_sesion(self, turnos: list[dict], etiqueta: str = "") -> None:
        """
        Reemplaza la vista actual por el historial completo de una sesión
        archivada (comando '/historial <n>'). La sesión en vivo sigue
        corriendo de fondo — se restaura con volver_a_sesion_actual().
        """
        if not self._chat_log:
            return
        self._viendo_archivo = True
        self.ocultar_streaming()
        self._chat_log.clear()
        titulo = f"── viendo sesión archivada: {etiqueta} ──" if etiqueta else "── viendo sesión archivada ──"
        self._chat_log.write(f"[bold]{titulo}[/bold]")
        for turno in turnos:
            self._escribir_turno(turno)
        self._chat_log.write("[dim]── fin de sesión archivada · '/historial actual' para volver ──[/dim]")
        self._chat_log.auto_scroll = True

    def volver_a_sesion_actual(self) -> None:
        """Restaura la vista de la sesión en vivo tal como quedó."""
        if not self._chat_log:
            return
        self._viendo_archivo = False
        self._chat_log.clear()
        for texto in self._vivo:
            self._chat_log.write(texto)
        self._chat_log.auto_scroll = True

    def _escribir_turno(self, turno: dict) -> None:
        rol = turno.get("rol", "")
        texto = turno.get("texto", "")
        if not texto:
            return
        tipo = "user" if rol == "usuario" else "assistant"
        self.agregar_mensaje(texto, tipo, from_history=True)

    def reset(self):
        """Resetea el chat."""
        self.ocultar_streaming()
        if self._chat_log:
            self._chat_log.clear()
        self._vivo.clear()
        self._viendo_archivo = False
