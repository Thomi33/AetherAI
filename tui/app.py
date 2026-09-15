"""
app.py — Aplicación principal TUI de Aether.
"""

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input, Button
from textual import work
from textual.timer import Timer
import re
import time

# ── Sanitización para widgets con markup ──
# rich.markup.escape solo escapa tags minúsculas ([a-z#/@]...) y deja
# [SHELL], [/SHELL], [INFO], [✓], JSON {"name":...} o ANSI \x1b[...] sin
# tocar → Textual los parsea como markup y tira MarkupError.
_RE_STREAM_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _sanear_streaming(texto: str) -> str:
    """Texto plano seguro para Static(markup=True) como streaming_line."""
    s = _RE_STREAM_ANSI.sub("", str(texto))
    return s.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


# La limpieza de la respuesta final se comparte con el backend web:
# ver core/utils/response_cleaner.py (backend/api/routes/chat.py la usa
# para el evento "done" del SSE). Nombre mantenido por compatibilidad.
from core.utils.response_cleaner import limpiar_respuesta_chat  # noqa: E402

from tui.widgets.chat_panel import ChatPanel
from tui.widgets.plan_panel import PlanPanel
from tui.widgets.status_bar import StatusBar
from tui.widgets.debug_panel import DebugPanel
from tui.widgets.paste_input import PasteInput

from tui.widgets.slash_completer import SlashCompleter
from tui.widgets.selector_screens import (
    ModelSelectorScreen,
    AgentSelectorScreen,
    ThemeSelectorScreen,
    McpSelectorScreen,
    EffortSelectorScreen,
    CommandPaletteScreen,
    MemoryEditorScreen,
    DirAuthScreen,
)

from tui.engine_bridge import (
    iter_eventos,
    inicializar_motor,
    calentar_modelo,
    obtener_historial_para_mostrar,
    NodeUpdateEvent,
    StdoutLineEvent,
    TokenEvent,
    DoneEvent,
    ErrorEvent,
)
from tui.status_messages import STATUS_CONFIG, choose_flavor, real_state_for_node

SLASH_HELP = (
    "Comandos disponibles:\n"
    "  F2                     — dictar por voz (push-to-talk, whisper local)\n"
    "  Ctrl+R                 — ver contenido completo de un pegado colapsado\n"
    "  /help                  — esta ayuda\n"
    "  /debug                 — muestra/oculta el panel de debug\n"
    "  /get [clave]           — ver config actual (todas, o una clave puntual)\n"
    "  /set <clave> <valor>   — cambiar un valor de config en caliente\n"
    "  /prompt                — system prompt en caliente (extra/override/preview)\n"
    "  /prompt                 — system prompt sin tocar código (ver/extra/override/sintesis/clear/preview)\n"
    "  /sesiones              — listar sesiones anteriores\n"
    "  /historial <n>         — ver la sesión #n de esa lista\n"
    "  /historial actual      — volver a la sesión en vivo\n"
    "  /models                — cambiar modelo de Ollama en vivo\n"
    "  /themes                — cambiar tema visual\n"
    "  /mcps                  — toggle servidores MCP\n"
    "  /effort                — nivel de esfuerzo (low/medium/high/max)\n"
    "  /agents                — cambiar agente\n"
    "  /new                   — nueva sesión\n"
    "  /play-roblox [google]  — iniciar/detener autonomía Roblox\n"
    "  /stop                  — detener la inferencia actual"
)


class AetherApp(App):
    """Aplicación TUI principal de Aether."""

    BINDINGS = [
        ("tab", "open_agents", "agents"),
        ("ctrl+p", "open_commands", "commands"),
        ("ctrl+c", "stop_inference", "⏹ detener inferencia"),
        ("f2", "toggle_dictado", "🎙️ dictar"),
    ]

    CSS = """
    Screen { background: $surface; color: $text; }
    #chat_panel { height: 1fr; }
    #chat_richlog { height: 1fr; }
    #chat_streaming { height: auto; max-height: 12; padding: 0 1; overflow-y: auto; }
    #streaming_line { height: auto; color: $secondary-lighten-2; padding: 0 1; }
    #plan_panel, #status_bar, #debug_panel { dock: bottom; }
    #status_bar { height: 2; background: $primary; color: $text; padding: 0 1; }
    #debug_panel { height: 10; background: $surface; color: $text; padding: 0 1; border-top: solid $accent; }
    #plan_panel { height: 8; background: $surface-darken-1; color: $text; padding: 0 1; }
    """

    def __init__(self, workdir: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._thinking = False
        self._respuesta_en_curso = ""
        self._ultimas_sesiones: list[dict] = []  # cache para '/historial <n>' tras '/sesiones'
        self._current_agent = "build"
        self._current_effort = "medium"
        self._workdir = workdir  # --workdir (lo aplica on_mount vía settings)
        self._grabador = None  # GrabadorAudio, lazy (ver core/services/stt_service.py)
        self._estado_real = ""
        self._inicio_operacion = 0.0
        self._flavor_timer: Timer | None = None
        self._startup_loading = False
        self._startup_tick = 0
        self._roblox_runtime = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield ChatPanel(id="chat_panel")
        yield Static("", id="streaming_line")
        yield PlanPanel(id="plan_panel")
        yield DebugPanel(id="debug_panel")
        yield StatusBar(id="status_bar")
        yield SlashCompleter(id="slash_completer")
        yield PasteInput(placeholder="Escribí tu mensaje aquí...", id="input_chat")
        yield Button("Enviar", id="btn_enviar")
        yield Button("⏹ Detener", id="btn_detener", variant="error", disabled=True)
        yield Footer()

    def on_mount(self) -> None:
        self._flavor_timer = self.set_interval(
            STATUS_CONFIG["flavor_interval_seconds"], self._mostrar_flavor
        )
        self.set_interval(0.35, self._animar_estado)
        # --workdir (TUI): fija AETHER_CWD antes de que el motor lea settings.
        # bin/aether ya exporta AETHER_CWD con el $PWD de invocación; acá solo
        # pisa si el usuario pasó --workdir explícito al comando `aether`.
        if getattr(self, "_workdir", None):
            import os
            os.environ["AETHER_CWD"] = self._workdir

        # Gate real de autorización (ver core/config/dir_authorization.py):
        # si esta carpeta nunca fue evaluada, se pregunta ANTES de inicializar
        # el motor -- que ya puede tocar archivos en la sesión. La terminal
        # (jarvis_new.py / aether_run.py task) hace lo mismo con
        # presentacion_y_confirmacion(); acá es el modal DirAuthScreen.
        from core.config.dir_authorization import (
            esta_autorizado,
            fue_evaluada,
            resolver_dir_trabajo,
        )
        self._dir_trabajo = resolver_dir_trabajo()
        if fue_evaluada(self._dir_trabajo) and esta_autorizado(self._dir_trabajo):
            self._arrancar_sesion()
        else:
            self.push_screen(DirAuthScreen(self._dir_trabajo), self._post_auth_gate)

    def _arrancar_sesion(self) -> None:
        """Continuación del arranque, después (o sin) el gate de directorio."""
        self._startup_loading = True
        self._mostrar_estado_real("iniciando grafo, espere")
        self._mostrar_banner_dir()
        self._inicializar_motor_bg()

    def _post_auth_gate(self, autorizado) -> None:
        """Continúa solo si el Creador autorizó el directorio de trabajo."""
        ruta = getattr(self, "_dir_trabajo", "?")
        if not autorizado:
            self.exit()
            return
        self._arrancar_sesion()

    @work(thread=True)
    def _inicializar_motor_bg(self) -> None:
        try:
            inicializar_motor()
            from core.agent.graph_builder import get_graph
            get_graph()
            calentar_modelo()
            historial = obtener_historial_para_mostrar()
            chat_panel = self.query_one("#chat_panel", ChatPanel)
            self.call_from_thread(chat_panel.cargar_historial, historial)
            self.call_from_thread(self._finalizar_inicio)
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(
                self.query_one("#chat_panel", ChatPanel).agregar_mensaje,
                f"⚠️ Error inicializando el motor: {e}",
                "assistant",
            )
            self.call_from_thread(self._finalizar_inicio)

    def _finalizar_inicio(self) -> None:
        self._startup_loading = False
        self._mostrar_estado_real("")

    def _animar_estado(self) -> None:
        if self._startup_loading:
            self._startup_tick = (self._startup_tick + 1) % 4
            self.query_one("#streaming_line", Static).update(
                _sanear_streaming(f"iniciando grafo, espere{'.' * self._startup_tick}")
            )
        elif self._thinking and self._estado_real:
            self._startup_tick = (self._startup_tick + 1) % 4
            self.query_one("#streaming_line", Static).update(
                _sanear_streaming(self._estado_real + "." * self._startup_tick)
            )
        elif not self._thinking and self._estado_real.startswith("["):
            self.query_one("#streaming_line", Static).update(
                _sanear_streaming(self._estado_real)
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_enviar":
            self._enviar_mensaje()
        elif event.button.id == "btn_detener":
            self.action_stop_inference()

    def action_stop_inference(self) -> None:
        if not self._thinking:
            return
        from tui.engine_bridge import cancelar_inferencia
        cancelar_inferencia()
        self.query_one("#chat_panel", ChatPanel).agregar_mensaje(
            "⏹ Deteniendo la inferencia...", "assistant"
        )

    async def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "input_chat":
            sc = self.query_one("#slash_completer", SlashCompleter)
            # FIX (crash slash commands): update_query es async en Textual 8.x
            # (clear/append de ListView retornan AwaitRemove/AwaitMount).
            await sc.update_query(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "input_chat":
            event.stop()
            sc = self.query_one("#slash_completer", SlashCompleter)
            # Si el completer está visible, Tab/Enter confirma el comando
            if sc.display:
                cmd = sc.confirm()
                if cmd:
                    input_widget = self.query_one("#input_chat", Input)
                    input_widget.value = cmd + " "
                    input_widget.cursor_position = len(input_widget.value)
                return
            self._enviar_mensaje()

    def on_key(self, event) -> None:
        """Flechas arriba/abajo navegan el completer; Escape lo cierra."""
        sc = self.query_one("#slash_completer", SlashCompleter)
        if not sc.display:
            return
        if event.key == "up":
            sc.move_up()
            event.stop()
        elif event.key == "down":
            sc.move_down()
            event.stop()
        elif event.key == "escape":
            sc.hide()
            event.stop()
        elif event.key == "tab":
            cmd = sc.confirm()
            if cmd:
                input_widget = self.query_one("#input_chat", Input)
                input_widget.value = cmd + " "
                input_widget.cursor_position = len(input_widget.value)
            event.stop()

    def _enviar_mensaje(self) -> None:
        input_widget = self.query_one("#input_chat", PasteInput)
        texto = input_widget.value.strip()
        if not texto:
            return
        if self._thinking:
            if texto.lower() == "/stop":
                input_widget.value = ""
                self.action_stop_inference()
            return
        # Resolver placeholders "[pasted N characters]" al texto real antes
        # de procesar el mensaje (ver PasteInput en widgets/paste_input.py).
        texto = input_widget.resolver_texto(texto)
        # Si el completer está visible al enviar, cerrarlo
        self.query_one("#slash_completer", SlashCompleter).hide()
        input_widget.value = ""
        input_widget.limpiar_pegados()

        if texto.startswith("/"):
            self._manejar_comando_slash(texto)
            return

        chat_panel = self.query_one("#chat_panel", ChatPanel)
        chat_panel.agregar_mensaje(texto, "user")
        chat_panel.ocultar_streaming()
        # Nuevo turno: limpiar la actividad del turno anterior.
        self.query_one("#plan_panel", PlanPanel).reset()

        self._thinking = True
        self._inicio_operacion = time.monotonic()
        self._mostrar_estado_real("[•] Analizando solicitud...")
        self._respuesta_en_curso = ""
        input_widget.disabled = False
        input_widget.placeholder = "Inferencia en curso — escribí /stop o usá Ctrl+C"
        self.query_one("#btn_detener", Button).disabled = False
        self._procesar_orden_bg(texto)

    def _manejar_comando_slash(self, texto: str) -> None:
        """Comandos '/' que se resuelven localmente en la TUI, sin pasar por el grafo."""
        chat_panel = self.query_one("#chat_panel", ChatPanel)
        partes = texto.strip().split(maxsplit=2)
        cmd = partes[0].lower()

        if cmd in ("/help", "/?"):
            chat_panel.agregar_mensaje(SLASH_HELP, "assistant")

        elif cmd == "/debug":
            debug_panel = self.query_one("#debug_panel", DebugPanel)
            debug_panel.display = not debug_panel.display
            estado = "visible" if debug_panel.display else "oculto"
            chat_panel.agregar_mensaje(f"Panel de debug: {estado}", "assistant")

        elif cmd == "/get":
            from core.config.config_manager import get_config_manager
            config = get_config_manager()
            clave = partes[1].upper() if len(partes) > 1 else None
            if clave:
                valor = config.get(clave)
                if valor is None and clave not in config.get_all():
                    texto_resp = f"⚠️ Clave desconocida: {clave}"
                else:
                    texto_resp = f"{clave} = {valor}"
            else:
                todas = config.get_all()
                texto_resp = "\n".join(f"{k} = {v}" for k, v in sorted(todas.items()))
            chat_panel.agregar_mensaje(texto_resp, "assistant")

        elif cmd == "/set":
            from core.config.config_manager import get_config_manager
            if len(partes) < 3:
                chat_panel.agregar_mensaje("Uso: /set <clave> <valor>  (ej: /set temperature 0.8)", "assistant")
            else:
                config = get_config_manager()
                clave = partes[1].upper()
                valor_raw = partes[2]
                # Intentar castear a tipo razonable (bool/int/float) antes de validar
                if valor_raw.lower() in ("true", "false"):
                    valor = valor_raw.lower() == "true"
                else:
                    try:
                        valor = int(valor_raw)
                    except ValueError:
                        try:
                            valor = float(valor_raw)
                        except ValueError:
                            valor = valor_raw

                # Claves que en realidad viven ANIDADAS dentro de
                # OLLAMA_GEN_OPTIONS (num_gpu, num_batch, num_thread, etc.).
                # Sin este mapeo, `/set num_gpu 32` crea una clave huérfana
                # de nivel superior que _llm_chat nunca lee (bug real que
                # nos costó una sesión entera de debugging).
                _CLAVES_GEN_OPTIONS = {"NUM_GPU", "NUM_BATCH", "NUM_THREAD", "TOP_P", "TOP_K"}
                if clave in _CLAVES_GEN_OPTIONS:
                    gen_opts = dict(config.get("OLLAMA_GEN_OPTIONS", {}) or {})
                    gen_opts[clave.lower()] = valor
                    ok = config.set("OLLAMA_GEN_OPTIONS", gen_opts, validate=True)
                    if ok:
                        texto_resp = f"✅ OLLAMA_GEN_OPTIONS.{clave.lower()} = {valor}"
                    else:
                        texto_resp = f"❌ No se pudo aplicar OLLAMA_GEN_OPTIONS.{clave.lower()} = {valor}"
                    chat_panel.agregar_mensaje(texto_resp, "assistant")
                else:
                    ok = config.set(clave, valor, validate=True)
                    if ok:
                        texto_resp = f"✅ {clave} = {valor}"
                    else:
                        texto_resp = f"❌ No se pudo aplicar {clave} = {valor} (validación falló o clave inválida)"
                    chat_panel.agregar_mensaje(texto_resp, "assistant")

        elif cmd == "/prompt":
            self._manejar_comando_prompt(partes)

        elif cmd == "/sesiones":
            self._listar_sesiones()

        elif cmd == "/historial":
            arg = partes[1] if len(partes) > 1 else ""
            self._cambiar_vista_historial(arg)

        elif cmd == "/models":
            self._abrir_model_selector()

        elif cmd == "/themes":
            self._abrir_theme_selector()

        elif cmd == "/mcps":
            self._abrir_mcp_selector()

        elif cmd == "/effort":
            self._abrir_effort_selector()

        elif cmd == "/memory":
            arg = partes[1].lower() if len(partes) > 1 else ""
            if arg == "consolidar":
                self._forzar_consolidacion_memoria()
            else:
                self._abrir_memory_editor()

        elif cmd in {"/play-roblox", "/pay-roblox"}:
            self._manejar_play_roblox(partes[1].lower() if len(partes) > 1 else "start")

        elif cmd == "/agents":
            self._abrir_agent_selector()

        elif cmd == "/new":
            self._nueva_sesion()

        else:
            chat_panel.agregar_mensaje(f"⚠️ Comando desconocido: {cmd}. Probá /help", "assistant")

    def _manejar_comando_prompt(self, partes: list[str]) -> None:
        """
        /prompt — System prompt editable SIN tocar código, en caliente.

        Subcomandos:
          /prompt                        — estado actual (override/extra/sintesis)
          /prompt extra <texto...>       — extra para TODOS los prompts
          /prompt override <texto...>    — reemplaza la persona por defecto
          /prompt sintesis <texto...>    — extra solo para la persona de síntesis
          /prompt clear [extra|override|sintesis|all]
          /prompt preview                — muestra el system prompt final real
        """
        from core.config.config_manager import get_config_manager
        chat_panel = self.query_one("#chat_panel", ChatPanel)
        config = get_config_manager()

        claves = {
            "extra": "SYSTEM_PROMPT_EXTRA",
            "override": "SYSTEM_PROMPT_OVERRIDE",
            "sintesis": "SYSTEM_PROMPT_SINTESIS_EXTRA",
        }
        sub = partes[1].lower() if len(partes) > 1 else ""

        if sub in ("", "show", "ver"):
            lineas = ["System prompt configurable (config.json — sin tocar código):"]
            for nombre, clave in claves.items():
                valor = config.get(clave, "") or ""
                estado = "ACTIVO" if valor.strip() else "vacío (default del código)"
                lineas.append(f"  {nombre:9} [{clave}] = {estado}")
                if valor.strip():
                    lineas.append(f"    → {valor}")
            lineas.append(
                "Uso: /prompt extra <texto> · /prompt override <texto> · "
                "/prompt sintesis <texto> · /prompt clear [cual] · /prompt preview"
            )
            chat_panel.agregar_mensaje("\n".join(lineas), "assistant")

        elif sub in claves:
            if len(partes) < 3:
                chat_panel.agregar_mensaje(
                    f"Uso: /prompt {sub} <texto...>  (texto vacío no está permitido; para desactivar: /prompt clear {sub})",
                    "assistant",
                )
                return
            texto = partes[2].strip()
            ok = config.set(claves[sub], texto, validate=True)
            if ok:
                chat_panel.agregar_mensaje(
                    f"✅ System prompt actualizado ({sub}). Se aplica al PRÓXIMO turno.",
                    "assistant",
                )
            else:
                chat_panel.agregar_mensaje(
                    f"❌ No se pudo aplicar {claves[sub]} (validación falló).",
                    "assistant",
                )

        elif sub == "clear":
            objetivo = partes[2].lower() if len(partes) > 2 else "all"
            if objetivo == "all":
                objetivos = list(claves.values())
            elif objetivo in claves:
                objetivos = [claves[objetivo]]
            else:
                chat_panel.agregar_mensaje(
                    "Uso: /prompt clear [extra|override|sintesis|all]", "assistant"
                )
                return
            for clave in objetivos:
                config.set(clave, "", validate=True)
            chat_panel.agregar_mensaje(
                f"✅ System prompt reseteado: {', '.join(objetivos)} (usa el default del código).",
                "assistant",
            )

        elif sub == "preview":
            try:
                from core.agent.prompts import construir_backstory
                preview = construir_backstory("")
                chat_panel.agregar_mensaje(preview, "assistant")
            except Exception as exc:  # noqa: BLE001
                chat_panel.agregar_mensaje(
                    f"❌ No se pudo construir el preview: {type(exc).__name__}: {exc}",
                    "assistant",
                )

        else:
            chat_panel.agregar_mensaje(
                "Uso: /prompt [extra|override|sintesis <texto>] · "
                "/prompt clear [extra|override|sintesis|all] · /prompt preview · /prompt",
                "assistant",
            )

    def _manejar_comando_prompt(self, partes: list) -> None:
        """
        Ajusta el system prompt EN CALIENTE sin tocar código (config.json).

        Claves (ver core/agent/prompts.py — _aplicar_overrides_prompt):
          SYSTEM_PROMPT_OVERRIDE        → reemplaza la persona por defecto
          SYSTEM_PROMPT_EXTRA           → extra a TODOS los prompts
          SYSTEM_PROMPT_SINTESIS_EXTRA  → extra solo para la persona de síntesis

        Uso:
          /prompt                        — estado actual
          /prompt extra <texto>          — setea el extra general
          /prompt override <texto>       — reemplaza la persona por defecto
          /prompt sintesis <texto>       — extra solo de síntesis
          /prompt clear [extra|override|sintesis|all]
          /prompt preview                — el prompt final que verá el modelo
        """
        from core.config.config_manager import get_config_manager
        config = get_config_manager()
        chat_panel = self.query_one("#chat_panel", ChatPanel)
        sub = partes[1].lower() if len(partes) > 1 else ""

        _CLAVES = {
            "extra": "SYSTEM_PROMPT_EXTRA",
            "override": "SYSTEM_PROMPT_OVERRIDE",
            "sintesis": "SYSTEM_PROMPT_SINTESIS_EXTRA",
        }

        if sub in ("", "show", "ver"):
            def _estado(clave: str) -> str:
                valor = (config.get(clave, "") or "").strip()
                estado = "ACTIVO" if valor else "off"
                if len(valor) > 80:
                    muestra = valor[:80] + "…"
                else:
                    muestra = valor or "(vacío)"
                return f"  {clave} [{estado}]: {muestra}"

            chat_panel.agregar_mensaje(
                "System prompt configurable (config.json — sin tocar código):\n"
                + "\n".join(_estado(c) for c in _CLAVES.values())
                + "\n\nUso: /prompt extra|override|sintesis <texto> · "
                "/prompt clear [extra|override|sintesis|all] · /prompt preview",
                "assistant",
            )

        elif sub in _CLAVES:
            if len(partes) < 3 or not partes[2].strip():
                chat_panel.agregar_mensaje(
                    f"Uso: /prompt {sub} <texto> — el texto no puede quedar vacío "
                    f"(para desactivar: /prompt clear {sub})",
                    "assistant",
                )
                return
            ok = config.set(_CLAVES[sub], partes[2].strip())
            if ok:
                chat_panel.agregar_mensaje(
                    f"System prompt actualizado ({_CLAVES[sub]}). Aplica en el próximo turno.",
                    "assistant",
                )
            else:
                chat_panel.agregar_mensaje(f"No se pudo aplicar {_CLAVES[sub]}.", "assistant")

        elif sub == "clear":
            objetivo = partes[2].lower() if len(partes) > 2 else "all"
            if objetivo == "all":
                claves = list(_CLAVES.values())
            elif objetivo in _CLAVES:
                claves = [_CLAVES[objetivo]]
            else:
                chat_panel.agregar_mensaje(
                    "Uso: /prompt clear [extra|override|sintesis|all]", "assistant"
                )
                return
            for clave in claves:
                config.set(clave, "")
            chat_panel.agregar_mensaje(
                f"System prompt limpio ({objetivo}): volvió al prompt por defecto del código.",
                "assistant",
            )

        elif sub == "preview":
            from core.agent.prompts import construir_backstory
            preview = construir_backstory("")
            chat_panel.agregar_mensaje(
                f"PREVIEW — system prompt final ({len(preview)} chars):\n\n{preview}",
                "assistant",
            )

        else:
            chat_panel.agregar_mensaje(
                "Uso: /prompt [extra|override|sintesis|clear|preview|show]", "assistant"
            )

    def _manejar_play_roblox(self, accion: str) -> None:
        """Controla el runtime Roblox y permite elegir el proveedor de visión."""
        from core.tools.roblox_bridge import RobloxRuntime
        chat_panel = self.query_one("#chat_panel", ChatPanel)
        try:
            if accion == "stop":
                if self._roblox_runtime is None:
                    ok, message = False, "Roblox autónomo no está activo."
                else:
                    ok, message = self._roblox_runtime.stop()
            elif accion in {"start", "status", "google", "ollama", "hybrid"}:
                if accion in {"google", "ollama", "hybrid"}:
                    if self._roblox_runtime is not None and self._roblox_runtime.running:
                        ok, message = False, "Detené el runtime antes de cambiar el proveedor."
                    else:
                        self._roblox_runtime = RobloxRuntime(vision_provider=accion)
                        ok, message = self._roblox_runtime.start()
                elif accion == "start":
                    if self._roblox_runtime is None:
                        self._roblox_runtime = RobloxRuntime()
                    ok, message = self._roblox_runtime.start()
                else:
                    ok = True
                    message = (
                        f"Roblox autónomo activo (visión: {self._roblox_runtime.vision_provider})."
                        if self._roblox_runtime is not None and self._roblox_runtime.running
                        else "Roblox autónomo detenido."
                    )
            else:
                ok, message = False, "Uso: /play-roblox [start|stop|status|google|ollama|hybrid]"
        except (OSError, RuntimeError, ValueError) as exc:
            ok, message = False, str(exc)
        prefix = "✅" if ok else "⚠️"
        chat_panel.agregar_mensaje(f"{prefix} {message}", "assistant")

    def _listar_sesiones(self) -> None:
        from core.memory.memory_manager import listar_sesiones
        chat_panel = self.query_one("#chat_panel", ChatPanel)
        sesiones = listar_sesiones()
        self._ultimas_sesiones = sesiones
        if not sesiones:
            chat_panel.agregar_mensaje("No hay sesiones guardadas todavía.", "assistant")
            return
        lineas = ["Sesiones recientes (usá /historial <n> para verlas):"]
        for i, s in enumerate(sesiones, start=1):
            preview = s["preview"] or "(sin mensajes de usuario)"
            lineas.append(f"  {i}. [{s['inicio']}] {s['turnos']} turnos — {preview}")
        chat_panel.agregar_mensaje("\n".join(lineas), "assistant")

    # ── Acciones de teclado (Tab / Ctrl+P) ──────────────────────────────────

    def action_open_agents(self) -> None:
        """Tab → abre el selector de agente."""
        if not self._thinking:
            self._abrir_agent_selector()

    def action_open_commands(self) -> None:
        """Ctrl+P → abre la paleta de comandos."""
        if not self._thinking:
            self._abrir_command_palette()

    def action_toggle_dictado(self) -> None:
        """
        F2 → push-to-talk: primera pulsación arranca a grabar audio con
        arecord, la segunda corta la grabación y dispara la transcripción
        (faster-whisper, en el worker aislado de core/services/stt_service.py).
        El texto transcripto queda cargado en el input, listo para revisar
        antes de enviar (no se auto-envía: dictado, no comando de voz).
        """
        if self._thinking:
            return

        from core.config.config_manager import get_config_manager
        chat_panel = self.query_one("#chat_panel", ChatPanel)

        if not get_config_manager().get("STT_ENABLED", True):
            chat_panel.agregar_mensaje("⚠️ Dictado por voz deshabilitado (/set STT_ENABLED true para activarlo).", "assistant")
            return

        if self._grabador is None:
            from core.services.stt_service import GrabadorAudio
            self._grabador = GrabadorAudio()

        input_widget = self.query_one("#input_chat", Input)

        if not self._grabador.grabando:
            try:
                self._grabador.iniciar()
            except Exception as e:  # noqa: BLE001
                chat_panel.agregar_mensaje(f"⚠️ No se pudo iniciar la grabación: {e}", "assistant")
                return
            input_widget.placeholder = "🔴 Grabando... (F2 para detener y transcribir)"
            return

        wav_path = self._grabador.detener()
        input_widget.placeholder = "Escribí tu mensaje aquí..."

        if wav_path is None:
            chat_panel.agregar_mensaje("⚠️ Grabación vacía, no se transcribió nada.", "assistant")
            return

        self._thinking = True
        input_widget.disabled = True
        input_widget.placeholder = "🎧 Transcribiendo..."
        self._transcribir_bg(wav_path)

    @work(thread=True, exclusive=True, group="dictado")
    def _transcribir_bg(self, wav_path) -> None:
        from core.services.stt_service import transcribir_wav, SttNoDisponible
        from core.config.config_manager import get_config_manager

        idioma = get_config_manager().get("STT_LANGUAGE", "es") or None
        try:
            texto = transcribir_wav(wav_path, idioma=idioma)
        except SttNoDisponible as e:
            self.call_from_thread(self._on_dictado_error, str(e))
            return
        except Exception as e:  # noqa: BLE001 — cualquier falla debe llegar a la UI, no colgar el input
            self.call_from_thread(self._on_dictado_error, f"{type(e).__name__}: {e}")
            return
        finally:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
        self.call_from_thread(self._on_dictado_listo, texto)

    def _on_dictado_listo(self, texto: str) -> None:
        input_widget = self.query_one("#input_chat", Input)
        self._thinking = False
        input_widget.disabled = False
        input_widget.placeholder = "Escribí tu mensaje aquí..."
        texto = (texto or "").strip()
        if texto:
            input_widget.value = texto
            input_widget.cursor_position = len(input_widget.value)
        else:
            self.query_one("#chat_panel", ChatPanel).agregar_mensaje(
                "⚠️ No se entendió nada en la grabación.", "assistant"
            )
        input_widget.focus()

    def _on_dictado_error(self, mensaje: str) -> None:
        input_widget = self.query_one("#input_chat", Input)
        self._thinking = False
        input_widget.disabled = False
        input_widget.placeholder = "Escribí tu mensaje aquí..."
        self.query_one("#chat_panel", ChatPanel).agregar_mensaje(f"⚠️ Error transcribiendo: {mensaje}", "assistant")
        input_widget.focus()

    def on_unmount(self) -> None:
        """Corta cualquier grabación en curso y apaga el worker de STT al cerrar la TUI."""
        if self._roblox_runtime is not None:
            self._roblox_runtime.stop()
            self._roblox_runtime = None
        try:
            if self._grabador and self._grabador.grabando:
                self._grabador.descartar()
        except Exception:
            pass
        try:
            from core.services.stt_service import get_stt_worker
            get_stt_worker().cerrar()
        except Exception:
            pass

    # ── Selectores modales ────────────────────────────────────────────────────

    def _abrir_model_selector(self) -> None:
        from core.config.config_manager import get_config_manager
        config = get_config_manager()
        current = config.get("MODELO", "ornith:9b")

        def _on_model(model: str | None) -> None:
            if model:
                # Aceptar cualquier modelo sin validar (viene de ollama list)
                config.set("MODELO", model, validate=False)
                self._actualizar_status_bar()
                self.query_one("#chat_panel", ChatPanel).agregar_mensaje(
                    f"Modelo cambiado a [bold]{model}[/bold]", "assistant"
                )

        self.push_screen(ModelSelectorScreen(current_model=current), _on_model)

    def _abrir_theme_selector(self) -> None:
        from core.config.config_manager import get_config_manager
        config = get_config_manager()
        current = config.get("THEME", "system")

        def _on_theme(theme: str | None) -> None:
            if theme:
                # Intentar aplicar tema nativo de Textual. Muchos nombres de la lista
                # (estilo OpenCode: cobalt2, cursor, opencode, etc.) no existen en el
                # registro de temas de Textual y no tienen paleta implementada aca,
                # asi que solo se guarda la preferencia y se avisa que no hay render.
                aplicado = False
                try:
                    self.theme = theme
                    aplicado = True
                except Exception:
                    aplicado = False
                config.set("THEME", theme, validate=False)
                chat_panel = self.query_one("#chat_panel", ChatPanel)
                if aplicado:
                    chat_panel.agregar_mensaje(
                        f"Tema aplicado: [bold]{theme}[/bold]", "assistant"
                    )
                else:
                    chat_panel.agregar_mensaje(
                        f"Guardado '{theme}' como preferencia, pero Textual no tiene ese tema "
                        f"registrado nativamente, asi que no se ve reflejado visualmente todavia. "
                        f"Temas nativos disponibles: {', '.join(sorted(self.available_themes.keys()))}",
                        "assistant"
                    )

        self.push_screen(ThemeSelectorScreen(current_theme=current), _on_theme)

    def _abrir_mcp_selector(self) -> None:
        def _on_done(states: dict | None) -> None:
            if states is None:
                return
            # Persistir el estado de los toggles en mcp_servers.json
            import json, pathlib
            p = pathlib.Path(__file__).parent.parent / "core" / "config" / "mcp_servers.json"
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    for s in data:
                        name = s.get("name", s.get("url", ""))
                        if name in states:
                            s["enabled"] = states[name]
                    out = data
                elif isinstance(data, dict) and "servers" in data:
                    for s in data["servers"]:
                        name = s.get("name", s.get("url", ""))
                        if name in states:
                            s["enabled"] = states[name]
                    out = data
                elif isinstance(data, dict):
                    # Formato real: {"nombre_server": {config...}, ...}
                    for name, cfg in data.items():
                        if name in states and isinstance(cfg, dict):
                            cfg["enabled"] = states[name]
                    out = data
                else:
                    out = data
                p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
            except Exception:
                pass
            enabled = [k for k, v in states.items() if v]
            self.query_one("#chat_panel", ChatPanel).agregar_mensaje(
                f"MCPs activos: {', '.join(enabled) or 'ninguno'}", "assistant"
            )

        self.push_screen(McpSelectorScreen(on_done=_on_done), _on_done)

    def _abrir_effort_selector(self) -> None:
        def _on_effort(effort: str | None) -> None:
            if effort:
                self._current_effort = effort
                # Mapear effort → parámetros LLM.
                #
                # temperature: baja a medida que sube el effort (menos aleatoriedad,
                # respuestas más deterministas/precisas en 'max').
                #
                # num_predict: sube a medida que sube el effort. Este es el lever
                # real de "dejar que razone más" — es el presupuesto de tokens de
                # generación, que incluye el bloque <think> de Ornith. La temperatura
                # sola NO hace que el modelo piense más, solo cambia qué tan
                # aleatoria es cada eleccion de token; sin mas num_predict, 'max'
                # igual se cortaria al mismo largo que 'low'.
                #
                # num_ctx: sube en paralelo para que el razonamiento largo (mas
                # num_predict) tenga lugar en la ventana de contexto.
                effort_map = {
                    "low":    {"temperature": 0.9,  "num_ctx": 4096,  "num_predict": 768},
                    "medium": {"temperature": 0.6,  "num_ctx": 8192,  "num_predict": 2048},
                    "high":   {"temperature": 0.35, "num_ctx": 16384, "num_predict": 4096},
                    "max":    {"temperature": 0.15, "num_ctx": 32768, "num_predict": 8192},
                }
                from core.config.config_manager import get_config_manager
                config = get_config_manager()
                params = effort_map.get(effort, {})
                for k, v in params.items():
                    config.set(k.upper(), v, validate=False)
                self._actualizar_status_bar()
                self.query_one("#chat_panel", ChatPanel).agregar_mensaje(
                    f"Effort: [bold]{effort}[/bold] "
                    f"(temperatura {params.get('temperature')}, "
                    f"ctx {params.get('num_ctx')}, "
                    f"max tokens de razonamiento {params.get('num_predict')})",
                    "assistant"
                )

        self.push_screen(EffortSelectorScreen(current=self._current_effort), _on_effort)

    def _abrir_memory_editor(self) -> None:
        from core.memory.memory_manager import obtener_resumen, guardar_resumen

        texto_actual = obtener_resumen().get("texto", "")

        def _on_save(nuevo_texto: str | None) -> None:
            if nuevo_texto is None:
                return
            # Edición manual: no toca ultimo_turno_id (ver guardar_resumen),
            # así que la próxima consolidación automática sigue integrando
            # desde donde iba, sin re-procesar turnos ya vistos.
            guardar_resumen(nuevo_texto)
            self._motor_actualizar_resumen(nuevo_texto)
            self.query_one("#chat_panel", ChatPanel).agregar_mensaje(
                "Resumen de memoria actualizado.", "assistant"
            )

        self.push_screen(MemoryEditorScreen(texto_actual=texto_actual, on_save=_on_save), _on_save)

    def _forzar_consolidacion_memoria(self) -> None:
        from core.memory.consolidator import consolidar_resumen
        chat_panel = self.query_one("#chat_panel", ChatPanel)
        nuevo = consolidar_resumen(forzar=True)
        if nuevo is None:
            chat_panel.agregar_mensaje(
                "No había turnos nuevos para consolidar (o la consolidación falló; ver logs).",
                "assistant",
            )
        else:
            self._motor_actualizar_resumen(nuevo)
            chat_panel.agregar_mensaje("Resumen consolidado manualmente.", "assistant")

    def _motor_actualizar_resumen(self, texto: str) -> None:
        """Refleja el resumen nuevo en el dict RAM del motor (engine_bridge),
        para que el próximo turno ya lo vea sin esperar a recargar memoria."""
        try:
            from tui.engine_bridge import _motor
            _motor.mem["resumen"] = texto
        except Exception:
            pass

    def _abrir_agent_selector(self) -> None:
        def _on_agent(agent: str | None) -> None:
            if agent:
                self._current_agent = agent
                self._actualizar_status_bar()
                self.query_one("#chat_panel", ChatPanel).agregar_mensaje(
                    f"Agente: [bold]{agent}[/bold]", "assistant"
                )

        self.push_screen(AgentSelectorScreen(current_agent=self._current_agent), _on_agent)

    def _abrir_command_palette(self) -> None:
        def _on_cmd(cmd: str | None) -> None:
            if cmd:
                # Simular que el usuario escribió el comando
                self._manejar_comando_slash(cmd)

        self.push_screen(CommandPaletteScreen(), _on_cmd)

    def _nueva_sesion(self) -> None:
        """Limpia la vista y reinicia la sesión."""
        chat_panel = self.query_one("#chat_panel", ChatPanel)
        chat_panel.reset()
        self.query_one("#plan_panel", PlanPanel).reset()
        self.query_one("#debug_panel", DebugPanel).reset()
        chat_panel.agregar_mensaje("Nueva sesión iniciada.", "assistant")

    def _actualizar_status_bar(self) -> None:
        """Refresca la barra de estado con el modelo, agente y effort actuales."""
        from core.config.config_manager import get_config_manager
        config = get_config_manager()
        sb = self.query_one("#status_bar", StatusBar)
        sb.actualizar(
            model=config.get("MODELO", "ornith:9b"),
            provider="Ollama",
            agent=self._current_agent,
            effort=self._current_effort,
        )
        sb._repintar()

    def _cambiar_vista_historial(self, arg: str) -> None:
        from core.memory.memory_manager import obtener_turnos_por_sesion
        chat_panel = self.query_one("#chat_panel", ChatPanel)
        arg = arg.strip().lower()

        if arg in ("", "actual", "vivo", "live"):
            chat_panel.volver_a_sesion_actual()
            return

        if not arg.isdigit() or not self._ultimas_sesiones:
            chat_panel.agregar_mensaje("Usá '/sesiones' primero, después '/historial <n>'.", "assistant")
            return

        idx = int(arg) - 1
        if idx < 0 or idx >= len(self._ultimas_sesiones):
            chat_panel.agregar_mensaje(f"No hay sesión #{arg}. Probá '/sesiones' de nuevo.", "assistant")
            return

        sesion = self._ultimas_sesiones[idx]
        turnos = obtener_turnos_por_sesion(sesion["sesion_id"])
        chat_panel.mostrar_sesion(turnos, etiqueta=f"{sesion['inicio']} ({sesion['turnos']} turnos)")

    @work(thread=True, exclusive=True)
    def _procesar_orden_bg(self, orden: str) -> None:
        for evento in iter_eventos(orden):
            self.call_from_thread(self._manejar_evento, evento)

    def _manejar_evento(self, evento) -> None:
        chat_panel = self.query_one("#chat_panel", ChatPanel)
        streaming_line = self.query_one("#streaming_line", Static)

        if isinstance(evento, TokenEvent):
            self._respuesta_en_curso += evento.fragmento
            # El streaming se ve en vivo dentro del chat, bajo "Aether:".
            # streaming_line queda solo para estados ([•]/[✓]/flavor).
            chat_panel.mostrar_streaming(self._respuesta_en_curso)

        elif isinstance(evento, StdoutLineEvent):
            self.query_one("#debug_panel", DebugPanel).agregar_log(evento.texto)

        elif isinstance(evento, NodeUpdateEvent):
            estado = real_state_for_node(evento.nodo, evento.delta)
            if estado:
                self._mostrar_estado_real(estado)
            plan_panel = self.query_one("#plan_panel", PlanPanel)
            if hasattr(plan_panel, "actualizar_delta"):
                plan_panel.actualizar_delta(evento.nodo, evento.delta)

        elif isinstance(evento, DoneEvent):
            # FIX (parche): priorizar la respuesta final limpia (parseada por
            # _parse_ornith_thinking) sobre el streaming crudo acumulado en
            # _respuesta_en_curso, que puede incluir el bloque 
            # completo del modelo. Además, el DoneEvent llega INMEDIATO al
            # terminar el grafo (ver engine_bridge), así que el usuario puede
            # volver a escribir sin esperar la consolidación de memoria.
            chat_panel.ocultar_streaming()
            chat_panel.agregar_mensaje(limpiar_respuesta_chat(evento.respuesta), "assistant")
            streaming_line.update("")
            self._mostrar_estado_real("[✓] Operación completada")
            self._finalizar_turno()

        elif isinstance(evento, ErrorEvent):
            prefijo = "⏹" if "cancelada" in evento.mensaje.lower() else "⚠️ Error:"
            chat_panel.ocultar_streaming()
            chat_panel.agregar_mensaje(f"{prefijo} {evento.mensaje}", "assistant")
            streaming_line.update("")
            self._mostrar_estado_real("[!] Se detectó un error.")
            self._finalizar_turno()

    def _mostrar_estado_real(self, estado: str) -> None:
        self._estado_real = estado
        # streaming_line es Static(markup=True) por defecto: sanear corchetes
        # ([•], [✓], [!]) y ANSI antes de actualizar para evitar MarkupError.
        self.query_one("#streaming_line", Static).update(_sanear_streaming(estado))

    def _mostrar_flavor(self) -> None:
        if not self._thinking:
            return
        if time.monotonic() - self._inicio_operacion < STATUS_CONFIG["min_operation_seconds"]:
            return
        flavor = choose_flavor()
        if flavor:
            self.query_one("#streaming_line", Static).update(
                f"{_sanear_streaming(self._estado_real)}\n  {_sanear_streaming(flavor)}"
            )

    def _mostrar_banner_dir(self) -> None:
        """Banner de sesión: dónde trabaja + estado de autorización."""
        try:
            from core.config.dir_authorization import (
                esta_autorizado, fue_evaluada, resolver_dir_trabajo)
            ruta = resolver_dir_trabajo()
            if esta_autorizado(ruta):
                estado = "autorizado"
            elif fue_evaluada(ruta):
                estado = "denegado por el Creador"
            else:
                estado = "dir. nuevo (registrado)"
            try:
                chat = self.query_one("#chat_panel", ChatPanel)
                chat.agregar_mensaje(
                    f"Trabajando en: {ruta}  ({estado})\n"
                    f"   Cambia con: aether --workdir <ruta>  |  dirs: ~/.aether/allowed_dirs.json",
                    "assistant")
            except Exception:
                pass
            try:
                self.title = f"Aether - {ruta.name} [{ruta}]"
            except Exception:
                pass
        except Exception:
            pass

    def _finalizar_turno(self) -> None:
        self._thinking = False
        self._respuesta_en_curso = ""
        self._startup_tick = 0
        self.query_one("#streaming_line", Static).update(self._estado_real)
        input_widget = self.query_one("#input_chat", Input)
        input_widget.disabled = False
        input_widget.placeholder = "Escribí tu mensaje aquí..."
        self.query_one("#btn_detener", Button).disabled = True
        input_widget.focus()



def run(workdir: str | None = None) -> None:
    app = AetherApp(workdir=workdir)
    app.run()