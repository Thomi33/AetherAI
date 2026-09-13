"""
Widget Plan Panel para mostrar el progreso de ejecución del plan en la TUI.
Muestra pasos completados y actualiza visualmente según eventos del motor.

Además es el "qué está haciendo Aether en vivo": cada NodeUpdateEvent del
grafo (planner / plan_executor / agent_loop) deja acá una línea legible de
actividad — crear/escribir/leer/listar archivos y carpetas, shell, web, etc.
Los nodos fs_* no son nodos propios del grafo (corren DENTRO de agent_loop
o plan_executor vía get_node_func), así que la única forma de verlos en vivo
es leer `agent_pasos_log` / `tool_actual` / `fs_result` del delta.
"""


import re

from textual.widgets import Static

_RE_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(texto: str) -> str:
    return _RE_ANSI.sub("", texto)


def _escapar_corchetes(texto: str) -> str:
    # Static es markup=True por defecto: cualquier [TAG] literal (rutas con
    # corchetes, [FS], [DIR], JSON...) tira MarkupError si no se escapa.
    return (texto
            .replace(chr(92), chr(92) + chr(92))
            .replace('[', chr(92) + '[')
            .replace(']', chr(92) + ']'))


_ICONOS_TOOL = {
    "text": "💬", "web": "🔍", "shell": "🖥️", "launch": "🚀",
    "vision": "👁️", "codigo": "💻", "memory": "🧩",
    "file_write": "💾", "extract": "🧹", "mcp": "🔌",
    "computer_use": "🖱️",
    "fs_write": "💾", "fs_read": "📖", "fs_mkdir": "📁", "fs_list": "📂",
}

# Etiqueta humana por tool para la línea de actividad.
_ETIQUETAS_TOOL = {
    "fs_write": "Escribiendo archivo",
    "fs_read": "Leyendo archivo",
    "fs_mkdir": "Creando carpeta",
    "fs_list": "Listando carpeta",
    "file_write": "Guardando archivo",
    "shell": "Ejecutando comando",
    "codigo": "Generando código",
    "web": "Buscando en la web",
    "vision": "Analizando pantalla",
    "mcp": "Llamando MCP",
    "computer_use": "Controlando interfaz",
    "launch": "Abriendo app",
    "memory": "Gestionando memoria",
    "text": "Respondiendo",
}


def _resumir_args(tool: str, args: object) -> str:
    """Saca lo más informativo de los args: path/files/command/query/app."""
    if not isinstance(args, dict):
        return ""
    # fs_write multi-archivo: mostrar cuántos + el primero.
    files = args.get("files")
    if isinstance(files, list) and files:
        rutas = []
        for item in files[:3]:
            if isinstance(item, dict):
                rutas.append(str(item.get("path") or item.get("filename") or "?"))
        extra = f" (+{len(files) - 3} más)" if len(files) > 3 else ""
        return f"{len(files)} archivos: {', '.join(rutas)}{extra}"
    for clave in ("path", "filename", "command", "query", "app", "instruccion"):
        val = args.get(clave)
        if isinstance(val, str) and val.strip():
            v = val.strip()
            if len(v) > 80:
                v = v[:80] + "…"
            return v
    return ""


def _linea_de_paso(tool: str, args: object, resultado: object = "") -> str:
    icono = _ICONOS_TOOL.get(tool, "•")
    etiqueta = _ETIQUETAS_TOOL.get(tool, tool)
    detalle = _resumir_args(tool, args)
    linea = f"{icono} {etiqueta}"
    if detalle:
        linea += f" → {detalle}"
    # Resultado corto: "Guardado en ...", "Directorio creado: ...", errores.
    if isinstance(resultado, str) and resultado.strip():
        primera = resultado.strip().split("\n", 1)[0]
        if len(primera) > 90:
            primera = primera[:90] + "…"
        # Evitar duplicar la ruta si ya está en el detalle.
        if primera and primera not in (detalle or ""):
            # Para fs_read el contenido crudo no aporta: solo contar chars.
            if tool == "fs_read":
                linea += f" ({len(resultado)} caracteres)"
            # Para fs_list mostrar cantidad de entradas.
            elif tool == "fs_list":
                n = len([l for l in resultado.strip().splitlines() if l.strip()])
                linea += f" ({n} entradas)"
            else:
                linea += f" — {primera}"
        elif tool == "fs_read" and isinstance(resultado, str):
            linea += f" ({len(resultado)} caracteres)"
    return linea


class PlanPanel(Static):
    """Panel que muestra el estado del plan de ejecucion + actividad en vivo."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._plan_pasos = []
        self._plan_index = 0
        self._plan_activo = False
        self._actividad: list[str] = []
        self._firmas_vistas: set[str] = set()
        self._max_actividad = 4  # visibles (el CSS le da height: 6)

    def actualizar(self, plan_pasos=None, plan_index=None, plan_activo=None):
        """Actualiza los datos del panel de plan."""
        if plan_pasos is not None:
            self._plan_pasos = plan_pasos
        if plan_index is not None:
            self._plan_index = plan_index
        if plan_activo is not None:
            self._plan_activo = plan_activo
        self._repintar()

    def actualizar_delta(self, nodo: str, delta: dict | None) -> None:
        """Consume un NodeUpdateEvent y deja actividad legible en vivo."""
        if not isinstance(delta, dict):
            return
        key = (nodo or "").rsplit(".", 1)[-1]

        if key == "planner":
            if isinstance(delta.get("plan_pasos"), list):
                self._plan_pasos = delta["plan_pasos"]
            if isinstance(delta.get("plan_index"), int):
                self._plan_index = delta["plan_index"]
            if isinstance(delta.get("plan_activo"), bool):
                self._plan_activo = delta["plan_activo"]
            self._repintar()
            return

        if key == "plan_executor":
            tool = delta.get("tool_actual") or "?"
            args: object = {}
            try:
                idx = int(delta.get("plan_index", 0)) - 1
                pasos = self._plan_pasos or []
                if 0 <= idx < len(pasos) and isinstance(pasos[idx], dict):
                    args = (pasos[idx].get("args") or {})
            except Exception:
                args = {}
            resultado = (
                delta.get("fs_result")
                or delta.get("shell_output")
                or delta.get("final_response")
                or ""
            )
            if isinstance(resultado, str) and len(resultado) > 400:
                resultado = resultado[:400]
            self._agregar_actividad(_linea_de_paso(str(tool), args, resultado))
            if isinstance(delta.get("plan_index"), int):
                self._plan_index = delta["plan_index"]
            self._repintar()
            return

        if key == "agent_loop":
            pasos = delta.get("agent_pasos_log")
            if isinstance(pasos, list):
                for paso in pasos:
                    if not isinstance(paso, dict):
                        continue
                    firma = f"{paso.get('tool')}|{paso.get('args')}|{str(paso.get('resultado') or '')[:80]}"
                    if firma in self._firmas_vistas:
                        continue
                    self._firmas_vistas.add(firma)
                    self._agregar_actividad(_linea_de_paso(
                        str(paso.get("tool") or "?"),
                        paso.get("args") or {},
                        paso.get("resultado") or "",
                    ))
                if len(self._firmas_vistas) > 64:
                    self._firmas_vistas = set(list(self._firmas_vistas)[-64:])
                self._repintar()
            return

        if isinstance(delta.get("plan_pasos"), list):
            self._plan_pasos = delta["plan_pasos"]
            self._repintar()

    def _agregar_actividad(self, linea: str) -> None:
        linea = _strip_ansi(linea).strip()
        if not linea:
            return
        if self._actividad and self._actividad[-1] == linea:
            return
        self._actividad.append(linea)
        if len(self._actividad) > 12:
            self._actividad = self._actividad[-12:]

    def _repintar(self):
        """Reconstruye el contenido visual del panel."""
        partes: list[str] = []
        if self._plan_pasos:
            for i, paso in enumerate(self._plan_pasos):
                if isinstance(paso, dict):
                    tool = paso.get("tool", "?")
                else:
                    tool = str(paso)
                icono = _ICONOS_TOOL.get(tool, "•")
                estado = "✓" if (i + 1) <= self._plan_index else "○"
                partes.append(f"{estado} {icono} {tool}")
        for linea in self._actividad[-self._max_actividad:]:
            partes.append(f"▶ {linea}")
        if not partes:
            texto = "planificando..." if self._plan_activo else ""
            if self._plan_activo and not texto:
                texto = "Ejecutando..."
            self.update(_escapar_corchetes(_strip_ansi(texto)))
            return
        self.update(_escapar_corchetes(_strip_ansi("\n".join(partes))))

    def reset(self):
        """Resetea el panel a su estado inicial."""
        self._plan_pasos = []
        self._plan_index = 0
        self._plan_activo = False
        self._actividad = []
        self._firmas_vistas = set()
        self._repintar()

    @property
    def plan_pasos(self) -> list:
        return self._plan_pasos

    @property
    def plan_index(self) -> int:
        return self._plan_index

    @property
    def plan_activo(self) -> bool:
        return self._plan_activo