"""
Ejecutor de Comandos de Consola de Aether en Zsh.

Controla de forma segura la ejecución de subprocesos en el sistema operativo,
previniendo la ejecución de comandos peligrosos, aislando las aplicaciones GUI
en segundo plano y bloqueando de forma absoluta editores de texto interactivos
para evitar congelamientos.
"""
import os
import re
import subprocess
import tempfile
import time

from core.config.settings import (
    TIMEOUT_CMD,
    MAX_KB_SALIDA_TOOL,
    MAX_LINEAS_SALIDA_TOOL,
)
from core.agent.grounding import (
    dominios_comando_no_permitidos,
    truncar_lineas,
)

# Patrones destructivos que el sistema impedirá ejecutar por seguridad
_PATRONES_PELIGROSOS = [
    r"rm\s+-rf\s+/(?:\s|$)",
    r"mkfs\.",
    r"dd\s+if=",
    r"chmod\s+[0-7]{3,4}\s+/",
    r":\(\)\{.*\}",
    r"kill\s+-9\s+1\b",
    r"> /dev/sd[a-z]",
]

# Lista de editores interactivos prohibidos a nivel de sistema
EDITORES_BANEADOS = ["nano", "vim", "vi", "micro", "emacs"]

# Lanzadores gráficos o herramientas de monitoreo interactivo que deben correr en segundo plano.
#
# FIX (bug real, sesión 2026-09-04): antes esto incluía
# `python\s+-m\s+|python3?\s+\S+\.py` y `\./`, así que CUALQUIER ejecución
# de un script Python (ej. "python3 /tmp/escribir.py", "python3
# temp_converter.py") o de un ejecutable local ("./algo") se clasificaba
# como lanzamiento GUI: se corría en background con stdout a DEVNULL y sin
# esperar a que termine de verdad. Esto rompió tareas de escritura de
# archivos vía heredoc (el ls posterior corría contra un archivo que todavía
# no se había terminado de escribir, o el error quedaba silenciado) y
# cualquier test rápido de un script ("python3 app.py") quedaba corriendo
# en background en vez de mostrar su salida/error de forma síncrona.
#
# La detección de GUI ahora se limita a apps conocidas por nombre. Si en el
# futuro hace falta lanzar un script Python como proceso GUI de larga
# duración, el propio comando puede pedirlo explícitamente con `&` al final
# (ver wants_pid_capture / "echo $!" más abajo) en vez de adivinarlo por el
# nombre del archivo.
_LANZADORES_GUI = re.compile(
    r"\b(flatpak\s+run|steam|lutris|heroic|bottles|gamescope"
    r"|nvtop|btop|htop|glxgears|obs|kdenlive|gimp|inkscape"
    r"|brave|electron|appimage|prismlauncher"
    r")",
    re.IGNORECASE,
)


def _es_peligroso(cmd: str) -> bool:
    """Detecta patrones de comandos potencialmente destructivos."""
    return any(re.search(p, cmd, re.IGNORECASE) for p in _PATRONES_PELIGROSOS)


# ── Allowlist de dominios (curl/wget/http) ────────────────────────────────

def validar_comando_red(cmd: str) -> tuple[bool, str]:
    """
    Rechaza comandos de red (curl/wget/http) cuyo destino NO esté en la
    allowlist de dominios de Aether (settings.ALLOWLIST_DOMINIOS_RED).

    Fix del bug "curl inventado": el modelo cerró una síntesis sugiriendo un
    `curl` a api.threatconnect.com, un dominio inexistente que él mismo
    había alucinado. Cualquier destino fuera de la allowlist se rechaza con
    aviso; NO se ejecuta nada del comando en ese caso.

    Returns (ok, motivo) — motivo solo si ok=False.
    """
    bloqueados = dominios_comando_no_permitidos(cmd or "")
    if bloqueados:
        return False, (
            f"dominio(s) no permitido(s) por la allowlist de Aether: "
            f"{', '.join(bloqueados)}. Agregalo/s a ALLOWLIST_DOMINIOS_RED en "
            f"config.json si de verdad los necesitás."
        )
    return True, ""


# ── Normalización de relevadores de red (evita volcar sockets unix) ───────

def _segmento_tiene_flag(segmento_tokens: list[str], letra: str) -> bool:
    """True si algún flag corto (-t, -tunap, ...) o largo (--tcp) del
    segmento contiene la opción dada."""
    for tok in segmento_tokens:
        if not tok.startswith("-"):
            continue
        if tok.startswith("--"):
            if letra in tok[2:]:
                return True
        elif letra in tok[1:]:
            return True
    return False


def normalizar_comando_red(cmd: str) -> str:
    """
    `netstat`/`ss` genéricos vuelcan también los sockets UNIX (cientos de
    líneas irrelevantes para revisar tráfico de red). Si el relevador no
    pide explícitamente TCP ni UDP, se agregan -t -u al PRIMER segmento del
    pipe (grep y compañía quedan intactos). Nunca toca comandos que ya
    filtran (-t, -u, --tcp, --udp, -tunap, ...) ni los que piden sockets
    unix a propósito (-x, --unix): en ese caso el volcado es intencional.
    La reescritura se loguea.
    """
    if not cmd:
        return cmd
    m = re.match(r"^(?P<primero>[^|]*?)(?P<resto>\s*\|.*)?$", cmd, re.DOTALL)
    if not m:
        return cmd
    resto = m.group("resto") or ""
    primer = m.group("primero").strip()
    tokens = primer.split()
    if len(tokens) < 1:
        return cmd
    base_idx = 0
    # tolerar prefijos: sudo y asignaciones de entorno
    while base_idx < len(tokens) and (
        tokens[base_idx] == "sudo"
        or ("=" in tokens[base_idx] and not tokens[base_idx].startswith("-"))
    ):
        base_idx += 1
    if base_idx >= len(tokens):
        return cmd
    base = tokens[base_idx].lower()
    if base not in ("netstat", "ss"):
        return cmd
    if _segmento_tiene_flag(tokens, "t") or _segmento_tiene_flag(tokens, "u"):
        return cmd
    # -x / --unix: pidió sockets unix explícitamente → no meter -t -u
    if _segmento_tiene_flag(tokens, "x"):
        return cmd
    tokens.insert(base_idx + 1, "-t")
    tokens.insert(base_idx + 2, "-u")
    normalizado = " ".join(tokens) + resto
    if normalizado != cmd:
        print(f"   └─ [SHELL]: relevador de red sin filtro de protocolo; "
              f"agregando -t -u para no volcar sockets unix: {normalizado}")
    return normalizado


def _truncar_salida_tool(salida: str) -> str:
    """
    Tope configurable de salida: MAX_LINEAS_SALIDA_TOOL líneas y
    MAX_KB_SALIDA_TOOL KB, con aviso honesto de lo omitido. Evita que un
    `netstat -an` (o un cat de un archivo gigante) arruine el contexto del
    modelo con miles de líneas.
    """
    truncada, aviso = truncar_lineas(salida, MAX_LINEAS_SALIDA_TOOL)
    if aviso:
        truncada = f"{truncada}\n{aviso}"
    max_bytes = int(MAX_KB_SALIDA_TOOL) * 1024
    if max_bytes > 0 and len(truncada.encode("utf-8", errors="replace")) > max_bytes:
        cruda = truncada.encode("utf-8", errors="replace")[:max_bytes].decode(
            "utf-8", errors="ignore")
        omitidos = len(truncada) - len(cruda)
        truncada = f"{cruda}\n[truncado: {omitidos} caracteres omitidos (tope de {MAX_KB_SALIDA_TOOL} KB)]"
    return truncada


def ejecutar_comando(cmd: str, cwd: str | None = None) -> tuple[str, bool]:
    """
    Ejecuta comando en zsh con protección de seguridad integrada.
    Retorna (salida, hubo_error).
    `cwd`: directorio de ejecución. Si es None usa RUTA_TRABAJO (el dir
    desde el que se invocó a Aether), para que `ls`, `cat archivo.txt`,
    `./script.sh` etc. operen sobre ~/Documents, ~/Proyecto, ... según
    corresponda en vez de sobre la carpeta del proyecto Aether.
    """
    # 1. Validación de seguridad contra comandos destructivos
    if _es_peligroso(cmd):
        return "⛔ CANCELADO: Operación identificada como potencialmente destructiva.", True

    # 1a. Allowlist de dominios: cualquier curl/wget/http fuera de la lista
    # se rechaza ANTES de ejecutar nada (fix del curl inventado).
    ok_red, motivo_red = validar_comando_red(cmd)
    if not ok_red:
        return f"⛔ CANCELADO: {motivo_red}", True

    # 1a-bis. netstat/ss genéricos: preferir -t -u (no volcar sockets unix).
    cmd = normalizar_comando_red(cmd)

    # 1b. Directorio de ejecución: RUTA_TRABAJO por defecto (lazy: la env
    # AETHER_CWD puede setearse después de los imports, así que NO se
    # cachea en import — se lee acá en cada llamada).
    if cwd is None:
        try:
            from core.config import settings as _s
            _rt = _s.RUTA_TRABAJO
            cwd = str(_rt) if _rt.is_dir() else None
        except Exception:
            cwd = None

    # 2. Interceptor de editores interactivos (Baneo a nivel de sistema)
    # Evita que se congele esperando entrada manual del usuario
    cmd_limpio = cmd.strip()
    for editor in EDITORES_BANEADOS:
        patron = rf"\b{editor}\b"
        if re.search(patron, cmd_limpio.lower()):
            error_sistema = (
                f"🚫 [SISTEMA - OPERACIÓN BLOQUEADA]:\n"
                f"Se denegó la ejecución del comando porque contiene el editor interactivo '{editor}'.\n"
                f"La terminal de ejecución no es interactiva. Sigue estas pautas:\n"
                f"  - Para LEER archivos mostrando líneas: Usa 'cat -n <archivo>'\n"
                f"  - Para EDITAR o REEMPLAZAR texto: Usa 'sed -i' o scripts inline de Python (python3 -c '...')"
            )
            return error_sistema, True

    cmd_limpio = cmd.strip()

    # No tratar como lanzamiento GUI las consultas de descubrimiento (which, command -v, etc.)
    # Esto evita que "which prismlauncher" o "command -v xxx" se confundan con lanzar el programa.
    if re.search(r'^\s*(which|command\s+-v|command\s+-V|type\s+-p|whereis)\b', cmd_limpio, re.IGNORECASE):
        es_gui = False
    else:
        es_gui = bool(_LANZADORES_GUI.search(cmd_limpio))

    try:
        # 3. Lógica para aplicaciones GUI (Lanzamiento seguro en segundo plano)
        if es_gui:
            # Detectamos si el comando está diseñado para capturar PID de background (launch path)
            wants_pid_capture = "echo $!" in cmd or "& echo" in cmd or "echo $!" in cmd_limpio

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jarvis_err")
            stdout_target = subprocess.PIPE if wants_pid_capture else subprocess.DEVNULL
            proc = subprocess.Popen(
                cmd, shell=True, executable="/bin/zsh",
                stdout=stdout_target, stderr=tmp,
                preexec_fn=os.setpgrp,
                cwd=cwd or None,
            )
            tmp.close()

            if wants_pid_capture:
                # Para lanzamientos con `foo > /dev/null & echo $!`, leemos el PID que imprimió el shell
                try:
                    out, _ = proc.communicate(timeout=3)
                    pid_output = (out or b"").decode(errors="ignore").strip()
                    for tok in reversed(pid_output.split()):
                        if tok.isdigit():
                            if os.path.exists(tmp.name):
                                os.unlink(tmp.name)
                            return tok, False
                except Exception:
                    pass
                # Si no pudimos obtener el PID del echo, caemos al comportamiento normal

            time.sleep(1.5)
            if proc.poll() is not None and proc.returncode != 0:
                with open(tmp.name) as f:
                    err = f.read().strip()
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                return f"El proceso terminó con error:\n{err}", True
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            return f"[Proceso lanzado en segundo plano. PID: {proc.pid}]", False

        # 4. Lógica para comandos estándar interactivos
        #    Sin text=True: se decodifica a mano con errors='replace' porque
        #    la salida puede ser binaria (ej. `cat foto.jpg` generado por un
        #    modelo confundido). Antes, 'utf-8' lanzaba UnicodeDecodeError y
        #    el modelo recibía un "Error de subproceso" sin contexto útil.
        resultado = subprocess.run(
            cmd, shell=True, executable="/bin/zsh",
            capture_output=True, timeout=TIMEOUT_CMD,
            cwd=cwd or None,
        )
        raw = (resultado.stdout or b"") + (resultado.stderr or b"")
        salida = raw.decode("utf-8", errors="replace").strip()
        if "\ufffd" in salida:
            # Contenido binario: recortar y avisar — el modelo entiende este
            # mensaje mucho mejor que un stream de caracteres de reemplazo.
            if sum(1 for c in salida if c == "\ufffd") > 16:
                salida = (salida[:1000]
                          + "\n…[salida binaria truncada: el archivo no es texto]")
        # Tope configurable de salida (líneas/KB) con aviso honesto: sin
        # esto un `netstat -an` completo satura el contexto del modelo.
        salida = _truncar_salida_tool(salida)
        return salida or "[Comando completado sin salida]", resultado.returncode != 0

    except subprocess.TimeoutExpired:
        return f"Tiempo límite de {TIMEOUT_CMD}s agotado. Proceso interrumpido.", True
    except Exception as e:
        return f"Error de subproceso: {e}", True