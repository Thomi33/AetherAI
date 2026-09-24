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

from core.config.settings import TIMEOUT_CMD

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
        return salida or "[Comando completado sin salida]", resultado.returncode != 0

    except subprocess.TimeoutExpired:
        return f"Tiempo límite de {TIMEOUT_CMD}s agotado. Proceso interrumpido.", True
    except Exception as e:
        return f"Error de subproceso: {e}", True