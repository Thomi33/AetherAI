"""
net_audit.py — Auditoría de RED de solo lectura para Aether.

Motivación (bug real): ante "revisá mi tráfico de red, a ver si no tengo algo
sospechoso", el agent loop terminaba corriendo `netstat -an | grep ...` a
cegas y luego la síntesis alucinaba una intrusión completa. Esta tool releva
la red de forma ESTRUCTURADA y determinista; el LLM solo resume el JSON.

Qué hace (todo READ-ONLY, sin sudo, sin cambios en el sistema):
    1. `ss -tunap` (degrada a `ss -tun` si no hay permisos de proceso y
       avisa; NUNCA escala con sudo) → conexiones establecidas, cada una con
       proto/local/remoto/estado/proceso/pid y clase (loopback|lan|externa).
    2. `ss -tunlp` → puertos en escucha, marcando expuestos (0.0.0.0/::/*)
       vs. loopback.
    3. `journalctl -u sshd` (o sshd.service / ssh.service según exista) →
       intentos fallidos recientes. Si no hay acceso → `ssh_fallidos: null`
       con el motivo.
    4. `last -n 10` → últimos logins.
    5. mtime de ~/.ssh/authorized_keys si existe.

Cada subcomprobación falla de forma AISLADA: su error queda en
`errores[<subcheck>]` y el resto del JSON sigue válido.
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path


_TIMEOUT = 15  # seg por subcomando


def _run(cmd: str, runner=None) -> tuple[str, int]:
    """Ejecuta un comando read-only y devuelve (stdout, returncode).
    `runner` inyectable para tests."""
    if runner is not None:
        return runner(cmd)
    try:
        r = subprocess.run(
            cmd, shell=True, executable="/bin/zsh",
            capture_output=True, timeout=_TIMEOUT,
        )
        raw = (r.stdout or b"") + (r.stderr or b"")
        return raw.decode("utf-8", errors="replace"), r.returncode
    except Exception as e:
        return f"(error ejecutando: {e})", 124


# ── Clasificación de direcciones ──────────────────────────────────────────

_RE_PRIVADA = (
    re.compile(r"^10\."),                        # 10/8
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),   # 172.16/12
    re.compile(r"^192\.168\."),                  # 192.168/16
    re.compile(r"^169\.254\."),                  # link-local
    re.compile(r"^fe[89ab]", re.IGNORECASE),     # fe80::/10 link-local IPv6
    re.compile(r"^f[cd]", re.IGNORECASE),        # fc00::/7 ULA IPv6
)


def clasificar_direccion(host: str) -> str:
    """Clase de una IP/host: loopback | lan | externa."""
    h = (host or "").strip().strip("[]").lower()
    if not h or h == "*":
        return "desconocida"
    if h in ("127.0.0.1", "::1", "localhost") or h.startswith("127."):
        return "loopback"
    if any(p.match(h) for p in _RE_PRIVADA):
        return "lan"
    return "externa"


def _parse_endpoint(endpoint: str) -> tuple[str, str]:
    """'192.168.1.10:54321' o '[::1]:22' o '*:631' → (host, puerto)."""
    e = (endpoint or "").strip()
    if e.startswith("["):                       # IPv6 [::1]:22
        host, _, resto = e.partition("]:")
        return host.strip("[]"), resto or ""
    host, sep, puerto = e.rpartition(":")
    if not sep:                                 # sin puerto
        return e, ""
    return host, puerto


def _parse_proceso(col_proceso: str) -> tuple[str, str]:
    """Columna Process de ss: users:(("sshd",pid=841,fd=3)) → (nombre, pid)."""
    m = re.search(r'\("([^"]+)",pid=(\d+)', col_proceso or "")
    if m:
        return m.group(1), m.group(2)
    return "", ""


def _parse_ss_salida(salida: str) -> list[dict]:
    """
    Parsea la salida de ss (columnas: Netid State Recv-Q Send-Q Local Peer
    [Process]). Devuelve filas normalizadas; el header se descarta.
    """
    filas: list[dict] = []
    lineas = [l for l in (salida or "").splitlines() if l.strip()]
    for linea in lineas:
        partes = linea.split()
        if len(partes) < 5:
            continue
        if partes[0].lower() not in ("tcp", "udp", "tcp6", "udp6", "vsock"):
            continue
        proto, estado = partes[0], partes[1]
        local = partes[4]
        remoto = partes[5] if len(partes) > 5 else ""
        proceso_crudo = " ".join(partes[6:]) if len(partes) > 6 else ""
        nombre, pid = _parse_proceso(proceso_crudo)
        l_host, l_puerto = _parse_endpoint(local)
        r_host, r_puerto = _parse_endpoint(remoto) if remoto else ("", "")
        filas.append({
            "proto": proto,
            "estado": estado,
            "local": local,
            "local_host": l_host,
            "local_puerto": l_puerto,
            "remoto": remoto or None,
            "remoto_host": r_host,
            "remoto_puerto": r_puerto,
            "proceso": nombre or None,
            "pid": pid or None,
        })
    return filas


# ═══════════════════════════════════════════════════════════════════════
# SUBCHEQUEOS (cada uno aislado: falla propia, error en errores[...])
# ═══════════════════════════════════════════════════════════════════════

def _subconexiones(runner) -> dict:
    """Conexiones establecidas con clase loopback/lan/externa.
    `ss -tunap`; si falla (o no hay permisos de proceso), degrada a
    `ss -tun` y avisa — NUNCA escala con sudo."""
    salida, rc = _run("ss -tunap", runner)
    nota_procesos = None
    filas = _parse_ss_salida(salida)
    if rc != 0 or not filas:
        salida, rc = _run("ss -tun", runner)
        filas = _parse_ss_salida(salida)
        if rc == 0:
            nota_procesos = ("sin datos de proceso: se necesita root para ver "
                             "los procesos de sockets ajenos (no se escala "
                             "con sudo por política)")
    if rc != 0 and not filas:
        raise RuntimeError(f"ss falló (rc={rc}): {(salida or '')[:120]}")

    establecidas = []
    for f in filas:
        if f["estado"].upper() not in ("ESTAB", "ESTABLISHED"):
            continue
        clase = clasificar_direccion(f["remoto_host"])
        establecidas.append({
            "proto": f["proto"],
            "local": f["local"],
            "remoto": f["remoto"],
            "remoto_host": f["remoto_host"],
            "remoto_puerto": f["remoto_puerto"],
            "estado": f["estado"],
            "proceso": f["proceso"],
            "pid": f["pid"],
            "clase": clase,
        })
    return {"conexiones_establecidas": establecidas, "nota_procesos": nota_procesos}


def _subescucha(runner) -> list[dict]:
    """Puertos en escucha: expuestos (0.0.0.0/::/*) vs. loopback."""
    salida, rc = _run("ss -tunlp", runner)
    if rc != 0:
        salida, rc = _run("ss -tunl", runner)
    if rc != 0:
        raise RuntimeError(f"ss -tunl(p) falló (rc={rc}): {(salida or '')[:120]}")
    puertos: list[dict] = []
    vistos: set[tuple[str, str]] = set()
    for f in _parse_ss_salida(salida):
        estado = f["estado"].upper()
        # TCP: LISTEN; UDP "en escucha" aparece como UNCONN
        if estado not in ("LISTEN", "UNCONN", "LISTENING"):
            continue
        host = f["local_host"]
        puerto = f["local_puerto"]
        clave = (f["proto"], puerto)
        if clave in vistos:
            continue
        vistos.add(clave)
        expuesto = host in ("0.0.0.0", "*", "::", "::*", "")
        puertos.append({
            "proto": f["proto"],
            "local": f["local"],
            "puerto": puerto,
            "expuesto": expuesto,
            "nota": ("bindeado a 0.0.0.0/:: (accesible desde fuera)"
                     if expuesto else "bindeado a loopback/IP concreta"),
            "proceso": f["proceso"],
        })
    return puertos


_RE_FALLO_SSH = re.compile(
    r"(?i)failed password|invalid user|authentication failure|"
    r"connection closed by authenticating user|maximum authentication attempts"
)


def _sub_ssh_fallidos(runner) -> dict:
    """Intentos fallidos de SSH recientes vía journalctl. Si no hay acceso a
    ningún unit sshd/ssh → null con motivo (no tumba el resto)."""
    unidades = ("sshd", "sshd.service", "ssh.service")
    for unidad in unidades:
        salida, rc = _run(f"journalctl -u {unidad} -n 200 --no-pager", runner)
        texto = (salida or "").lower()
        if rc == 0 and "no journal files" not in texto \
                and "could not be found" not in texto:
            lineas_fallo = [l for l in (salida or "").splitlines()
                            if _RE_FALLO_SSH.search(l)]
            return {
                "unidad": unidad,
                "ssh_fallidos": len(lineas_fallo),
                "ultimo_fallo": lineas_fallo[-1][:200] if lineas_fallo else None,
            }
    return {
        "unidad": None,
        "ssh_fallidos": None,
        "motivo": ("sin acceso al journal de sshd/ssh (permisos o "
                   "systemd-journal no disponible para este usuario)"),
    }


def _sub_last_logins(runner) -> list[str]:
    """`last -n 10` crudo (líneas), para que el LLM las reporte."""
    salida, rc = _run("last -n 10 --time-format iso", runner)
    if rc != 0:
        salida, rc = _run("last -n 10", runner)
    if rc != 0:
        raise RuntimeError(f"last falló (rc={rc}): {(salida or '')[:120]}")
    return [l for l in (salida or "").splitlines() if l.strip()]


def _sub_authorized_keys() -> dict:
    """mtime/tamaño de ~/.ssh/authorized_keys si existe."""
    ruta = Path.home() / ".ssh" / "authorized_keys"
    if not ruta.exists():
        return {"ruta": str(ruta), "existe": False}
    st = ruta.stat()
    return {
        "ruta": str(ruta),
        "existe": True,
        "modificado": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "bytes": st.st_size,
    }


# ═══════════════════════════════════════════════════════════════════════
# HALLAZGOS Y VEREDICTO DETERMINISTAS
# ═══════════════════════════════════════════════════════════════════════

def _evaluar_hallazgos(conexiones, puertos, ssh_fallidos, authorized_keys):
    """
    Reglas deterministas (sin LLM) sobre los datos ya relevados:

    - media:  puerto en escucha en 0.0.0.0/:: que no está en la lista de
              puertos esperados (settings.PUERTOS_ESCUCHA_ESPERADOS).
    - alta:   conexión externa a puerto no estándar SIN proceso identificable
              (settings.PUERTOS_EXTERNOS_ESTANDAR).
    - alta:   fallos SSH >= settings.UMBRAL_FALLOS_SSH_ALERTA
              (info si hay pero está por debajo del umbral).
    - media:  ~/.ssh/authorized_keys modificado dentro de
              settings.UMBRAL_DIAS_AUTHORIZED_KEYS días.

    Veredicto: "alerta" si hay algún hallazgo alta; "revisar" si hay alguno
    medio; "sin_hallazgos" en cualquier otro caso. La síntesis del LLM solo
    puede afirmar este veredicto y los conteos — jamás concluir "intrusión"
    por su cuenta (el validador de graph_nodes lo impone).

    Devuelve (hallazgos, veredicto, conteos).
    """
    from core.config.settings import (
        PUERTOS_ESCUCHA_ESPERADOS, PUERTOS_EXTERNOS_ESTANDAR,
        UMBRAL_FALLOS_SSH_ALERTA, UMBRAL_DIAS_AUTHORIZED_KEYS,
    )

    hallazgos: list[dict] = []

    # R1: puertos expuestos no esperados → media
    esperados = {str(p) for p in PUERTOS_ESCUCHA_ESPERADOS}
    for p in puertos or []:
        if p.get("expuesto") and str(p.get("puerto")) not in esperados:
            hallazgos.append({
                "severidad": "media",
                "tipo": "puerto_expuesto_no_esperado",
                "detalle": (f"puerto {p.get('puerto')}/{p.get('proto')} escuchando "
                            f"en {p.get('local')} (accesible desde fuera)"),
            })

    # R2: conexión externa a puerto no estándar sin proceso → alta
    estandar = {str(p) for p in PUERTOS_EXTERNOS_ESTANDAR}
    for c in conexiones or []:
        if c.get("clase") != "externa" or c.get("proceso"):
            continue
        puerto = str(c.get("remoto_puerto") or "")
        if puerto and puerto not in estandar:
            hallazgos.append({
                "severidad": "alta",
                "tipo": "conexion_externa_sospechosa",
                "detalle": (f"conexión externa a {c.get('remoto')} sin proceso "
                            f"identificable en puerto no estándar"),
            })

    # R3: fallos SSH → alta sobre umbral / info por debajo
    if ssh_fallidos is not None:
        n = int(ssh_fallidos)
        umbral = int(UMBRAL_FALLOS_SSH_ALERTA)
        if n >= umbral:
            hallazgos.append({
                "severidad": "alta",
                "tipo": "ssh_fallos_sobre_umbral",
                "detalle": f"{n} intentos SSH fallidos (umbral de alerta: {umbral})",
            })
        elif n > 0:
            hallazgos.append({
                "severidad": "info",
                "tipo": "ssh_fallos_bajo_umbral",
                "detalle": (f"{n} intentos SSH fallidos (por debajo del umbral "
                            f"de alerta de {umbral})"),
            })

    # R4: authorized_keys modificado recientemente → media
    if isinstance(authorized_keys, dict) and authorized_keys.get("existe"):
        mod = authorized_keys.get("modificado")
        if mod:
            try:
                dt = datetime.fromisoformat(str(mod))
                dias = (datetime.now() - dt).total_seconds() / 86400.0
                if 0 <= dias <= float(UMBRAL_DIAS_AUTHORIZED_KEYS):
                    hallazgos.append({
                        "severidad": "media",
                        "tipo": "authorized_keys_reciente",
                        "detalle": (f"~/.ssh/authorized_keys modificado hace "
                                    f"{dias:.1f} días"),
                    })
            except ValueError:
                pass

    severidades = {h.get("severidad") for h in hallazgos}
    if "alta" in severidades:
        veredicto = "alerta"
    elif "media" in severidades:
        veredicto = "revisar"
    else:
        veredicto = "sin_hallazgos"

    conteos = {
        "conexiones_establecidas": len(conexiones or []),
        "conexiones_por_clase": {
            clase: sum(1 for c in (conexiones or []) if c.get("clase") == clase)
            for clase in ("loopback", "lan", "externa")
        },
        "puertos_en_escucha": len(puertos or []),
        "puertos_expuestos": sum(1 for p in (puertos or []) if p.get("expuesto")),
        "ssh_fallidos": ssh_fallidos,
    }
    return hallazgos, veredicto, conteos


# ═══════════════════════════════════════════════════════════════════════
# ORQUESTADOR
# ═══════════════════════════════════════════════════════════════════════

def auditar_red(runner=None) -> dict:
    """
    Auditoría de red completa, read-only. Devuelve un dict JSON-able:

        {
          "generado": "...",
          "conexiones_establecidas": [...],
          "puertos_escucha": [...],
          "ssh_fallidos": int|null,
          "last_logins": [...],
          "authorized_keys": {...}|null,
          "errores": {"<subcheck>": "<motivo>"}   (solo si algo falló)
        }

    `runner(cmd) -> (stdout, rc)` inyectable para tests.
    """
    informe: dict = {"generado": datetime.now().isoformat(timespec="seconds")}
    errores: dict[str, str] = {}

    try:
        sub = _subconexiones(runner)
        informe["conexiones_establecidas"] = sub["conexiones_establecidas"]
        if sub.get("nota_procesos"):
            informe["nota_procesos"] = sub["nota_procesos"]
    except Exception as e:
        informe["conexiones_establecidas"] = []
        errores["conexiones_establecidas"] = str(e)

    try:
        informe["puertos_escucha"] = _subescucha(runner)
    except Exception as e:
        informe["puertos_escucha"] = []
        errores["puertos_escucha"] = str(e)

    try:
        sub_ssh = _sub_ssh_fallidos(runner)
        informe["ssh_fallidos"] = sub_ssh.get("ssh_fallidos")
        if sub_ssh.get("unidad"):
            informe["ssh_unidad"] = sub_ssh["unidad"]
        if sub_ssh.get("ultimo_fallo"):
            informe["ssh_ultimo_fallo"] = sub_ssh["ultimo_fallo"]
        if sub_ssh.get("motivo"):
            informe["ssh_fallidos_motivo"] = sub_ssh["motivo"]
    except Exception as e:
        informe["ssh_fallidos"] = None
        errores["ssh_fallidos"] = str(e)

    try:
        informe["last_logins"] = _sub_last_logins(runner)
    except Exception as e:
        informe["last_logins"] = []
        errores["last_logins"] = str(e)

    try:
        informe["authorized_keys"] = _sub_authorized_keys()
    except Exception as e:
        informe["authorized_keys"] = None
        errores["authorized_keys"] = str(e)

    # 6: hallazgos + veredicto DETERMINISTAS (sin LLM). La síntesis solo
    # puede afirmar este veredicto y los conteos; el validador de
    # graph_nodes castiga cualquier conclusión de alarma no respaldada.
    try:
        hallazgos, veredicto, conteos = _evaluar_hallazgos(
            informe.get("conexiones_establecidas") or [],
            informe.get("puertos_escucha") or [],
            informe.get("ssh_fallidos"),
            informe.get("authorized_keys"),
        )
        informe["hallazgos"] = hallazgos
        informe["veredicto"] = veredicto
        informe["conteos"] = conteos
    except Exception as e:
        errores["hallazgos"] = str(e)

    if errores:
        informe["errores"] = errores

    # Reordenar: veredicto/hallazgos/conteos/errores van PRIMEROS en el JSON.
    # Así sobreviven a los truncamientos posteriores (6000 chars del registro
    # en RAM, 2000 chars del synthesizer, 40 líneas del contexto) y el
    # validador de veredicto siempre los ve, incluso en reportes enormes.
    resumen_claves = ("veredicto", "hallazgos", "conteos", "errores")
    compacto = {k: informe.pop(k) for k in resumen_claves if k in informe}
    ordenado: dict = {"generado": informe.pop("generado", "")}
    ordenado.update(compacto)
    ordenado.update(informe)
    return ordenado
