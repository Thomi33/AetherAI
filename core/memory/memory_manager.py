"""
memory_manager.py — Gestor de memoria de Aether.

Persistencia: un ÚNICO archivo JSON (ver core/memory/memoria_store.py):

    <BASE_AETHER>/db/memoria.json   (+ memoria.json.bak rotativo)

NO hay SQLite en ningún lado: la escritura es atómica, con backup y
recuperación automática ante corrupción. La primera vez que arranca sin
memoria.json, el store importa UNA SOLA VEZ el contenido de la DB legacy
(current.db) si existiera (one-shot dentro de memoria_store); después el
formato viejo no se vuelve a tocar.

Estructura gestionada:
  conversaciones/comandos — historial operativo (acotado)
  recuerdos        — hechos permanentes (archival memory)
  core             — estado siempre presente
  resumen          — rolling summary (curado por consolidación)
"""
from __future__ import annotations

from datetime import datetime

from core.memory import memoria_store

# ── Límites ──────────────────────────────────────────────────────────
MAX_HISTORIAL_RAM = 200   # turnos que se mantienen en el dict RAM
MAX_TEXTO         = 1000  # caracteres máximos por turno/comando/recuerdo


# ══════════════════════════════════════════════════════════════════════
# COMPAT DE ARRANQUE (antes era el inicializador del esquema sqlite)
# ══════════════════════════════════════════════════════════════════════
def asegurar_esquema() -> None:
    """Con el store JSON no hay esquema que preparar: solo garantiza que el
    archivo exista (y dispara la migración legacy si corresponde)."""
    try:
        memoria_store.guardar(memoria_store.cargar())
        print("[MEMORIA] Store JSON verificado en", memoria_store.RUTA_JSON)
    except Exception as e:
        print(f"[MEMORIA] ⚠️  No se pudo inicializar el store JSON: {e}")


def inicializar_db() -> None:
    """Compat con arranques viejos (era el inicializador sqlite)."""
    asegurar_esquema()


# ══════════════════════════════════════════════════════════════════════
# IDENTIDAD DEL USUARIO — sin hardcodear nombres en código
# ══════════════════════════════════════════════════════════════════════
def nombre_usuario_default() -> str:
    """Nombre del usuario desde el perfil de Account (config ACCOUNT_PROFILE).

    La identidad NUNCA se hardcodea en código: si no hay perfil de cuenta
    configurado, devuelve "" y el prompt/contexto no menciona ningún nombre.
    """
    try:
        from core.config.config_manager import get_config_manager
        perfil = get_config_manager().get("ACCOUNT_PROFILE", {}) or {}
        return str(perfil.get("nombre") or "").strip()
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════
# DICT RAM — estructura en memoria durante la sesión
# ══════════════════════════════════════════════════════════════════════
def _memoria_vacia() -> dict:
    return {
        "preferencias":      {"nombre_usuario": nombre_usuario_default(),
                              "navegador": "brave", "notas": []},
        "flatpaks":          {},
        "conversacion":       [],   # [{rol, texto, fecha, sesion_id, tema}]
        "historial_comandos": [],   # [{orden, cmd, fecha, sesion_id}]
        "core":               {},   # espejo de core
        "resumen":            "",   # espejo de resumen.texto
        "sesion_id":          "",
    }


def normalizar_mem(mem: dict | None) -> dict:
    """Garantiza que el dict RAM tenga la estructura correcta."""
    base = _memoria_vacia()
    if not isinstance(mem, dict):
        return base
    for clave, valor in base.items():
        mem.setdefault(clave, valor.copy() if isinstance(valor, dict)
                       else list(valor) if isinstance(valor, list) else valor)
    if not isinstance(mem.get("preferencias"), dict):
        mem["preferencias"] = dict(base["preferencias"])
    else:
        prefs = mem["preferencias"]
        # Una clave vacía/legacy ("" o ausente) se completa desde la cuenta.
        if not str(prefs.get("nombre_usuario") or "").strip():
            prefs["nombre_usuario"] = nombre_usuario_default()
        prefs.setdefault("navegador", "brave")
        if not isinstance(prefs.get("notas"), list):
            prefs["notas"] = []
    if not isinstance(mem.get("flatpaks"), dict):
        mem["flatpaks"] = {}
    if not isinstance(mem.get("conversacion"), list):
        mem["conversacion"] = []
    if not isinstance(mem.get("historial_comandos"), list):
        mem["historial_comandos"] = []
    if not isinstance(mem.get("core"), dict):
        mem["core"] = {}
    if "sesion_id" not in mem:
        mem["sesion_id"] = ""
    if not isinstance(mem.get("resumen"), str):
        mem["resumen"] = ""
    return mem


# ══════════════════════════════════════════════════════════════════════
# CORE MEMORY — siempre presente en el contexto
# ══════════════════════════════════════════════════════════════════════
def leer_core_memory() -> dict:
    """Lee toda la core como dict {clave: valor}."""
    try:
        return dict(memoria_store.cargar()["core"])
    except Exception as e:
        print(f"[MEMORIA] Error leyendo core: {e}")
        return {}


def actualizar_core(clave: str, valor: str) -> None:
    """Actualiza o inserta un valor en core."""
    try:
        data = memoria_store.cargar()
        data["core"][str(clave)] = str(valor)
        memoria_store.guardar(data)
    except Exception as e:
        print(f"[MEMORIA] Error actualizando core '{clave}': {e}")


# ══════════════════════════════════════════════════════════════════════
# RESUMEN ACUMULATIVO (rolling summary) — lo cura el consolidador.
# ══════════════════════════════════════════════════════════════════════
def obtener_resumen() -> dict:
    """{"texto": str, "ultimo_turno_id": int, "actualizado": str}."""
    try:
        return dict(memoria_store.cargar()["resumen"])
    except Exception as e:
        print(f"[MEMORIA] Error leyendo resumen: {e}")
        return {"texto": "", "ultimo_turno_id": 0, "actualizado": ""}


def guardar_resumen(texto: str, ultimo_turno_id: int | None = None) -> None:
    """Reemplaza el resumen acumulativo completo.

    Si ultimo_turno_id es None, preserva el cursor de consolidación vigente
    (ediciones manuales desde la Web UI/TUI no re-procesan turnos viejos).
    """
    try:
        data = memoria_store.cargar()
        if ultimo_turno_id is None:
            ultimo_turno_id = data["resumen"].get("ultimo_turno_id", 0)
        data["resumen"] = {
            "texto": str(texto),
            "ultimo_turno_id": int(ultimo_turno_id),
            "actualizado": datetime.now().isoformat(),
        }
        memoria_store.guardar(data)
        print(f"[MEMORIA] Resumen guardado ({len(texto)} chars, "
              f"hasta turno #{ultimo_turno_id}).")
    except Exception as e:
        print(f"[MEMORIA] Error guardando resumen: {e}")


def obtener_turnos_pendientes_de_resumen(limit: int = 300) -> list[dict]:
    """Turnos posteriores al último ya consolidado en el resumen."""
    ultimo_id = int(obtener_resumen().get("ultimo_turno_id") or 0)
    try:
        turnos = memoria_store.cargar()["conversaciones"]
        pendientes = [t for t in turnos if int(t.get("id") or 0) > ultimo_id]
        return pendientes[:max(1, int(limit))]
    except Exception as e:
        print(f"[MEMORIA] Error leyendo turnos pendientes: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════
# CARGA INICIAL (DB → dict RAM)
# ══════════════════════════════════════════════════════════════════════
def cargar_memoria() -> dict:
    """Carga el dict RAM desde el store JSON. Nunca rompe el arranque."""
    mem = _memoria_vacia()
    try:
        data = memoria_store.cargar()
        mem["core"] = dict(data["core"])
        mem["resumen"] = data["resumen"].get("texto", "")
        mem["conversacion"] = [
            {"rol": t.get("rol", ""), "texto": t.get("texto", ""),
             "fecha": t.get("fecha", ""), "sesion_id": t.get("sesion_id", ""),
             "tema": t.get("tema", "")}
            for t in data["conversaciones"][-MAX_HISTORIAL_RAM:]
        ]
        mem["historial_comandos"] = [
            {"orden": c.get("orden", ""), "cmd": c.get("cmd", ""),
             "fecha": c.get("fecha", ""), "sesion_id": c.get("sesion_id", "")}
            for c in data["comandos"][-50:]
        ]
    except Exception as e:
        print(f"[MEMORIA] Error cargando memoria: {e}")
    return normalizar_mem(mem)


# ══════════════════════════════════════════════════════════════════════
# CONVERSACIÓN
# ══════════════════════════════════════════════════════════════════════
def registrar_turno(
    mem: dict,
    rol: str,
    texto: str,
    sesion_id: str = "",
    tema: str = "",
    proyecto: str = "",
) -> None:
    """Registra un turno en el store JSON y en el dict RAM."""
    texto = str(texto)[:MAX_TEXTO]
    fecha = datetime.now().isoformat()
    try:
        data = memoria_store.cargar()
        data["conversaciones"].append({
            "id": memoria_store.proximo_id(data, "conversaciones"),
            "fecha": fecha, "rol": str(rol), "texto": texto,
            "sesion_id": str(sesion_id or ""), "tema": str(tema or ""),
            "proyecto": str(proyecto or ""),
        })
        data["conversaciones"] = data["conversaciones"][-memoria_store.MAX_TURNOS_JSON:]
        memoria_store.guardar(data)
        print(f"[MEMORIA] Turno guardado | rol={rol} | tema={tema} | texto={texto[:40]}...")
    except Exception as e:
        print(f"[MEMORIA] Error registrando turno: {e}")

    if not isinstance(mem.get("conversacion"), list):
        mem["conversacion"] = []
    mem["conversacion"].append({
        "rol": rol, "texto": texto,
        "fecha": fecha, "sesion_id": sesion_id, "tema": tema,
    })
    mem["conversacion"] = mem["conversacion"][-MAX_HISTORIAL_RAM:]


# ══════════════════════════════════════════════════════════════════════
# COMANDOS
# ══════════════════════════════════════════════════════════════════════
def registrar_comando(
    mem: dict,
    orden: str,
    cmd: str,
    sesion_id: str = "",
    exitoso: bool = True,
) -> None:
    """Registra un comando shell en el store JSON y en el dict RAM."""
    cmd   = str(cmd)[:MAX_TEXTO]
    orden = str(orden)[:MAX_TEXTO]
    fecha = datetime.now().isoformat()
    try:
        data = memoria_store.cargar()
        data["comandos"].append({
            "id": memoria_store.proximo_id(data, "comandos"),
            "fecha": fecha, "orden": orden, "cmd": cmd,
            "sesion_id": str(sesion_id or ""), "exitoso": bool(exitoso),
        })
        data["comandos"] = data["comandos"][-memoria_store.MAX_COMANDOS_JSON:]
        memoria_store.guardar(data)
    except Exception as e:
        print(f"[MEMORIA] Error registrando comando: {e}")

    mem["historial_comandos"].append({
        "orden": orden, "cmd": cmd, "fecha": fecha, "sesion_id": sesion_id,
    })
    mem["historial_comandos"] = mem["historial_comandos"][-50:]


# ══════════════════════════════════════════════════════════════════════
# RECUERDOS (archival memory)
# ══════════════════════════════════════════════════════════════════════
def guardar_recuerdo(
    contenido: str,
    categoria: str = "",
    importancia: int = 1,
) -> None:
    """Guarda un hecho permanente en recuerdos (store JSON)."""
    contenido = str(contenido or "").strip()[:MAX_TEXTO]
    if not contenido:
        print("[MEMORIA] Recuerdo vacío: no se guarda.")
        return
    try:
        data = memoria_store.cargar()
        data["recuerdos"].append({
            "id": memoria_store.proximo_id(data, "recuerdos"),
            "fecha": datetime.now().isoformat(),
            "categoria": str(categoria or "").strip(),
            "contenido": contenido,
            "importancia": max(1, min(int(importancia), 10)),
        })
        memoria_store.guardar(data)
    except Exception as e:
        print(f"[MEMORIA] Error guardando recuerdo: {e}")


def obtener_recuerdos(
    categoria: str | None = None,
    importancia_min: int = 1,
    limit: int = 20,
) -> list[dict]:
    """Recuerdos filtrados por categoría e importancia mínima (orden: más
    importantes primero, y a igualdad los más recientes)."""
    try:
        recs = [r for r in memoria_store.cargar()["recuerdos"]
                if int(r.get("importancia") or 1) >= int(importancia_min)
                and (not categoria or r.get("categoria", "") == categoria)]
        recs.sort(key=lambda r: (-int(r.get("importancia") or 1),
                                 -int(r.get("id") or 0)))
        return recs[:max(1, int(limit))]
    except Exception as e:
        print(f"[MEMORIA] Error leyendo recuerdos: {e}")
        return []


def borrar_recuerdo(recuerdo_id: int) -> bool:
    """Borra un recuerdo por id. True si existía (botón de la Web UI)."""
    try:
        data = memoria_store.cargar()
        rid = int(recuerdo_id)
        antes = len(data["recuerdos"])
        data["recuerdos"] = [r for r in data["recuerdos"]
                             if int(r.get("id") or -1) != rid]
        if len(data["recuerdos"]) == antes:
            return False
        memoria_store.guardar(data)
        return True
    except Exception as e:
        print(f"[MEMORIA] Error borrando recuerdo {recuerdo_id}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════
# RECUPERACIÓN POR RELEVANCIA / SESIONES (Context Manager, Web UI, TUI)
# ══════════════════════════════════════════════════════════════════════
def obtener_turnos_por_tema(tema: str, sesion_id: str = "", limit: int = 10) -> list[dict]:
    """Turnos anteriores del mismo tema. Si se pasa sesion_id, filtra SOLO
    turnos de esa sesión (evita contaminar el contexto con sesiones viejas)."""
    try:
        turnos = memoria_store.cargar()["conversaciones"]
        filas = [t for t in turnos if t.get("tema") == tema]
        if sesion_id:
            filas = [t for t in filas if t.get("sesion_id") == sesion_id]
        filas = filas[-max(1, int(limit)):]
        return [
            {"fecha": t.get("fecha", ""), "rol": t.get("rol", ""),
             "texto": t.get("texto", ""), "sesion_id": t.get("sesion_id", "")}
            for t in filas
        ]
    except Exception as e:
        print(f"[MEMORIA] Error buscando turnos por tema: {e}")
        return []


def obtener_turnos_por_sesion(sesion_id: str) -> list[dict]:
    """Todos los turnos de una sesión, en orden cronológico."""
    try:
        filas = [t for t in memoria_store.cargar()["conversaciones"]
                 if t.get("sesion_id") == sesion_id]
        return [
            {"fecha": t.get("fecha", ""), "rol": t.get("rol", ""),
             "texto": t.get("texto", ""), "tema": t.get("tema", "")}
            for t in filas
        ]
    except Exception as e:
        print(f"[MEMORIA] Error buscando sesión: {e}")
        return []


def obtener_ultimos_turnos(n: int = 5) -> list[dict]:
    """Últimos N turnos sin filtro (debug/fallback)."""
    try:
        filas = memoria_store.cargar()["conversaciones"][-max(1, int(n)):]
        return [
            {"fecha": t.get("fecha", ""), "rol": t.get("rol", ""),
             "texto": t.get("texto", ""), "tema": t.get("tema", ""),
             "sesion_id": t.get("sesion_id", "")}
            for t in filas
        ]
    except Exception as e:
        print(f"[MEMORIA] Error leyendo últimos turnos: {e}")
        return []


def listar_sesiones(limit: int = 15) -> list[dict]:
    """Resumen de las últimas `limit` sesiones distintas: sesion_id, fecha de
    inicio, cantidad de turnos y preview del primer mensaje del usuario.
    (Equivale a /sesiones de la TUI y al sidebar de la Web UI.)"""
    try:
        agrupadas: dict[str, dict] = {}
        for t in memoria_store.cargar()["conversaciones"]:
            sid = t.get("sesion_id") or ""
            if not sid:
                continue
            g = agrupadas.setdefault(sid, {"inicio": t.get("fecha", ""),
                                           "turnos": 0, "preview": ""})
            g["turnos"] += 1
            if not t.get("fecha") or t["fecha"] < g["inicio"]:
                g["inicio"] = t["fecha"]
            if not g["preview"] and t.get("rol") == "usuario" and t.get("texto"):
                g["preview"] = str(t["texto"])[:60]
        sesiones = [
            {"sesion_id": sid, "inicio": g["inicio"],
             "turnos": g["turnos"], "preview": g["preview"]}
            for sid, g in agrupadas.items()
        ]
        sesiones.sort(key=lambda s: s["inicio"], reverse=True)
        return sesiones[:max(1, min(int(limit), 50))]
    except Exception as e:
        print(f"[MEMORIA] Error listando sesiones: {e}")
        return []



# ══════════════════════════════════════════════════════════════════════════════
# CONSOLIDACIÓN DE RESUMEN (antes en consolidator.py — ahora síncrono y simple)
# ══════════════════════════════════════════════════════════════════════════════

# Umbral por defecto: no vale la pena gastar una llamada al LLM por cada
# turno nuevo. Se consolida cuando hay un lote razonable acumulado.
MIN_TURNOS_PARA_CONSOLIDAR = 12
COMPACTACION_MIN_CHARS = 12000

_SYSTEM_PROMPT_CONSOLIDATOR = (
    "Sos el consolidador de memoria de Aether. Tu única salida es un resumen "
    "acumulativo actualizado — NO un log de conversación, NO una lista de "
    "turnos, NO explicaciones sobre lo que hiciste.\n\n"
    "Reglas:\n"
    "- Reescribís el resumen COMPLETO cada vez, integrando lo nuevo relevante "
    "y descartando charla efímera, saludos, o información que ya quedó "
    "obsoleta (ej: un problema que ya se resolvió, un plan que cambió).\n"
    "- Priorizá: hechos estables del usuario, proyectos activos y su estado, "
    "preferencias técnicas, decisiones tomadas, problemas pendientes.\n"
    "- Nunca inventes datos que no estén en el resumen previo o los turnos "
    "nuevos.\n"
    "- Prosa organizada en secciones cortas si hace falta, sin relleno, en "
    "español.\n"
    "- Si un turno nuevo contradice al resumen previo (ej: un dato cambió), "
    "el turno nuevo gana — no dejes versiones viejas y nuevas conviviendo."
)


def _bloque_turnos(turnos: list[dict]) -> str:
    lineas = []
    for t in turnos:
        rol = str(t.get("rol", "?")).upper()
        texto = str(t.get("texto", "")).strip()
        if texto:
            lineas.append(f"[{rol}] {texto}")
    return "\n".join(lineas)


def consolidar_resumen(forzar: bool = False, min_turnos: int = MIN_TURNOS_PARA_CONSOLIDAR) -> str | None:
    """
    Actualiza el resumen acumulativo con los turnos nuevos desde la última
    consolidación.
    
    Retorna el nuevo resumen si se consolidó, None si no hizo falta.
    """
    from core.config.settings import RUTA_NOTAS

    pendientes = obtener_turnos_pendientes_de_resumen()
    if not forzar and (len(pendientes) < min_turnos):
        return None

    resumen_previo = obtener_resumen()
    prompt = f"{_SYSTEM_PROMPT_CONSOLIDATOR}\n\n[RESUMEN PREVIO]:\n{resumen_previo or '(vacío)'} \n\n[TURNOS NUEVOS]:\n{_bloque_turnos(pendientes)}"

    from core.config.settings import OLLAMA_HOST, MODELO, NUM_CTX
    import requests

    payload = {
        "model": MODELO,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_ctx": NUM_CTX},
        "keep_alive": "10m",
    }

    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    except Exception as e:
        print(f"[MEMORIA] Error llamando al modelo para consolidar: {e}")
        return None

    if r.status_code != 200:
        print(f"[MEMORIA] Error HTTP {r.status_code} consolidando: {r.text[:200]}")
        return None

    data = r.json()
    if "error" in data:
        print(f"[MEMORIA] Error del modelo consolidando: {data['error']}")
        return None

    nuevo_resumen = data.get("response", "").strip()
    if not nuevo_resumen:
        print("[MEMORIA] Resumen vacío devuelto por el modelo")
        return None

    guardar_resumen(nuevo_resumen)
    print(f"[MEMORIA] Resumen consolidado ({len(pendientes)} turnos integrados)")
    return nuevo_resumen


def compactar_contexto_si_necesario() -> bool:
    """Compacta turnos pendientes cuando el contexto crudo ya es demasiado grande."""
    pendientes = obtener_turnos_pendientes_de_resumen()
    chars = sum(len(str(t.get("texto", ""))) for t in pendientes)
    if len(pendientes) < MIN_TURNOS_PARA_CONSOLIDAR and chars < COMPACTACION_MIN_CHARS:
        return False
    return consolidar_resumen(forzar=True, min_turnos=1) is not None


def programar_consolidacion(mem: dict | None = None) -> bool:
    """
    Ejecuta consolidación síncrona (sin worker thread).
    Si hay turnos pendientes, consolida y actualiza mem['resumen'] si se pasó.
    """
    nuevo_resumen = consolidar_resumen()
    if nuevo_resumen is not None and isinstance(mem, dict):
        mem["resumen"] = nuevo_resumen
    return nuevo_resumen is not None


def consolidar_memoria_central() -> dict | None:
    """Decaimiento + barrido TTL de la memoria central compartida.
    
    Fail-open: cualquier error se traga — no bloquea el worker.
    """
    import os
    if os.environ.get("AETHER_CENTRAL_MEMORY", "1") == "0":
        return None
    try:
        from core.memory.central_store_v2 import get_central_memory
        return get_central_memory().consolidate()
    except Exception as e:
        print(f"[MEMORIA] Consolidación central falló (no bloqueante): {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# ALIAS / COMPAT — para nodos del grafo que todavía esperan estas funcs
# ══════════════════════════════════════════════════════════════════════
def guardar_memoria(mem: dict) -> None:
    """
    Stub de compatibilidad.
    Algunos nodos viejos del grafo siguen importando `guardar_memoria`,
    pero ya no hace falta: cada escritura (registrar_turno / registrar_comando
    / actualizar_core / guardar_recuerdo) persiste al instante en el JSON.
    """
    return None
