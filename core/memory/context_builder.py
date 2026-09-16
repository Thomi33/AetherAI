"""
context_builder.py — Constructor de contexto para Aether.

Construye el contexto que recibe el LLM antes de cada inferencia.
En vez de volcar todo el historial, arma slots independientes
y filtra por relevancia al tema actual.

Slots:
    [SISTEMA]         — core_memory (siempre presente)
    [RECUERDOS]       — hechos permanentes relevantes al tema
    [CONVERSACIÓN]    — solo turnos del tema actual (no todo el historial)
    [COMANDOS]        — últimos comandos ejecutados (solo si aplica)

Context Dump:
    Genera un log legible de qué entró y qué se descartó.
    Útil para depurar por qué el modelo respondió de cierta manera.
"""

from core.memory.memory_manager import (
    obtener_recuerdos,
    obtener_turnos_por_tema,
    obtener_ultimos_turnos,
)
from core.config.settings import (
    CONTEXTO_CONV_MAX_CHARS,
    MAX_TURNOS_CONTEXTO,
    MAX_TURNOS_CONTEXTO_CHAT,
    MAX_TURNOS_CONTEXTO_PLAN,
)

# Temas que NO necesitan historial de comandos en el contexto
_TEMAS_SIN_COMANDOS = {"chat", "text", "memory", "web", ""}

# Máximo de recuerdos a inyectar
MAX_RECUERDOS     = 10


# ══════════════════════════════════════════════════════════════════════
# BUILDER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

_TEMAS_CHAT = ("chat", "text", "")


def _limite_turnos(tema: str, es_multitool: bool) -> int:
    """
    Ventana adaptativa de contexto (ver la sección de rendimiento del README).

    - Charla (chat/text/sin tema): ventana chica, evita que el modelo
      "siga" una tarea vieja de sesiones/turnos pasados.
    - Paso de un plan multi-tool: sin historial conversacional (cada paso
      se ejecuta con su propia instrucción + resultados de pasos previos,
      que es lo único relevante — el historial largo CONTAMINA cada paso).
    - Tool única no-chat (ej. un solo "busca X"): presupuesto amplio,
      acotado igual por CONTEXTO_CONV_MAX_CHARS más abajo.
    """
    if tema in _TEMAS_CHAT:
        return MAX_TURNOS_CONTEXTO_CHAT
    if es_multitool:
        return MAX_TURNOS_CONTEXTO_PLAN
    return MAX_TURNOS_CONTEXTO


def construir_contexto_memoria(
    mem: dict, tema: str = "", sesion_id: str = "", es_multitool: bool = False
) -> str:
    """
    Construye el contexto para el LLM filtrando por relevancia.

    Args:
        mem:  dict RAM de Aether (cargado por memory_manager)
        tema: intención detectada por el planner ("web", "shell", "chat", etc.)
              Si está vacío, solo inyecta core_memory y recuerdos importantes.
        es_multitool: True si el plan activo tiene más de 1 paso. Fuerza
              MAX_TURNOS_CONTEXTO_PLAN (ventana adaptativa, ver arriba).

    Returns:
        String listo para inyectar en el system prompt.
    """
    slots  = {}
    descartado = []
    limite_turnos = _limite_turnos(tema, es_multitool)

    # ── Slot 1: SISTEMA (core_memory — siempre presente) ─────────────
    core = mem.get("core") or {}
    if core:
        lineas = [f"  {k}: {v}" for k, v in core.items() if v]
        slots["SISTEMA"] = "\n".join(lineas)
    else:
        slots["SISTEMA"] = "  (sin datos de core)"

    # Compatibilidad útil: el perfil explícito sigue siendo más legible para
    # el modelo que inferir estos datos desde una conversación o resumen.
    prefs = mem.get("preferencias") or {}
    if prefs:
        perfil = []
        if prefs.get("nombre_usuario"):
            perfil.append(f"  nombre: {prefs['nombre_usuario']}")
        if prefs.get("navegador"):
            perfil.append(f"  navegador: {prefs['navegador']}")
        if prefs.get("notas"):
            perfil.extend(f"  nota: {nota}" for nota in prefs["notas"][-10:] if nota)
        if perfil:
            slots["PERFIL DEL CREADOR"] = "\n".join(perfil)

    # ── Slot 1.5: RESUMEN acumulativo (rolling summary, siempre presente) ──
    # Fuente PRINCIPAL de memoria de largo plazo. A diferencia del historial
    # crudo de abajo (acotado a la sesión/tema actual por diseño anti-
    # contaminación), esto es la síntesis curada de todo lo consolidado
    # hasta ahora, sin volcar transcripciones completas al prompt.
    # Ver core/memory/consolidator.py.
    resumen = (mem.get("resumen") or "").strip()
    if resumen:
        slots["RESUMEN DEL USUARIO"] = "\n".join(f"  {linea}" for linea in resumen.splitlines())
    else:
        descartado.append("resumen (todavía sin consolidar)")

    # ── Slot 2: RECUERDOS relevantes al tema ─────────────────────────
    # Si hay tema, buscar recuerdos de esa categoría primero
    recuerdos = []
    if tema and tema not in ("text", "chat", ""):
        recuerdos = obtener_recuerdos(categoria=tema, importancia_min=1, limit=MAX_RECUERDOS)

    # Completar con recuerdos de alta importancia si hay espacio
    if len(recuerdos) < MAX_RECUERDOS:
        generales = obtener_recuerdos(importancia_min=2, limit=MAX_RECUERDOS - len(recuerdos))
        # Evitar duplicados
        ids_ya = {r.get("contenido") for r in recuerdos}
        recuerdos += [r for r in generales if r.get("contenido") not in ids_ya]

    if recuerdos:
        lineas = [
            f"  [{r.get('categoria', 'general')}] {r.get('contenido', '')}"
            for r in recuerdos
        ]
        slots["RECUERDOS"] = "\n".join(lineas)
    else:
        descartado.append("recuerdos (ninguno relevante)")

    # ── Slot 2b: APRENDIZAJES de la memoria central compartida ─────────
    # Recuerdos aprendidos por CUALQUIER runtime (terminal, web, roblox).
    # Fail-open: si la memoria central no existe o falla, no se altera el
    # contexto. Se desactiva por completo con AETHER_CENTRAL_MEMORY=0.
    aprendizajes = _aprendizajes_centrales(tema)
    if aprendizajes:
        slots["APRENDIZAJES"] = aprendizajes

    # ── Slot 3: CONVERSACIÓN filtrada por tema ────────────────────────
    turnos_relevantes = []

    if tema and limite_turnos > 0:
        turnos_relevantes = obtener_turnos_por_tema(
            tema, sesion_id=sesion_id, limit=limite_turnos
        )

    if not turnos_relevantes and limite_turnos > 0:
        # Usar sesion_id explícito, o fallback al último de la RAM
        sesion_actual = sesion_id or _sesion_actual(mem)
        turnos_ram = mem.get("conversacion") or []
        if sesion_actual:
            turnos_relevantes = [
                t for t in turnos_ram if t.get("sesion_id") == sesion_actual
            ][-limite_turnos:]
        else:
            turnos_relevantes = turnos_ram[-limite_turnos:]
            if len(turnos_ram) > limite_turnos:
                descartado.append(
                    f"conversación ({len(turnos_ram) - limite_turnos} turnos antiguos omitidos)"
                )
    elif limite_turnos == 0:
        descartado.append("conversación (paso de plan multi-tool: sin historial, anti-contaminación)")

    if turnos_relevantes:
        seleccion = []
        total_chars = 0
        for t in reversed(turnos_relevantes):
            rol   = str(t.get("rol", "?")).upper()
            texto = str(t.get("texto", ""))
            linea = f"  [{rol}]: {texto}"
            if total_chars + len(linea) > CONTEXTO_CONV_MAX_CHARS:
                descartado.append(f"conversación (truncada por presupuesto de caracteres)")
                break
            seleccion.append(linea)
            total_chars += len(linea)
        seleccion.reverse()
        slots[f"CONVERSACIÓN — {len(seleccion)} turnos"] = "\n".join(seleccion)
    else:
        descartado.append("conversación (sin historial relevante)")

    # ── Slot 4: COMANDOS (solo para temas que lo necesitan) ──────────
    if tema not in _TEMAS_SIN_COMANDOS:
        historial = mem.get("historial_comandos") or []
        if historial:
            ultimos = historial[-5:]
            lineas  = [f"  {e.get('fecha', '')} → {e.get('cmd', '')}" for e in ultimos]
            slots["COMANDOS RECIENTES"] = "\n".join(lineas)
        else:
            descartado.append("comandos (sin historial)")
    else:
        descartado.append(f"comandos (innecesarios para tema '{tema}')")

    # ── Ensamblar contexto final ──────────────────────────────────────
    partes = []
    for nombre, contenido in slots.items():
        partes.append(f"[{nombre}]\n{contenido}")

    # Solo alertar por pérdida real de datos (truncamiento por presupuesto
    # de caracteres u omisión de turnos antiguos por límite de ventana).
    # El resto de "descartado" son estados normales (sin recuerdos, sin
    # comandos, tema de chat sin historial de comandos, etc.) que pasan en
    # casi todos los turnos y no ameritan warning — avisar por esos ahogaría
    # la señal real en ruido.
    perdida_real = [d for d in descartado if "truncada" in d or "omitidos" in d]
    if perdida_real:
        print(f"⚠️  [CONTEXT]: posible pérdida de contexto — {'; '.join(perdida_real)}")

    return "\n\n".join(partes)


def construir_context_dump(mem: dict, tema: str = "", sesion_id: str = "", es_multitool: bool = False) -> str:
    """
    Genera un log legible de qué información entra al modelo y qué se descarta.
    Llamar antes de cada inferencia para facilitar el debug.

    Ejemplo de salida:
        === CONTEXT DUMP ===
        Tema:      shell
        Slots:     SISTEMA, RECUERDOS, COMANDOS RECIENTES
        Descartado: conversación (sin historial relevante)
        Turnos inyectados: 3
        ====================
    """
    core      = mem.get("core") or {}
    turnos    = mem.get("conversacion") or []
    comandos  = mem.get("historial_comandos") or []

    recuerdos_count = len(
        obtener_recuerdos(categoria=tema if tema else None, importancia_min=1, limit=MAX_RECUERDOS)
    )
    limite_turnos = _limite_turnos(tema, es_multitool)
    turnos_tema = (
        obtener_turnos_por_tema(tema, sesion_id=sesion_id, limit=limite_turnos)
        if tema and limite_turnos > 0 else []
    )
    
    slots_activos = ["SISTEMA"]
    if recuerdos_count:
        slots_activos.append("RECUERDOS")
    if turnos_tema or turnos:
        slots_activos.append("CONVERSACIÓN")
    if tema not in _TEMAS_SIN_COMANDOS and comandos:
        slots_activos.append("COMANDOS RECIENTES")

    lineas = [
        "=== DEBUG: CONTEXT DUMP ===",
        f"Tema:              {tema or '(sin tema)'}",
        f"Core memory:       {len(core)} claves",
        f"Slots activos:     {', '.join(slots_activos)}",
        f"Recuerdos:         {recuerdos_count} inyectados",
        f"Turnos (tema):     {len(turnos_tema)} inyectados (límite={limite_turnos}) de {len(turnos)} en RAM",
        f"Comandos en RAM:   {len(comandos)}",
        "====================",
    ]
    return "\n".join(lineas)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _aprendizajes_centrales(tema: str, *, limit: int = 5, max_chars: int = 600) -> str:
    """Recupera aprendizajes relevantes de la memoria central compartida.

    La memoria central (~/.aether/memory/) es compartida por todos los
    runtimes de Aether: lo que aprende el Roblox Player puede aparecer aquí
    y viceversa. Fail-open por diseño: cualquier error devuelve "".
    """
    import os
    if os.environ.get("AETHER_CENTRAL_MEMORY", "1") == "0":
        return ""
    try:
        from core.memory.central import get_central_memory
        central = get_central_memory()
        resultados = central.search(tema or "", limit=limit)
        if not resultados:
            return ""
        lineas = [f"  - {r['summary']}" for r in resultados]
        return "\n".join(lineas)[:max_chars]
    except Exception:
        return ""


def _sesion_actual(mem: dict) -> str:
    """Retorna el sesion_id del turno más reciente en RAM."""
    turnos = mem.get("conversacion") or []
    if turnos:
        return turnos[-1].get("sesion_id", "")
    return ""
