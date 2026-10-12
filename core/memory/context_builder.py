"""
context_builder.py — Constructor de contexto para Aether.

Construye el contexto que recibe el LLM antes de cada inferencia.
En vez de volcar todo el historial, arma slots independientes
y filtra por relevancia al tema actual.

Slots:
    [SISTEMA]         — core_memory (siempre presente)
    [RECUERDOS]       — hechos permanentes relevantes al tema
    [INTENTOS RECIENTES] — acciones que ya se probaron y NO dieron resultado
    [CONVERSACIÓN]    — turnos del tema actual; sin tema, los últimos N por recencia
    [COMANDOS]        — últimos comandos ejecutados (solo si aplica)

Context Dump:
    Genera un log legible de qué entró y qué se descartó.
    Útil para depurar por qué el modelo respondió de cierta manera.
"""

from core.memory.memory_manager import (
    obtener_recuerdos,
    obtener_turnos_por_tema,
    obtener_ultimos_turnos,
    obtener_resultados_tools,
)
from core.config.settings import (
    CONTEXTO_CONV_MAX_CHARS,
    MAX_TURNOS_CONTEXTO,
    MAX_TURNOS_CONTEXTO_CHAT,
    MAX_TURNOS_CONTEXTO_PLAN,
    MAX_RESULTADOS_TOOLS_CONTEXTO,
    MAX_LINEAS_RESULTADO_TOOL,
)
from core.agent.grounding import (
    truncar_lineas,
    INSTRUCCION_DATOS_TOOL,
    delimitar_bloque_datos,
)

# Temas que NO necesitan historial de comandos en el contexto
_TEMAS_SIN_COMANDOS = {"chat", "text", "memory", "web", ""}

# Máximo de recuerdos a inyectar
MAX_RECUERDOS     = 10


_PRONOMBRES_POR_GENERO = {
    "hombre": "él/lo/suyo (masculino)",
    "mujer": "ella/la/suya (femenino)",
    "no_binario": "elle/le/sue (neutro)",
}


def _lineas_perfil_cuenta() -> list[str]:
    """Datos del perfil de Account (config ACCOUNT_PROFILE) para el contexto.

    - género → pronombres: el agente se dirige al usuario con esos
      pronombres (masculino/femenino/neutro/pronombre a elección).
    - fecha de nacimiento: solo es relevante si HOY es su cumpleaños
      (en ese caso se le pide al agente que lo salude).
    - El avatar NUNCA entra al contexto: es decoración de la UI.
    """
    try:
        from core.config.config_manager import get_config_manager
        perfil = get_config_manager().get("ACCOUNT_PROFILE", {}) or {}
    except Exception:
        return []

    lineas: list[str] = []
    genero = str(perfil.get("genero") or "").strip().lower()
    if genero:
        pronombres = _PRONOMBRES_POR_GENERO.get(
            genero, str(perfil.get("pronombre_custom") or "").strip())
        if pronombres:
            lineas.append(
                f"  pronombres: {pronombres} — dirigite a la persona "
                "usando esos pronombres, siempre")
    fecha_nac = str(perfil.get("fecha_nacimiento") or "").strip()
    if fecha_nac:
        try:
            from datetime import date
            nacimiento = date.fromisoformat(fecha_nac)
            hoy = date.today()
            if (nacimiento.month, nacimiento.day) == (hoy.month, hoy.day):
                lineas.append(
                    "  ¡HOY ES SU CUMPLEAÑOS! Saludalo/a/e por su cumpleaños "
                    "de forma cálida y natural en tu primera respuesta.")
            else:
                lineas.append(
                    f"  fecha de nacimiento: {fecha_nac} (no la menciones; "
                    "solo importa el día de su cumpleaños)")
        except ValueError:
            pass
    return lineas


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


def _turnos_para_contexto(tema: str, sesion_id: str, mem: dict, limite: int) -> list:
    """
    Turnos que van a entrar al contexto (única fuente de verdad, compartida
    entre construir_contexto_memoria y construir_context_dump).

    - Con tema: solo turnos de ese tema (y sesión, si se sabe) — igual que
      siempre.
    - SIN tema (vacío/None): últimos N por recencia, preferentemente de la
      SESIÓN actual (aislamiento: no se mezclan sesiones). Fallback final:
      últimos N del store global.

    Fix del bug "Turnos (tema): 0 inyectados": la recuperación por recencia
    ya existía, pero el dump no la mostraba y el fallback global podía traer
    turnos de otras sesiones.
    """
    if limite <= 0:
        return []
    if tema:
        return obtener_turnos_por_tema(tema, sesion_id=sesion_id, limit=limite)
    sesion_actual = sesion_id or _sesion_actual(mem)
    if sesion_actual:
        # Aislamiento por sesión: quedan los turnos de la sesión actual y
        # los legacy guardados sin sesion_id (los bridges viejos no la
        # seteaban; no se les puede atribuir otra sesión). Se EXCLUYEN los
        # turnos de sesiones distintas.
        turnos_sesion = [
            t for t in (mem.get("conversacion") or [])
            if isinstance(t, dict)
            and (not t.get("sesion_id") or t.get("sesion_id") == sesion_actual)
        ]
        if turnos_sesion:
            return turnos_sesion[-limite:]
    return obtener_ultimos_turnos(limite)


def construir_contexto_memoria(
    mem: dict, tema: str = "", sesion_id: str = "", es_multitool: bool = False,
    orden: str = "",
) -> str:
    """
    Construye el contexto para el LLM filtrando por relevancia.

    Args:
        mem:  dict RAM de Aether (cargado por memory_manager)
        tema: intención detectada por el planner ("web", "shell", "chat", etc.)
              Si está vacío, el historial entra por recencia (últimos N turnos
              sin filtro de tema) además de core_memory y recuerdos.
        es_multitool: True si el plan activo tiene más de 1 paso. Fuerza
              MAX_TURNOS_CONTEXTO_PLAN (ventana adaptativa, ver arriba).
        orden: pedido original del usuario. Matchea los intentos fallidos
              previos ([INTENTOS RECIENTES]) contra lo que se pide ahora.

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
    perfil = []
    if prefs:
        if prefs.get("nombre_usuario"):
            perfil.append(f"  nombre: {prefs['nombre_usuario']}")
        if prefs.get("navegador"):
            perfil.append(f"  navegador: {prefs['navegador']}")
        if prefs.get("notas"):
            perfil.extend(f"  nota: {nota}" for nota in prefs["notas"][-10:] if nota)

    # Perfil de Account (Settings → Account): género → pronombres con los que
    # dirigirse al usuario, y fecha de nacimiento (relevante solo en su
    # cumpleaños). El avatar NUNCA se inyecta: es solo para la UI.
    perfil.extend(_lineas_perfil_cuenta())
    if perfil:
        slots["PERFIL DEL USUARIO"] = "\n".join(perfil)

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

    # ── Slot 2c: INTENTOS RECIENTES (memoria de intentos fallidos) ─────
    # "Si no dio resultado, no dio resultado": los intentos que ya fallaron
    # se inyectan para que el modelo NO repita lo que ya probó y cierre con
    # lo que hay en vez de convertir una búsqueda fallida en tarea larga.
    intentos = _intentos_recientes(tema, orden)
    if intentos:
        slots["INTENTOS RECIENTES"] = (
            "Acciones que YA se intentaron y NO dieron resultado.\n"
            "Si el pedido actual coincide con una, NO la reintentes: "
            "reportá que no hay resultados y cerrá ahí.\n" + intentos
        )

    # ── Slot 3: CONVERSACIÓN ───────────────────────────────────────────
    # Con tema: solo turnos de ese tema (y sesión, si se sabe). SIN tema no
    # hay criterio de relevancia → últimos N turnos por recencia, de la
    # sesión actual si se conoce (ver _turnos_para_contexto).
    turnos_relevantes = _turnos_para_contexto(
        tema, sesion_id, mem, limite_turnos
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

    # ── Slot 5: RESULTADOS DE TOOLS (follow-ups: "qué encontraste") ─────
    # Fix del bug "síntesis inventada": en un turno follow-up el modelo debe
    # ver las salidas REALES de las últimas tools de la sesión, verbatim.
    # Sin esto, "dame una síntesis de lo que encontraste" se respondía sin
    # ningún dato y el modelo rellenaba con IPs/puertos/procesos inventados.
    # Los datos van en bloques DELIMITADOS (<<<DATOS_TOOL ... DATOS_TOOL>>>)
    # y sanitizados: son datos observados, NUNCA instrucciones — una salida
    # maliciosa no puede cerrar el bloque ni inyectar órdenes (prompt
    # injection). No se inyecta en pasos multi-tool (anti-contaminación).
    if not es_multitool:
        resultados = obtener_resultados_tools(
            mem, sesion_id=sesion_id or _sesion_actual(mem),
            limit=MAX_RESULTADOS_TOOLS_CONTEXTO,
        )
        if resultados:
            lineas_rt = [INSTRUCCION_DATOS_TOOL]
            for r in resultados:
                cuerpo, aviso = truncar_lineas(
                    str(r.get("resultado") or ""), MAX_LINEAS_RESULTADO_TOOL)
                encabezado = f"{r.get('tool', '?')} {r.get('fecha', '')}"
                if aviso:
                    encabezado += f" {aviso}"
                lineas_rt.append(delimitar_bloque_datos(encabezado, cuerpo))
            slots["RESULTADOS DE TOOLS DEL TURNO ANTERIOR"] = "\n\n".join(lineas_rt)

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
    # Lo que REALMENTE entra al contexto (misma fuente de verdad que
    # construir_contexto_memoria). Antes, con tema vacío, el dump contaba
    # solo obtener_turnos_por_tema → reportaba "0 inyectados" aunque la
    # recuperación por recencia sí inyectara turnos.
    turnos_inyectados = _turnos_para_contexto(tema, sesion_id, mem, limite_turnos)
    turnos_tema = (
        [t for t in turnos_inyectados if isinstance(t, dict) and t.get("tema") == tema]
        if tema else []
    )
    resultados_tools = obtener_resultados_tools(
        mem, sesion_id=sesion_id or _sesion_actual(mem),
        limit=MAX_RESULTADOS_TOOLS_CONTEXTO,
    )

    slots_activos = ["SISTEMA"]
    if recuerdos_count:
        slots_activos.append("RECUERDOS")
    if turnos_inyectados or turnos:
        slots_activos.append("CONVERSACIÓN")
    if tema not in _TEMAS_SIN_COMANDOS and comandos:
        slots_activos.append("COMANDOS RECIENTES")
    if resultados_tools:
        slots_activos.append("RESULTADOS DE TOOLS DEL TURNO ANTERIOR")

    lineas = [
        "=== DEBUG: CONTEXT DUMP ===",
        f"Tema:              {tema or '(sin tema)'}",
        f"Core memory:       {len(core)} claves",
        f"Slots activos:     {', '.join(slots_activos)}",
        f"Recuerdos:         {recuerdos_count} inyectados",
        (f"Turnos (tema):     {len(turnos_inyectados)} inyectados "
         f"({'por recencia de la sesión' if not tema else f'{len(turnos_tema)} del tema'}) "
         f"(límite={limite_turnos}) de {len(turnos)} en RAM"),
        f"Comandos en RAM:   {len(comandos)}",
        f"Resultados tools:  {len(resultados_tools)} de la sesión (para follow-ups)",
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
        from core.memory.central_store_v2 import get_central_memory
        central = get_central_memory()
        # touch=False: la inyección de contexto NO es "uso". Antes el solo
        # aparecer en el prompt los reforzaba (+0.05 por turno), un bucle
        # de ricos-más-ricos que ensuciaba la señal de strength.
        resultados = central.search(tema or "", limit=limit, touch=False)
        if not resultados:
            return ""
        lineas = [f"  - {r['summary']}" for r in resultados]
        return "\n".join(lineas)[:max_chars]
    except Exception:
        return ""


def _intentos_recientes(tema: str, orden: str = "", *, limit: int = 5,
                        max_chars: int = 600) -> str:
    """Intentos fallidos recientes (outcomes de la memoria central).

    El punto: "si no dio resultado, no dio resultado". Inyectar qué
    búsquedas/lecturas ya fallaron le dice al modelo qué NO reintentar,
    en este turno y en los próximos (los outcomes tienen TTL). Fail-open
    por diseño: cualquier error devuelve "".
    """
    import os
    if os.environ.get("AETHER_CENTRAL_MEMORY", "1") == "0":
        return ""
    try:
        from core.memory.central_store_v2 import get_central_memory
        central = get_central_memory()
        resultados = central.recent_outcomes(
            query=f"{tema} {orden}".strip(), limit=limit)
        if not resultados:
            return ""
        lineas = []
        for r in resultados:
            data = r.get("data") or {}
            count = int(data.get("count", 1))
            lineas.append(f"  - {r.get('summary', '')} (veces: {count})")
        return "\n".join(lineas)[:max_chars]
    except Exception:
        return ""


def _sesion_actual(mem: dict) -> str:
    """Retorna el sesion_id del turno más reciente en RAM."""
    turnos = mem.get("conversacion") or []
    if turnos:
        return turnos[-1].get("sesion_id", "")
    return ""
