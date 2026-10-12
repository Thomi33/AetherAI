"""
Tests de la ALINEACIÓN DEL GROUNDING (2026-10).

Bug real que motivó el cambio (auditoría de solo lectura del almacenamiento
de Arch, 24 pasos de tool calling): el validador descartó la síntesis
completa por 4 ítems que NO eran invenciones —

    ['/home/thomi/.npm/_cacache)', '/home/thomi/.gradle/caches',
     'mount', 'Bottles']

eran (a) una ruta con ')' arrastrado por la regex de extracción, (b) una
ruta en otra forma, (c) comandos ejecutados citados entre backticks y
(d) un nombre de app con otra capitalización.

Política nueva (ver core/agent/grounding.py, sección ALINEACIÓN):
- Nivel 1 — reparación determinista de la extracción/matching: rutas sin
  puntuación de cierre, proceso case-insensitive, equivalencia ~ ↔ home ↔
  relativa, tools del sistema y comandos ejecutados como corpus.
- Nivel 2 — clasificación hard/soft: hard → regeneración con feedback
  ESPECÍFICO → sanitizado parcial → recién ahí resumen determinista;
  soft → aviso honesto de "no verificado", sin descartar.
- Nivel 3 — resumen determinista con las salidas más informativas y la
  lista de comandos ejecutados; el historial guarda la respuesta VALIDADA.

Todo determinista: la suite NO requiere Ollama.
"""
from core.agent.grounding import (
    validate_grounding,
    clasificar_sin_respaldo,
    sanitizar_texto_sin_respaldo,
    resumen_determinista,
    construir_feedback_regeneracion,
    aviso_no_verificados,
)
from core.memory.memory_manager import normalizar_mem


# Salidas típicas de una auditoría de almacenamiento (como la del bug).
SALIDA_DU = (
    "2.5G  /home/thomi\n"
    "812M  /home/thomi/.npm\n"
    "411M  /home/thomi/.npm/_cacache\n"
    "1.2G  /home/thomi/.gradle\n"
    "322M  /home/thomi/.cache\n"
)
SALIDA_MOUNT = (
    "sysfs on /sys type sysfs (rw,nosuid,nodev,noexec,relatime)\n"
    "/dev/nvme0n1p2 on / type btrfs (rw,relatime,ssd,space_cache=v2)\n"
    "tmpfs on /run type tmpfs (rw,nosuid,nodev,size=1631496k)\n"
)
SALIDA_FLATPAK_DIRS = (
    "1.1G  /home/thomi/.var/app/com.usebottles.bottles\n"
    "96M   /home/thomi/.var/app/org.gimp.GIMP\n"
)
SALIDA_NETSTAT = "\n".join(
    ["Active Internet connections (w/o servers)",
     "Proto Recv-Q Send-Q Local Address Foreign Address State",
     *[f"tcp 0 0 192.168.1.10:{40000 + i} "
       f"140.82.121.{i % 200 + 1}:443 ESTABLISHED" for i in range(34)],
     "tcp 0 0 127.0.0.1:42000 127.0.0.1:8000 ESTABLISHED"])


# ══════════════════════════════════════════════════════════════════════
# NIVEL 1 — reparación de la extracción y del matching
# ══════════════════════════════════════════════════════════════════════

def test_ruta_con_parentesis_final_no_es_falso_positivo():
    """El ítem exacto del bug: '/home/thomi/.npm/_cacache)' con ')' de
    cierre arrastrado. Ahora la puntuación se recorta y la ruta limpia
    sí matchea el corpus."""
    respuesta = ("El mayor consumo está en la caché de npm "
                 "(/home/thomi/.npm/_cacache) según el relevamiento.")
    ok, ungrounded = validate_grounding(respuesta, [SALIDA_DU])
    assert ok, ungrounded


def test_comando_ejecutado_es_citable():
    """El corpus de validación incluye los comandos ejecutados: citar
    `mount` / `du` entre backticks no es invención (el output de mount
    jamás contiene la palabra 'mount')."""
    respuesta = "Revisé los puntos de montaje con `mount` y el consumo con `du`."
    # Sin comandos en el corpus sigue fallando (comportamiento previo):
    ok_viejo, _ = validate_grounding(respuesta, [SALIDA_MOUNT, SALIDA_DU])
    assert not ok_viejo
    # Con los comandos ejecutados en el corpus (como hace el agent loop):
    ok, ungrounded = validate_grounding(
        respuesta, [SALIDA_MOUNT, SALIDA_DU, "mount", "du -h /home/thomi"])
    assert ok, ungrounded


def test_proceso_case_insensitive():
    """'Bottles' (nombre legible de la app) vs 'com.usebottles.bottles'
    (flatpak ID en las rutas de du): capitalización distinta, misma app."""
    respuesta = "También hay datos de la app `Bottles` en el home."
    ok, ungrounded = validate_grounding(respuesta, [SALIDA_FLATPAK_DIRS])
    assert ok, ungrounded


def test_ruta_relativa_o_tilde_en_la_salida_cuenta():
    """Equivalencia de forma: la salida tiene rutas relativas (du con cwd
    en el home) o con '~', y la respuesta usa la forma absoluta (o al
    revés). Es la MISMA ruta del mismo usuario."""
    salida_relativa = "1.2G  .gradle\n487M  .gradle/caches\n96M   .gradle/wrapper\n"
    ok, ungrounded = validate_grounding(
        "/home/thomi/.gradle/caches ocupa 487M.", [salida_relativa])
    assert ok, ungrounded
    ok2, ungrounded2 = validate_grounding(
        "~/.gradle/caches ocupa 487M.", ["/home/thomi/.gradle/caches  487M"])
    assert ok2, ungrounded2


def test_nombre_de_tool_citado_no_es_invencion():
    """Citar entre backticks la tool que uno usó (`shell`, `fs_read`) es un
    reporte honesto de la ejecución."""
    ok, ungrounded = validate_grounding(
        "Usé `fs_read` y `shell` para relevar el sistema.", [SALIDA_DU])
    assert ok, ungrounded


def test_datos_inventados_siguen_sin_respaldo():
    """Regresión del control positivo: IPs/puertos/procesos inventados y
    rutas ajenas a los datos siguen siendo HARD — la alineación no rebaja
    los controles anti-alucinación."""
    ok, ungrounded = validate_grounding(
        "Hay una conexión desde 185.234.12.45 al puerto 4444 con `payload.py`.",
        [SALIDA_NETSTAT])
    assert not ok
    hard, soft = clasificar_sin_respaldo(ungrounded, [SALIDA_NETSTAT])
    assert {v for _, v in hard} >= {"185.234.12.45", "4444", "payload.py"}
    assert soft == []


# ══════════════════════════════════════════════════════════════════════
# NIVEL 2 — clasificación hard/soft, feedback y sanitizado
# ══════════════════════════════════════════════════════════════════════

def test_clasificacion_soft_para_rutas_variante():
    """La ruta que EXTIENDE una ruta observada ('/home/thomi/.gradle' →
    '/home/thomi/.gradle/caches') es soft (inferencia marcable), no hard.
    Una ruta de un home raíz ('/home/thomi') NO cuenta como relación."""
    hard, soft = clasificar_sin_respaldo(
        [("ruta", "/home/thomi/.gradle/caches")], [SALIDA_DU])
    assert hard == []
    assert soft == [("ruta", "/home/thomi/.gradle/caches")]
    # ...pero una ruta inventada sobre un netstat (sin rutas) es hard.
    hard2, soft2 = clasificar_sin_respaldo(
        [("ruta", "/home/thomi/.ssh/authorized_keys")], [SALIDA_NETSTAT])
    assert hard2 and not soft2


def test_solo_soft_no_regenera_ni_descarta():
    """Con solo ítems soft la respuesta se entrega ENTERA con el aviso
    honesto de no verificado: cero regeneraciones, cero descartes."""
    import core.agent.graph_nodes as gn

    texto = ("Lo más pesado: /home/thomi/.gradle/caches según el du. "
             "Nada más para destacar en el home.")

    def _no_regenerar(*a, **k):
        raise AssertionError("Con solo ítems soft no se regenera")

    out = gn._aplicar_validacion_grounding(
        texto, "auditoría de solo lectura del almacenamiento",
        [SALIDA_DU], _no_regenerar)
    assert out.startswith(texto)                 # se entrega entera...
    assert "no aparece(n) literalmente" in out   # ...con el aviso honesto


def test_regeneracion_recibe_feedback_especifico():
    """La instrucción correctiva lista los ítems EXACTOS que fallaron (el
    bug de antes: regla genérica, el modelo no sabía qué quitar)."""
    import core.agent.graph_nodes as gn

    feedbacks = []

    def fake_regenerar(feedback=""):
        feedbacks.append(feedback)
        return ""      # regeneración fallida → cae a sanitizado/resumen

    texto = "Detecté una conexión desde 185.234.12.45 al puerto 4444 y `payload.py`."
    out = gn._aplicar_validacion_grounding(
        texto, "revisa mi trafico de red", [SALIDA_NETSTAT], fake_regenerar)

    assert feedbacks, "el feedback correctivo debía viajar en la regeneración"
    assert "185.234.12.45" in feedbacks[0]
    assert "4444" in feedbacks[0]
    # y al no quedar nada rescatable, el resumen no cita los inventos:
    assert "185.234.12.45" not in out
    assert "verificable" in out


def test_sanitizado_parcial_conserva_lo_respaldado():
    """Si la regeneración insiste con datos inventados, se quitan SOLO las
    oraciones con lo inventado y se conserva el resto verificado — el
    descarte total era lo que inutilizaba al agente tras 24 pasos."""
    import core.agent.graph_nodes as gn

    texto = ("Relevamiento: 40 conexiones establecidas, casi todas a "
             "140.82.121.1:443 (HTTPS saliente). Nada raro en los puertos "
             "en escucha. ADVERTENCIA: hay una conexión desde 185.234.12.45 "
             "al puerto 4444 con `payload.py`.")

    def regenerar_mal(feedback=""):
        return texto      # insiste con los inventos

    out = gn._aplicar_validacion_grounding(
        texto, "revisa mi trafico de red", [SALIDA_NETSTAT], regenerar_mal)
    # Lo respaldado sobrevive:
    assert "140.82.121.1:443" in out
    assert "40 conexiones" in out
    assert "Nada raro" in out
    # Lo inventado no:
    assert "185.234.12.45" not in out
    assert "4444" not in out
    assert "payload.py" not in out
    # Y el aviso es honesto sobre lo que se quitó:
    assert "Quité" in out


def test_sanitizar_quita_solo_la_oracion_mala():
    """Unit test del sanitizador: oración por oración."""
    texto = ("Primera oración verificada. Segunda cita 185.234.12.45 y es "
             "mala. Tercera también verificada.")
    limpio, quitados = sanitizar_texto_sin_respaldo(
        texto, [("ipv4", "185.234.12.45")])
    assert "Primera oración verificada." in limpio
    assert "Tercera también verificada." in limpio
    assert "185.234.12.45" not in limpio
    assert quitados == ["185.234.12.45"]


def test_aviso_y_feedback_listan_los_items():
    soft = [("ruta", "/home/thomi/.gradle/caches")]
    hard = [("ipv4", "185.234.12.45")]
    assert "/home/thomi/.gradle/caches" in aviso_no_verificados(soft)
    fb = construir_feedback_regeneracion(hard, soft)
    assert "185.234.12.45" in fb
    assert "/home/thomi/.gradle/caches" in fb


# ══════════════════════════════════════════════════════════════════════
# NIVEL 3 — resumen determinista útil y flujo del agent loop
# ══════════════════════════════════════════════════════════════════════

def test_resumen_incluye_salidas_informativas_y_comandos():
    """Antes solo salidas[-3:] × 3 líneas: tras 24 pasos el usuario se
    quedaba con 9 líneas de datos triviales. Ahora entran las salidas más
    informativas y la lista de comandos ejecutados."""
    salidas = [f"ok {i}" for i in range(25)] + [
        "du:\n" + "\n".join(f"/home/thomi/.dir{i:03d}  1G" for i in range(10))]
    resumen = resumen_determinista(
        salidas, [("ipv4", "1.2.3.4")],
        comandos=["du -h /home/thomi", "mount"])
    assert "du:" in resumen
    assert "/home/thomi/.dir000" in resumen
    assert "$ du -h /home/thomi" in resumen
    assert "$ mount" in resumen
    assert "verificable" in resumen


# ── Helpers del agent loop (mockea LLM y tools; nada de Ollama) ─────────

def _preparar_loop(monkeypatch, fake_llm, fake_node=None):
    import core.agent.graph_nodes as gn
    from core.agent import tool_registry as tr
    monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
    monkeypatch.setattr(tr, "construir_tools_ollama", lambda: [])
    monkeypatch.setattr(tr, "validar_tool_call", lambda t, a: (True, ""))
    if fake_node is not None:
        monkeypatch.setattr(tr, "get_node_func", lambda t: fake_node)
    return gn


def _correr_loop(gn, orden, mem, max_rondas=12, sesion_id="sesion-alin"):
    state = {"orden": orden, "mem": mem, "sesion_id": sesion_id,
             "agent_messages": None, "agent_pasos_log": None}
    for _ in range(max_rondas):
        out = gn.node_agent_loop(state)
        state.update(out)
        if out.get("done"):
            break
    return state


def test_sintesis_honesta_del_incidente_pasa_tal_cual(monkeypatch):
    """Reproducción end-to-end del incidente: auditoría de almacenamiento,
    el modelo ejecuta `du` y cierra con una síntesis honesta que cita el
    comando (`du`) y rutas reales — antes el validador la DESCARTABA entera
    (ítems: `mount`, rutas normalizadas); ahora se entrega tal cual y el
    agente vuelve a ser útil tras una investigación larga."""
    def fake_llm(messages, tools, min_predict=None):
        if fake_llm.n == 0:
            fake_llm.n += 1
            return {"content": "", "tool_calls": [{"function": {
                "name": "shell",
                "arguments": {"instruccion": "du -h /home/thomi | sort -h"}}}]}
        fake_llm.n += 1
        return {"content": (
            "Auditoría lista: usé `du` sobre tu home. Lo más pesado: "
            "/home/thomi/.npm/_cacache (411M) y /home/thomi/.gradle (1.2G). "
            "Nada fuera de lo normal."),
            "tool_calls": []}
    fake_llm.n = 0

    gn = _preparar_loop(monkeypatch, fake_llm,
                        lambda e: {"shell_output": SALIDA_DU})
    mem = normalizar_mem({"sesion_id": "sesion-alin"})
    state = _correr_loop(
        gn, "auditoría de solo lectura del almacenamiento", mem)

    final = state["final_response"]
    assert final.startswith("Auditoría lista")    # NO se descartó
    assert "_cacache" in final
    assert "/home/thomi/.gradle" in final
    # El historial guarda la respuesta VALIDADA, no una versión descartada:
    assert state["agent_messages"][-1]["content"].startswith("Auditoría lista")


def test_respuesta_validada_no_el_hallazgo_descartado(monkeypatch):
    """Alineación: cuando el sanitizado parcial repara la respuesta, el
    historial del agent loop conserva lo que el usuario RECIBIÓ (la parte
    respaldada), no la oración con datos inventados."""
    def fake_llm(messages, tools, min_predict=None):
        if fake_llm.n == 0:
            fake_llm.n += 1
            return {"content": "", "tool_calls": [{"function": {
                "name": "shell",
                "arguments": {"instruccion": "netstat -an"}}}]}
        fake_llm.n += 1
        return {"content": (
            "Hay una conexión desde 185.234.12.45 al puerto 4444 con "
            "`payload.py`. El resto del relevamiento no dio hallazgos."),
            "tool_calls": []}
    fake_llm.n = 0

    gn = _preparar_loop(monkeypatch, fake_llm,
                        lambda e: {"shell_output": SALIDA_NETSTAT})
    mem = normalizar_mem({"sesion_id": "sesion-alin-2"})
    state = _correr_loop(gn, "revisá mi tráfico de red", mem,
                         sesion_id="sesion-alin-2")

    final = state["final_response"]
    assert "185.234.12.45" not in final
    assert "payload.py" not in final
    assert "El resto del relevamiento no dio hallazgos." in final
    assert state["agent_messages"][-1]["content"] == final


