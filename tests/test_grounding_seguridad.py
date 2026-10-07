"""
Tests de regresión del bug "síntesis inventada" (sesión de seguridad real).

Escenario original: el usuario pidió "revisá mi tráfico de red" (netstat vía
shell) y luego "dame una síntesis de lo que encontraste, no reintentes
nada". El modelo, sin las salidas reales en contexto, inventó una intrusión
completa (IPs 185.234.12.45 / 192.168.1.100, puertos 4444/5000/8080,
procesos `payload.py`, un curl a api.threatconnect.com).

Todo con LLM y tools MOCKEADOS: la suite NO requiere Ollama.
"""
import pytest

from core.memory.memory_manager import (
    normalizar_mem,
    registrar_resultado_tool,
    obtener_resultados_tools,
)


# ══════════════════════════════════════════════════════════════════════
# FIXTURES — la salida REAL del netstat (40 conexiones; 192.168.1.100 NO
# está, ni los puertos 4444/5000/8080, ni payload.py)
# ══════════════════════════════════════════════════════════════════════

def _netstat_real_40_conexiones() -> str:
    lineas = [
        "Active Internet connections (w/o servers)",
        "Proto Recv-Q Send-Q Local Address           Foreign Address         State",
    ]
    # 34 conexiones HTTPS externas (GitHub-ish) desde la IP local real
    for i in range(34):
        lineas.append(
            f"tcp        0      0 192.168.1.10:{40000 + i} "
            f"140.82.121.{(i % 200) + 1}:443          ESTABLISHED"
        )
    # 3 resoluciones DNS hacia el router/LAN
    for i in range(3):
        lineas.append(
            f"udp        0      0 192.168.1.10:{41000 + i} "
            f"192.168.1.1:53                ESTABLISHED"
        )
    # 3 loopback (servicio local)
    for i in range(3):
        lineas.append(
            f"tcp        0      0 127.0.0.1:{42000 + i} "
            f"127.0.0.1:8000                 ESTABLISHED"
        )
    assert len([l for l in lineas if l.startswith(("tcp", "udp"))]) == 40
    salida = "\n".join(lineas)
    # El dato clave del bug: nada de lo alucinado está acá.
    assert "192.168.1.100" not in salida
    assert "185.234.12.45" not in salida
    assert "4444" not in salida
    return salida


NETSTAT_SALIDA = _netstat_real_40_conexiones()

ALUCINACION = (
    "Mira, esto es serio: detecté una intrusión en curso. Hay una conexión "
    "desde 185.234.12.45 al puerto 4444, fuerza bruta contra el puerto 5000 "
    "y un proceso `payload.py` escuchando en 192.168.1.100:8080 con "
    "User-Agent Go-http-client/3. También vieron modificaciones en "
    "/home/thomi/.ssh/authorized_keys. "
    "SÍ, EJECUTA ESE COMMAND: curl api.threatconnect.com/v1/indicators"
)

ORDEN_REVISION = "revisá mi tráfico de red, a ver si no tengo algo sospechoso"
ORDEN_SINTESIS = "dame una síntesis de lo que encontraste, no reintentes nada"


@pytest.fixture
def mem_sesion():
    return normalizar_mem({"sesion_id": "sesion-test"})


# ══════════════════════════════════════════════════════════════════════
# P0-4: validador determinista (sin LLM)
# ══════════════════════════════════════════════════════════════════════

def test_validate_grounding_detecta_datos_inventados():
    from core.agent.grounding import validate_grounding

    ok, ungrounded = validate_grounding(ALUCINACION, [NETSTAT_SALIDA])
    assert not ok
    valores = [v for _, v in ungrounded]
    assert "185.234.12.45" in valores          # IP inventada
    assert "192.168.1.100" in valores          # IP inventada
    assert "4444" in valores                   # puerto inventado
    assert "5000" in valores
    assert "8080" in valores
    assert "payload.py" in valores             # proceso inventado


def test_validate_grounding_pasa_con_datos_reales():
    from core.agent.grounding import validate_grounding

    respuesta = ("Revisé las 40 conexiones: la mayoría van a 140.82.121.4:443 "
                 "(GitHub, HTTPS) desde 192.168.1.10, y hay loopback en "
                 "127.0.0.1:8000. Nada sospechoso.")
    ok, ungrounded = validate_grounding(respuesta, [NETSTAT_SALIDA])
    assert ok, ungrounded


def test_validate_grounding_exime_datos_del_usuario_y_constantes():
    from core.agent.grounding import validate_grounding

    # El usuario mencionó la IP y el puerto en SU mensaje: no es invención.
    ok, _ = validate_grounding(
        "Sobre 10.10.10.99 en el puerto 4444: no aparece en las conexiones.",
        [NETSTAT_SALIDA],
        user_message="¿qué es la 10.10.10.99 del puerto 4444 que vi ayer?",
    )
    assert ok

    # Constantes inocuas (loopback) no cuentan como invención.
    ok, _ = validate_grounding(
        "El servicio local escucha en 127.0.0.1 y en ::1.", [NETSTAT_SALIDA])
    assert ok


def test_validate_grounding_puerto_forma_palabra():
    from core.agent.grounding import validate_grounding

    ok, ungrounded = validate_grounding(
        "El atacante usó el puerto 4444.", [NETSTAT_SALIDA])
    assert not ok
    assert "4444" in [v for _, v in ungrounded]


def test_resumen_determinista_solo_con_datos_reales():
    from core.agent.grounding import resumen_determinista

    resumen = resumen_determinista([NETSTAT_SALIDA], [("ipv4", "185.234.12.45")])
    assert "185.234.12.45" not in resumen
    assert "verificable" in resumen
    assert "140.82.121.1:443" in resumen      # primera línea útil de datos reales


# ══════════════════════════════════════════════════════════════════════
# Helpers del agent loop (mockea LLM y tools; nada de Ollama)
# ══════════════════════════════════════════════════════════════════════

def _preparar_loop(monkeypatch, fake_llm, fake_node=None):
    import core.agent.graph_nodes as gn
    from core.agent import tool_registry as tr
    monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
    monkeypatch.setattr(tr, "construir_tools_ollama", lambda: [])
    monkeypatch.setattr(tr, "validar_tool_call", lambda t, a: (True, ""))
    if fake_node is not None:
        monkeypatch.setattr(tr, "get_node_func", lambda t: fake_node)
    return gn


def _correr_loop(gn, orden, mem, max_rondas=12, sesion_id="sesion-test"):
    state = {"orden": orden, "mem": mem, "sesion_id": sesion_id,
             "agent_messages": None, "agent_pasos_log": None}
    for _ in range(max_rondas):
        out = gn.node_agent_loop(state)
        state.update(out)
        if out.get("done"):
            break
    return state


# ══════════════════════════════════════════════════════════════════════
# ESCENARIO ORIGINAL: shell→netstat real, luego síntesis que alucina
# ══════════════════════════════════════════════════════════════════════

def test_escenario_original_sintesis_no_alucina(monkeypatch, mem_sesion):
    """Turno 1: shell devuelve el netstat real (40 conexiones). Turno 2:
    síntesis con un LLM mock que alucina la intrusión. El validador la
    detecta, la regeneración sigue alucinando y la respuesta final es el
    resumen determinista: SIN IPs/puertos/procesos ausentes en la salida."""
    llamadas_llm = []

    def fake_llm(messages, tools, min_predict=None):
        llamadas_llm.append(messages[-1])
        n = fake_llm.ronda
        fake_llm.ronda += 1
        if n == 0:   # turno 1: el modelo pide correr el relevador de red
            return {"content": "", "tool_calls": [{"function": {
                "name": "shell",
                "arguments": {"instruccion": "netstat -an | grep ESTABLISHED"}}}]}
        if n == 1:   # turno 1: cierre honesto
            return {"content": "Relevamiento listo.", "tool_calls": []}
        if n == 2:   # turno 2: ALUCINACIÓN (así se comportó el modelo real)
            return {"content": ALUCINACION, "tool_calls": []}
        # regeneración: el modelo insiste con los mismos inventos
        return {"content": ALUCINACION, "tool_calls": []}
    fake_llm.ronda = 0

    def fake_node_shell(estado):
        return {"shell_output": NETSTAT_SALIDA}

    gn = _preparar_loop(monkeypatch, fake_llm, fake_node_shell)

    # ── Turno 1: la revisión de red ──
    state1 = _correr_loop(gn, ORDEN_REVISION, mem_sesion)
    assert state1["done"] is True
    assert len(state1["agent_pasos_log"]) == 1
    # La salida REAL quedó registrada en mem para los follow-ups (P0-2).
    resultados = obtener_resultados_tools(mem_sesion, sesion_id="sesion-test")
    assert resultados and "140.82.121" in resultados[-1]["resultado"]
    assert "185.234.12.45" not in resultados[-1]["resultado"]

    # ── Turno 2: la síntesis que alucina ──
    state2 = _correr_loop(gn, ORDEN_SINTESIS, mem_sesion)
    assert state2["done"] is True
    final = state2["final_response"]
    # La respuesta final NO contiene nada de lo aluculado.
    assert "185.234.12.45" not in final
    assert "192.168.1.100" not in final
    assert "4444" not in final
    assert "5000" not in final
    assert "8080" not in final
    assert "payload.py" not in final
    assert "threatconnect" not in final
    # Y avisa que se descartó lo no verificable, citando solo datos reales.
    assert "verificable" in final
    # Hubo exactamente una regeneración (el LLM se corrigió una vez).
    assert len(llamadas_llm) == 4   # 2 del turno 1 + alucinación + regeneración


def test_escenario_original_sintesis_honesta_pasa_tal_cual(monkeypatch, mem_sesion):
    """Si el modelo sintetiza SOLO con datos reales, la respuesta pasa tal
    cual (el validador no toca respuestas honestas)."""
    def fake_llm(messages, tools, min_predict=None):
        if fake_llm.n == 0:
            fake_llm.n += 1
            return {"content": "", "tool_calls": [{"function": {
                "name": "shell",
                "arguments": {"instruccion": "netstat -an | grep ESTABLISHED"}}}]}
        fake_llm.n += 1
        return {"content": ("Síntesis: 40 conexiones establecidas, casi todas "
                            "a 140.82.121.1:443 (HTTPS saliente) y tres "
                            "loopback a 127.0.0.1:8000. No veo nada raro."),
                "tool_calls": []}
    fake_llm.n = 0

    gn = _preparar_loop(monkeypatch, fake_llm,
                        lambda e: {"shell_output": NETSTAT_SALIDA})
    _correr_loop(gn, ORDEN_REVISION, mem_sesion)
    state2 = _correr_loop(gn, ORDEN_SINTESIS, mem_sesion)
    assert "140.82.121.1:443" in state2["final_response"]
    assert "no veo nada raro" in state2["final_response"].lower()


# ══════════════════════════════════════════════════════════════════════
# P0-3: síntesis SIN datos → mensaje determinista, LLM jamás invocado
# ══════════════════════════════════════════════════════════════════════

def test_sintesis_sin_datos_respuesta_determinista_sin_llm(monkeypatch, mem_sesion):
    import core.agent.graph_nodes as gn

    def _boom(*a, **k):
        raise AssertionError("El LLM no debía ser invocado en este turno")

    monkeypatch.setattr(gn, "_llm_chat", _boom)
    monkeypatch.setattr(gn, "_llm_chat_agente", _boom)
    monkeypatch.setattr(gn, "_confirmar_referencia_anaforica_llm", _boom)

    state = {"orden": ORDEN_SINTESIS, "mem": mem_sesion,
             "sesion_id": "sesion-test"}
    out = gn.node_planner(state)

    assert out["done"] is True
    assert out["final_response"] == gn.MENSAJE_SIN_DATOS
    assert "No tengo datos de esa revisión" in out["final_response"]
    # No hay plan ni agent loop: ninguna tool (memory incluida) se dispara.
    assert out.get("plan_pasos") == []
    assert not out.get("agent_activo")


def test_sintesis_sin_datos_en_el_agent_loop_es_determinista(monkeypatch):
    """Red de seguridad: si una síntesis llega igual al agent loop sin ningún
    dato (p. ej. redacción distinta), la respuesta final sigue siendo el
    mensaje determinista y no el texto alucinado del modelo."""
    def fake_llm(messages, tools, min_predict=None):
        return {"content": ALUCINACION, "tool_calls": []}

    gn = _preparar_loop(monkeypatch, fake_llm)
    mem = normalizar_mem({"sesion_id": "s-vacia"})
    state = _correr_loop(gn, ORDEN_SINTESIS, mem, sesion_id="s-vacia")

    assert state["final_response"] == gn.MENSAJE_SIN_DATOS
    assert "185.234.12.45" not in state["final_response"]


# ══════════════════════════════════════════════════════════════════════
# P0-1: tema vacío → recencia de la sesión, sin mezclar otras sesiones
# ══════════════════════════════════════════════════════════════════════

def _mem_con_turnos():
    mem = normalizar_mem({"sesion_id": "sesion-A"})
    for texto in ("turno viejo A1", "turno viejo A2"):
        mem["conversacion"].append(
            {"rol": "usuario", "texto": texto, "fecha": "2026-10-06",
             "sesion_id": "sesion-A", "tema": ""})
    mem["conversacion"].append(
        {"rol": "usuario", "texto": "turno de OTRA sesion", "fecha": "2026-10-05",
         "sesion_id": "sesion-B", "tema": ""})
    return mem


def test_tema_vacio_recupera_recencia_de_la_sesion(monkeypatch):
    from core.memory import context_builder as cb

    monkeypatch.setattr(cb, "obtener_recuerdos", lambda **k: [])
    monkeypatch.setattr(cb, "obtener_ultimos_turnos", lambda n: [])

    mem = _mem_con_turnos()
    # tema VACÍO: la recuperación devuelve los últimos N de la sesión...
    contexto = cb.construir_contexto_memoria(mem, tema="", sesion_id="sesion-A")
    assert "turno viejo A2" in contexto
    assert "turno viejo A1" in contexto
    # ...y NO mezcla turnos de otras sesiones.
    assert "turno de OTRA sesion" not in contexto


def test_tema_vacio_el_dump_refleja_la_recencia(monkeypatch):
    from core.memory import context_builder as cb

    monkeypatch.setattr(cb, "obtener_recuerdos", lambda **k: [])
    monkeypatch.setattr(cb, "obtener_ultimos_turnos", lambda n: [])

    mem = _mem_con_turnos()
    dump = cb.construir_context_dump(mem, tema="", sesion_id="sesion-A")
    # Antes decía "0 inyectados" con tema vacío aunque hubiera turnos.
    assert "2 inyectados" in dump
    assert "por recencia de la sesión" in dump


def test_con_tema_el_comportamiento_no_cambia(monkeypatch):
    from core.memory import context_builder as cb

    monkeypatch.setattr(cb, "obtener_recuerdos", lambda **k: [])
    monkeypatch.setattr(
        cb, "obtener_turnos_por_tema",
        lambda tema, sesion_id="", limit=10: [
            {"rol": "usuario", "texto": f"turno {tema} 1", "sesion_id": sesion_id}])

    mem = _mem_con_turnos()
    contexto = cb.construir_contexto_memoria(mem, tema="web", sesion_id="sesion-A")
    assert "turno web 1" in contexto
    assert "turno viejo A2" not in contexto


# ══════════════════════════════════════════════════════════════════════
# P0-2: slot [RESULTADOS DE TOOLS DEL TURNO ANTERIOR]
# ══════════════════════════════════════════════════════════════════════

def test_contexto_incluye_resultados_de_tools_verbatim(monkeypatch, mem_sesion):
    from core.memory import context_builder as cb

    monkeypatch.setattr(cb, "obtener_recuerdos", lambda **k: [])
    registrar_resultado_tool(
        mem_sesion, "shell", {"instruccion": "netstat -an"},
        NETSTAT_SALIDA, sesion_id="sesion-test")

    contexto = cb.construir_contexto_memoria(
        mem_sesion, tema="text", sesion_id="sesion-test")
    assert "[RESULTADOS DE TOOLS DEL TURNO ANTERIOR]" in contexto
    assert "140.82.121.1:443" in contexto          # verbatim
    assert "185.234.12.45" not in contexto


def test_contexto_trunca_resultados_con_aviso(monkeypatch, mem_sesion):
    from core.memory import context_builder as cb

    monkeypatch.setattr(cb, "obtener_recuerdos", lambda **k: [])
    salida_larga = "\n".join(f"linea {i}" for i in range(200))
    registrar_resultado_tool(
        mem_sesion, "shell", {}, salida_larga, sesion_id="sesion-test")

    contexto = cb.construir_contexto_memoria(
        mem_sesion, tema="text", sesion_id="sesion-test")
    assert "[truncado:" in contexto
    assert "omitidas]" in contexto
    assert "linea 199" not in contexto


def test_resultados_de_otra_sesion_no_se_inyectan(monkeypatch, mem_sesion):
    from core.memory import context_builder as cb

    monkeypatch.setattr(cb, "obtener_recuerdos", lambda **k: [])
    registrar_resultado_tool(
        mem_sesion, "shell", {}, "salida de otra sesion",
        sesion_id="sesion-OTRA")

    contexto = cb.construir_contexto_memoria(
        mem_sesion, tema="text", sesion_id="sesion-test")
    assert "salida de otra sesion" not in contexto
    assert "[RESULTADOS DE TOOLS" not in contexto


# ══════════════════════════════════════════════════════════════════════
# P2-9: routing de síntesis (no dispara memory ni ejecución)
# ══════════════════════════════════════════════════════════════════════

def test_sintesis_con_datos_va_a_text_y_no_dispara_memory(monkeypatch, mem_sesion):
    import core.agent.graph_nodes as gn

    registrar_resultado_tool(
        mem_sesion, "shell", {"instruccion": "netstat -an"},
        NETSTAT_SALIDA, sesion_id="sesion-test")

    def _boom(*a, **k):
        raise AssertionError("El LLM no debía ser invocado para rutear la síntesis")

    monkeypatch.setattr(gn, "_llm_chat", _boom)
    monkeypatch.setattr(gn, "_llm_chat_agente", _boom)
    monkeypatch.setattr(gn, "_confirmar_referencia_anaforica_llm", _boom)

    out = gn.node_planner({"orden": ORDEN_SINTESIS, "mem": mem_sesion,
                           "sesion_id": "sesion-test"})
    herramientas = [p["tool"] for p in out["plan_pasos"]]
    assert herramientas == ["text"]
    assert "memory" not in herramientas          # el bug: memory en el paso 1
    assert "shell" not in herramientas
    assert not out.get("agent_activo")            # no va al agent loop
    assert not out.get("done")                    # sí pasa por node_text con datos


def test_prohibicion_de_ejecucion_bloquea_tools(monkeypatch):
    """P2-8: con 'no ejecutes/no corras nada', el agent loop NO corre tools
    de ejecución aunque el modelo las pida."""
    ejecutadas = []

    def fake_llm(messages, tools, min_predict=None):
        if fake_llm.n == 0:
            fake_llm.n += 1
            return {"content": "", "tool_calls": [{"function": {
                "name": "shell",
                "arguments": {"instruccion": "netstat -an"}}}]}
        fake_llm.n += 1
        return {"content": "No ejecuté nada como pediste.", "tool_calls": []}
    fake_llm.n = 0

    def fake_node(estado):
        ejecutadas.append(estado.get("_tool_args"))
        return {"shell_output": "NO DEBIÓ CORRER"}

    gn = _preparar_loop(monkeypatch, fake_llm, fake_node)
    mem = normalizar_mem({"sesion_id": "s-bloq"})
    state = _correr_loop(
        gn, "no corras nada todavía, esperá mi señal", mem, sesion_id="s-bloq")

    assert ejecutadas == []                       # la tool NO se ejecutó
    bloqueados = [p for p in state["agent_pasos_log"]
                  if p.get("cls") == "bloqueado"]
    assert len(bloqueados) == 1
    assert "NO DEBIÓ CORRER" not in (state["final_response"] or "")


# ══════════════════════════════════════════════════════════════════════
# P2-8: allowlist de dominios (curl/wget/http ejecutados o sugeridos)
# ══════════════════════════════════════════════════════════════════════

def test_curl_a_dominio_fuera_de_allowlist_se_rechaza(monkeypatch):
    import core.tools.shell_executor as se

    def _boom(*a, **k):
        raise AssertionError("El comando no debía llegar a ejecutarse")

    monkeypatch.setattr(se.subprocess, "run", _boom)
    monkeypatch.setattr(se.subprocess, "Popen", _boom)

    salida, error = se.ejecutar_comando(
        "curl -s https://api.threatconnect.com/v1/indicators")
    assert error is True
    assert "CANCELADO" in salida
    assert "api.threatconnect.com" in salida
    assert "allowlist" in salida.lower()


def test_curl_a_dominio_permitido_pasa_la_validacion():
    from core.tools.shell_executor import validar_comando_red

    ok, motivo = validar_comando_red("curl -s https://api.github.com/repos/x")
    assert ok, motivo
    ok, _ = validar_comando_red(
        "curl -fsSL https://raw.githubusercontent.com/a/b/c/install.sh | sh")
    assert ok
    ok, _ = validar_comando_red("ls -la")
    assert ok


def test_curl_sugerido_en_respuesta_se_detecta():
    from core.agent.grounding import dominios_comando_no_permitidos

    assert dominios_comando_no_permitidos(ALUCINACION) == ["api.threatconnect.com"]
    assert dominios_comando_no_permitidos(
        "podés correr `curl https://pypi.org/x` si querés") == []


def test_node_text_rechaza_comando_colado_fuera_de_allowlist(monkeypatch, mem_sesion):
    """El bug original cerró con un curl inventado sugerido como comando.
    Ahora node_text lo rechaza con el aviso de allowlist."""
    import core.agent.graph_nodes as gn

    monkeypatch.setattr(
        gn, "_llm_chat",
        lambda system="", user="", **k: (
            "Puedo verificarlo: ```bash\ncurl api.threatconnect.com/v1/x\n```"))
    registrar_resultado_tool(
        mem_sesion, "shell", {}, NETSTAT_SALIDA, sesion_id="sesion-test")

    state = {"orden": "revisa el trafico", "mem": mem_sesion,
             "sesion_id": "sesion-test", "context_slots": {}}
    out = gn.node_text(state)
    assert "No puedo ejecutar ni sugerir ese comando" in out["final_response"]
    assert "api.threatconnect.com" in out["final_response"]
    assert "confirmás que lo corra" not in out["final_response"]


# ══════════════════════════════════════════════════════════════════════
# P2-7: fallos parciales con detalle
# ══════════════════════════════════════════════════════════════════════

def test_informe_de_fallos_detalle_por_paso():
    from core.services.graph_service import respuesta_final_o_informe

    pasos = []
    for i in range(6):
        pasos.append({"tool": "web", "args": {"query": f"busqueda {i}"},
                      "resultado": "ok", "cls": "ok"})
    pasos[2] = {
        "tool": "shell",
        "args": {"instruccion": "netstat -x"},
        "resultado": "[ERROR] ss: unsupported option\nsegunda linea",
        "cls": "error",
    }
    resultado = {"final_response": "", "agent_pasos_log": pasos}

    informe = respuesta_final_o_informe(resultado)
    assert "1 de 6" in informe
    assert "Paso 3" in informe                     # qué paso falló
    assert "shell" in informe                       # con qué tool
    assert "netstat -x" in informe                  # qué pedía
    assert "unsupported option" in informe         # el error real
    assert "afectada" in informe                    # impacto en la conclusión


def test_plan_synthesizer_valida_grounding_en_flujo_seguridad(monkeypatch):
    """Gap encontrado al reproducir el grafo completo: la síntesis de los
    resultados de net_audit pasa por node_plan_synthesizer, que TAMBIÉN debe
    validar que no se citen IPs/puertos/procesos ausentes en los datos."""
    import core.agent.graph_nodes as gn

    def fake_llm(system="", user="", **k):
        if fake_llm.n == 0:
            fake_llm.n += 1
            return ("Hay una conexión entrante desde 185.234.12.45 al puerto "
                    "4444 con un proceso `payload.py`. Es grave.")
        fake_llm.n += 1
        return "Insisto: el atacante es 185.234.12.45:4444."
    fake_llm.n = 0

    monkeypatch.setattr(gn, "_llm_chat", fake_llm)

    state = {
        "orden": "revisa mi trafico de red",
        "mem": normalizar_mem({"sesion_id": "s-sint"}),
        "sesion_id": "s-sint",
        "plan_pasos": [{"tool": "net_audit", "instruccion": "relevamiento"}],
        "plan_resultados": [NETSTAT_SALIDA],
        "context_slots": {"contexto": "[CTX]"},
    }
    out = gn.node_plan_synthesizer(state)
    final = out["final_response"]
    assert "185.234.12.45" not in final
    assert "4444" not in final
    assert "payload.py" not in final
    assert "verificable" in final
    assert fake_llm.n == 2                     # original + una regeneración
