"""Tests de la memoria de intentos (outcome memory) y de los fixes al
sistema de importancia de la memoria central.

Cubre:
- register_outcome / recent_outcomes: creación, dedup por fingerprint,
  filtro por TTL y orden por relevancia+recencia.
- search() / personality_signals(): outcomes excluidos por defecto; la
  inyección de contexto (touch=False) no refuerza memorias.
- consolidate(): outcomes decaen por DÍA; los vencidos por TTL se barren.
- learn(): dedup de resúmenes idénticos refuerza en vez de duplicar;
  refuerzo asintótico que no satura.
- Agent loop: clasificación de resultados negativos, guía terminal en el
  mensaje de tool, registro persistente y cierre forzado del loop al
  agotar el tope de fallos de una tool de búsqueda.
- Prompts: "sin resultados" como resultado válido; sin "a como de lugar".
"""
from datetime import date, timedelta, datetime

import pytest

from core.memory.central_store_v2 import CentralMemory


@pytest.fixture
def mem(tmp_path):
    return CentralMemory(str(tmp_path / "memory"))


@pytest.fixture
def central_tmp(tmp_path, monkeypatch):
    """Apunta la memoria central (singleton) a un tmp_path limpio y la
    activa. Los helpers del agent loop usan get_central_memory() sin root
    explícito, así que hace falta resetear el singleton del módulo."""
    root = str(tmp_path / "memory")
    monkeypatch.setenv("AETHER_CENTRAL_MEMORY_PATH", root)
    monkeypatch.setenv("AETHER_CENTRAL_MEMORY", "1")
    from core.memory.central_store_v2 import _default as central_store_default, _default_lock
    with _default_lock:
        import core.memory.central_store_v2 as csv2
        csv2._default = None
    yield CentralMemory(root)
    import core.memory.central_store_v2 as csv2
    with _default_lock:
        csv2._default = None


@pytest.fixture
def builder_limpio(monkeypatch):
    """Aísla al context builder del store local (memoria.json real)."""
    from core.memory import context_builder as cb
    monkeypatch.setattr(cb, "obtener_recuerdos", lambda *a, **k: [])
    monkeypatch.setattr(cb, "obtener_turnos_por_tema", lambda *a, **k: [])
    monkeypatch.setattr(cb, "obtener_ultimos_turnos", lambda *a, **k: [])
    return cb


# ══════════════════════════════════════════════════════════════════════
# STORE: memoria de intentos (outcomes)
# ══════════════════════════════════════════════════════════════════════

def test_register_outcome_crea_recuerdo_operativo(mem):
    m = mem.register_outcome("web", {"instruccion": "buscar X"}, "sin_resultados")
    assert m["kind"] == "outcome"
    assert m["namespace"] == "intentos"
    assert m["importance"] == 2
    assert m["data"]["tool"] == "web"
    assert m["data"]["clase"] == "sin_resultados"
    assert m["data"]["count"] == 1
    assert "buscar X" in m["summary"]
    assert "SIN RESULTADOS" in m["summary"]


def test_register_outcome_dedup_por_fingerprint(mem):
    a = mem.register_outcome("web", {"instruccion": "Buscar X"}, "sin_resultados")
    # Mismo intento (args normalizados a minúsculas): NO duplica.
    b = mem.register_outcome("web", {"instruccion": "buscar x"}, "sin_resultados")
    assert b["id"] == a["id"]
    assert b["data"]["count"] == 2
    assert mem.stats()["learnings"] == 1
    # Un intento DISTINTO sí crea su propio outcome.
    c = mem.register_outcome("web", {"instruccion": "buscar Y"}, "sin_resultados")
    assert c["id"] != a["id"]
    assert mem.stats()["learnings"] == 2


def test_outcomes_excluidos_de_search_y_personalidad_por_defecto(mem):
    mem.learn("aprendizaje real del usuario", importance=5)
    mem.register_outcome("web", {"instruccion": "buscar Zeta"}, "sin_resultados")

    # search() por defecto no devuelve outcomes (son operativos).
    res = mem.search("buscar Zeta")
    assert all(r["kind"] != "outcome" for r in res)
    # …salvo que se pidan explícitamente.
    assert len(mem.search("buscar Zeta", include_outcomes=True)) == 1
    # personality_signals() tampoco los cuenta como personalidad.
    assert all("Zeta" not in s for s in mem.personality_signals())


def test_recent_outcomes_filtra_vencidos_y_prioriza_relevantes(mem):
    mem.register_outcome("web", {"instruccion": "precio del oro en Marte"},
                         "sin_resultados")
    mem.register_outcome("fs_read", {"path": "/tmp/inexistente.txt"}, "error")

    res = mem.recent_outcomes(query="buscá el precio del oro")
    assert res[0]["data"]["tool"] == "web"

    # Vencido por TTL (48h): no aparece ni en recent ni en el store tras
    # el barrido de consolidate().
    target = res[0]
    central = mem._load_learning()["memories"][target["id"]]
    central["last_used"] = (date.today() - timedelta(days=5)).isoformat()
    mem._save_learning(central)
    # El vencido desaparece de los recientes; el de fs_read sigue vivo.
    assert all(o["data"]["tool"] != "web" for o in mem.recent_outcomes())
    stats = mem.consolidate()
    assert stats["outcomes_removed"] == 1
    assert mem.get(target["id"]) is None


def test_consolidate_decae_outcomes_por_dia(mem):
    m = mem.register_outcome("web", {"instruccion": "buscar Ayer"}, "sin_resultados")
    antes = m["strength"]
    central = mem._load_learning()["memories"][m["id"]]
    central["last_used"] = (datetime.now().astimezone() - timedelta(hours=36)).isoformat()
    mem._save_learning(central)
    mem.consolidate()
    # Un día sin uso y ya perdió fuerza (decaimiento DIARIO, no semanal).
    assert mem.get(m["id"])["strength"] < antes


# ══════════════════════════════════════════════════════════════════════
# STORE: fixes al sistema de importancia
# ══════════════════════════════════════════════════════════════════════

def test_learn_dedup_mismo_resumen_refuerza(mem):
    a = mem.learn("el usuario prefiere respuestas breves", importance=5)
    b = mem.learn("El usuario  prefiere   respuestas breves", importance=5)
    assert b["id"] == a["id"]
    assert b["uses"] == 1
    assert mem.stats()["learnings"] == 1


def test_learn_resumenes_distintos_no_deduplica(mem):
    mem.learn("uno")
    mem.learn("dos")
    assert mem.stats()["learnings"] == 2


def test_strength_inicial_monotona_con_importancia(mem):
    bajo = mem.learn("dato bajo", importance=1)["strength"]
    ok = mem.learn("dato medio", importance=5)["strength"]
    alto = mem.learn("dato alto", importance=10)["strength"]
    assert 0.15 < bajo < ok < alto <= 1.0


def test_refuerzo_asintotico_no_satura(mem):
    m = mem.learn("dato usado mucho", importance=5)
    for _ in range(30):
        mem.search("dato usado mucho")
    assert mem.get(m["id"])["strength"] < 1.0


# ══════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER: slot [INTENTOS RECIENTES] + sin auto-refuerzo
# ══════════════════════════════════════════════════════════════════════

def test_slot_intentos_recientes_se_inyecta(central_tmp, builder_limpio):
    central_tmp.register_outcome("web", {"instruccion": "precio del oro en Marte"},
                                 "sin_resultados")
    contexto = builder_limpio.construir_contexto_memoria(
        {}, tema="web", orden="buscá el precio del oro en Marte")
    assert "INTENTOS RECIENTES" in contexto
    assert "precio del oro en marte" in contexto.lower()
    assert "NO la reintentes" in contexto


def test_slot_intentos_desactivable_por_env(builder_limpio, monkeypatch):
    monkeypatch.setenv("AETHER_CENTRAL_MEMORY", "0")
    contexto = builder_limpio.construir_contexto_memoria({}, tema="web", orden="x")
    assert "INTENTOS RECIENTES" not in contexto


def test_inyeccion_de_contexto_no_refuerza(central_tmp):
    from core.memory.context_builder import _aprendizajes_centrales
    m = central_tmp.learn("al usuario le gusta el café sin azúcar", importance=5)
    antes = central_tmp.get(m["id"])
    _aprendizajes_centrales("café")
    despues = central_tmp.get(m["id"])
    assert despues["strength"] == antes["strength"]
    assert despues["uses"] == antes["uses"]


# ══════════════════════════════════════════════════════════════════════
# CONSOLIDADOR: el scheduler ahora también consolida la memoria central
# ══════════════════════════════════════════════════════════════════════

def test_consolidador_central_barre_outcomes_vencidos(central_tmp):
    from core.memory.memory_manager import consolidar_memoria_central
    m = central_tmp.register_outcome("web", {"instruccion": "buscar viejo"},
                                     "sin_resultados")
    central = central_tmp._load_learning()["memories"][m["id"]]
    central["last_used"] = (datetime.now().astimezone() - timedelta(days=5)).isoformat()
    central_tmp._save_learning(central)

    stats = consolidar_memoria_central()
    assert stats["outcomes_removed"] == 1
    assert central_tmp.get(m["id"]) is None


# ══════════════════════════════════════════════════════════════════════
# AGENT LOOP: clasificación, guía terminal, registro y tope duro
# ══════════════════════════════════════════════════════════════════════

def test_clasificar_resultado_tool():
    import core.agent.graph_nodes as gn
    assert gn._clasificar_resultado_tool(
        "Sin resultados disponibles para esa consulta, perdoname."
    ) == "sin_resultados"
    assert gn._clasificar_resultado_tool(
        "Imposible conectar con ningún motor de búsqueda, perdoname."
    ) == "sin_resultados"
    assert gn._clasificar_resultado_tool("[ERROR] Llamada inválida") == "error"
    assert gn._clasificar_resultado_tool("salida normal del comando") == "ok"
    assert gn._clasificar_resultado_tool("") == "ok"


def test_fallos_de_tool():
    import core.agent.graph_nodes as gn
    log = [
        {"tool": "web", "cls": "sin_resultados"},
        {"tool": "shell", "cls": "ok"},
        {"tool": "web", "cls": "error"},
        {"tool": "web", "cls": "ok"},
    ]
    assert gn._fallos_de_tool(log, "web") == 2
    assert gn._fallos_de_tool(log, "shell") == 0
    assert gn._fallos_de_tool([], "web") == 0


def _preparar_loop(monkeypatch, fake_llm, fake_node):
    import core.agent.graph_nodes as gn
    from core.agent import tool_registry as tr
    monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
    monkeypatch.setattr(tr, "construir_tools_ollama", lambda: [])
    monkeypatch.setattr(tr, "validar_tool_call", lambda t, a: (True, ""))
    monkeypatch.setattr(tr, "get_node_func", lambda t: fake_node)
    return gn


def _correr_loop(gn, orden, max_rondas=12):
    state = {"orden": orden, "mem": {}, "agent_messages": None,
             "agent_pasos_log": None}
    for _ in range(max_rondas):
        out = gn.node_agent_loop(state)
        state.update(out)
        if out.get("done"):
            break
    return state


def test_resultado_sin_resultados_recibe_guia_y_se_registra(central_tmp, monkeypatch):
    def fake_llm(messages, tools, min_predict=None):
        if fake_llm.ronda == 0:
            fake_llm.ronda = 1
            return {"content": "", "tool_calls": [{"function": {
                "name": "web",
                "arguments": {"instruccion": "buscar el precio del oro en Marte"}}}]}
        return {"content": "no encontré nada en la web", "tool_calls": []}
    fake_llm.ronda = 0

    def fake_node(estado):
        return {"web_results": "Sin resultados disponibles para esa consulta, perdoname."}

    gn = _preparar_loop(monkeypatch, fake_llm, fake_node)
    state = _correr_loop(gn, "buscá el precio del oro en Marte")

    paso = state["agent_pasos_log"][0]
    assert paso["cls"] == "sin_resultados"
    assert "[SISTEMA]" in paso["resultado"]
    assert "resultado válido" in paso["resultado"]
    # El intento quedó persistido para los próximos turnos.
    outcomes = central_tmp.recent_outcomes(query="oro marte")
    assert outcomes and outcomes[0]["data"]["tool"] == "web"
    assert outcomes[0]["data"]["clase"] == "sin_resultados"


def test_resultado_ok_no_registra_ni_agrega_guia(central_tmp, monkeypatch):
    def fake_llm(messages, tools, min_predict=None):
        if fake_llm.ronda == 0:
            fake_llm.ronda = 1
            return {"content": "", "tool_calls": [{"function": {
                "name": "shell", "arguments": {"command": "ls"}}}]}
        return {"content": "listo", "tool_calls": []}
    fake_llm.ronda = 0

    gn = _preparar_loop(monkeypatch, fake_llm,
                        lambda e: {"shell_output": "file1\nfile2"})
    state = _correr_loop(gn, "listá archivos")

    paso = state["agent_pasos_log"][0]
    assert paso["cls"] == "ok"
    assert "[SISTEMA]" not in paso["resultado"]
    assert central_tmp.stats()["outcomes"] == 0


def test_tope_de_fallos_cierra_el_loop_forzado(central_tmp, monkeypatch):
    def fake_llm(messages, tools, min_predict=None):
        fake_llm.n += 1
        return {"content": "", "tool_calls": [{"function": {
            "name": "web",
            "arguments": {"instruccion": f"query distinta {fake_llm.n}"}}}]}

    fake_llm.n = 0

    def fake_node(estado):
        return {"web_results": "Sin resultados disponibles para esa consulta, perdoname."}

    gn = _preparar_loop(monkeypatch, fake_llm, fake_node)
    state = _correr_loop(gn, "buscá cualquier cosa")

    assert state["done"] is True
    assert state["agent_activo"] is False
    assert "no dio resultado" in state["final_response"]
    # 3 ejecuciones fallidas y corte: nunca hubo una 4ª ronda.
    fallos = [p for p in state["agent_pasos_log"]
              if p.get("cls") == "sin_resultados"]
    assert len(fallos) == 3
    assert central_tmp.stats()["outcomes"] == 3


def test_cuarta_llamada_en_la_misma_ronda_se_rechaza(central_tmp, monkeypatch):
    # El modelo emitió 4 calls de web en UNA ronda: la 4ª ya no se ejecuta
    # (tope alcanzado dentro de la misma ronda) y el loop cierra.
    def fake_llm(messages, tools, min_predict=None):
        return {"content": "", "tool_calls": [
            {"function": {"name": "web", "arguments": {"instruccion": f"query {i}"}}}
            for i in range(4)]}

    def fake_node(estado):
        return {"web_results": "Sin resultados disponibles para esa consulta, perdoname."}

    gn = _preparar_loop(monkeypatch, fake_llm, fake_node)
    state = _correr_loop(gn, "buscá varias cosas")

    assert state["done"] is True
    pasos = state["agent_pasos_log"]
    assert len(pasos) == 4
    assert [p["cls"] for p in pasos] == [
        "sin_resultados", "sin_resultados", "sin_resultados", "error"]
    assert "ya se intentó" in pasos[3]["resultado"]


# ══════════════════════════════════════════════════════════════════════
# PROMPTS: "sin resultados" es un resultado válido
# ══════════════════════════════════════════════════════════════════════

def test_prompts_tratan_sin_resultados_como_resultado_valido():
    from core.agent.prompts import (
        construir_backstory,
        construir_prompt_agent_loop,
        construir_task_description,
    )
    loop = construir_prompt_agent_loop("contexto de test")
    assert "sin resultados" in loop.lower()
    assert "tope de intentos fallidos" in loop.lower()

    backstory = construir_backstory("contexto de test")
    assert "a como de lugar" not in backstory.lower()
    assert "persistencia razonable" in backstory.lower()

    task = construir_task_description("buscá la versión de X")
    assert "no reintentes indefinidamente" in task.lower()
