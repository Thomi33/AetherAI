"""
QA del bug "ya va como 4 pasos que Aether busca lo mismo" (ley 18331).

Log real: ante "aether, que dice ley numero 18331 uruguay?" el agent
loop ejecutó 4+ pasos de tool=web con instrucciones cada vez más
refinadas, pero todos buscaron/leyeron lo mismo (la portada de IMPO) y
el loop nunca convergió. Causas raíz (fixes F-1..F-5 en
core/agent/graph_nodes.py y core/agent/tool_registry.py):

1. (F-1) La instruccion que el modelo escribe en args['instruccion'] se
   descartaba: la base de cada paso era SIEMPRE la orden original →
   todos los pasos condensaban la misma query.
2. (F-2) node_web leía siempre urls[0] y el modelo no tenía forma de
   pedir una query exacta ni una URL concreta (el schema solo
   declaraba 'instruccion').
3. (F-3) El anti-bucle comparaba args crudos (parafrasear lo evadía) y
   ninguna URL ya leída se deduplicaba dentro del turno.
4. (F-4) Una búsqueda exitosa pero repetida clasificaba 'ok': nunca
   alimentaba el tope duro (_MAX_FALLOS_BUSQUEDA).
5. (F-5) El condensador devolvía queries degeneradas ('aether, que
   dice ley numero 18331 uruguay? [contexto') por contaminación del
   [CONTEXTO DE PASOS PREVIOS] y sin saneo de wake words.

Los tests reproducen el escenario del log SIN red ni Ollama: modelo,
buscador y lector de URLs fakeados; node_agent_loop, node_web,
clasificación y tope duro REALES.
"""
from __future__ import annotations

from types import SimpleNamespace

import core.agent.graph_nodes as gn
from core.agent import tool_registry as tr


_ORDEN_18331 = "aether, que dice ley numero 18331 uruguay?"
_URL_IMPO = "https://www.impo.com.uy/bases/leyes/18331-2008"
_URL_BCU = "https://www.bcu.gub.uy/Ley%2018331.pdf"

_RESULTADOS_1URL = (
    "[SEARXNG]\n"
    "- Título: Ley N° 18331 - IMPO\n"
    f"  URL: {_URL_IMPO}\n"
    "  Resumen: Ley de Protección de Datos Personales, portada.\n"
)

_RESULTADOS_2URLS = (
    _RESULTADOS_1URL
    + f"- Título: PDF Ley 18.331 - BCU\n  URL: {_URL_BCU}\n"
      "  Resumen: texto completo de la ley en PDF.\n"
)

_INSTRUCCIONES_PARAFRASEADAS = [
    "buscar el texto exacto del artículo 18331 del Código Penal uruguayo (ley 18331). "
    "Necesito el contenido completo del artículo.",
    "buscar el artículo 18331 específico en el texto de la ley. "
    "Necesito el número exacto del artículo y su contenido.",
    "buscar en el texto completo de la ley 18331 la frase 'derecho al olvido' "
    "o 'olvido' o 'eliminación de datos'.",
    "buscar el texto completo de la ley 18331 de Uruguay en formato PDF o texto plano.",
]


# ── Helpers (patrón de test_outcome_memory._preparar_loop) ────────────────

def _preparar_loop(monkeypatch, fake_llm, fake_node=None):
    """Prepara el agent loop con el modelo fake.

    fake_node=None → se usa el node_web REAL (tests F-3/F-4: buscar_web
    y leer_url se fakean aparte con _fake_web_tools). fake_node≠None →
    spy/stub del nodo (tests F-1). validar_tool_call queda REAL.
    """
    monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
    monkeypatch.setattr(tr, "construir_tools_ollama", lambda: [])
    if fake_node is not None:
        monkeypatch.setattr(tr, "get_node_func", lambda t: fake_node)
    return gn


def _correr_loop(modulo, orden, max_rondas=12):
    state = {"orden": orden, "mem": {}, "agent_messages": None,
             "agent_pasos_log": None}
    for _ in range(max_rondas):
        out = modulo.node_agent_loop(state)
        state.update(out)
        if out.get("done"):
            break
    return state


def _fake_web_tools(monkeypatch, resultados):
    """Fakea buscar_web/leer_url con espías; devuelve (busquedas, lecturas)."""
    busquedas: list = []
    lecturas: list = []

    def _leer(url):
        lecturas.append(url)
        return f"[CONTENIDO DE {url}]\nTexto principal de la página {url}. Artículo 1 ..."

    monkeypatch.setattr(
        gn, "buscar_web",
        SimpleNamespace(invoke=lambda q: busquedas.append(q) or resultados))
    monkeypatch.setattr(gn, "leer_url", SimpleNamespace(invoke=_leer))
    return busquedas, lecturas


def _llm_web_calls(instrucciones, respuesta_final="listo, respondí con lo que encontré"):
    """Modelo fake que emite calls web parafraseadas y luego cierra."""
    def fake_llm(messages, tools, min_predict=None):
        i = fake_llm.i
        fake_llm.i += 1
        if i < len(instrucciones):
            return {"content": "", "tool_calls": [{"function": {
                "name": "web",
                "arguments": {"instruccion": instrucciones[i]},
            }}]}
        return {"content": respuesta_final, "tool_calls": []}
    fake_llm.i = 0
    return fake_llm


# ══════════════════════════════════════════════════════════════════════
# TC-01 · F-1: la instruccion del modelo es la base del paso
# ══════════════════════════════════════════════════════════════════════

class TestF1InstruccionDelModelo:

    def test_llega_al_nodo_como_orden_base(self, monkeypatch):
        """El nodo recibe la instruccion del modelo, no la orden original
        del usuario (que antes era la base SIEMPRE)."""
        capturas: list = []

        def fake_llm(messages, tools, min_predict=None):
            if fake_llm.ronda == 0:
                fake_llm.ronda = 1
                return {"content": "", "tool_calls": [{"function": {
                    "name": "web",
                    "arguments": {"instruccion": "texto del artículo 25 de la ley 18331"},
                }}]}
            return {"content": "listo", "tool_calls": []}
        fake_llm.ronda = 0

        def fake_node(estado):
            capturas.append(estado.get("orden", ""))
            return {"web_results": "datos"}

        gn_ = _preparar_loop(monkeypatch, fake_llm, fake_node)
        _correr_loop(gn_, _ORDEN_18331)

        assert capturas, "el nodo no llegó a ejecutarse"
        assert capturas[0].startswith("texto del artículo 25 de la ley 18331")
        # la orden original con la wake word NO es la base del paso
        assert "aether" not in capturas[0].lower()

    def test_query_en_args_tambien_es_base(self, monkeypatch):
        """args={'query': ...} (nuevo schema F-2) también define el paso."""
        capturas: list = []

        def fake_llm(messages, tools, min_predict=None):
            if fake_llm.ronda == 0:
                fake_llm.ronda = 1
                return {"content": "", "tool_calls": [{"function": {
                    "name": "web",
                    "arguments": {"query": "habeas data uruguay 18331"},
                }}]}
            return {"content": "listo", "tool_calls": []}
        fake_llm.ronda = 0

        def fake_node(estado):
            capturas.append(estado.get("orden", ""))
            return {"web_results": "datos"}

        gn_ = _preparar_loop(monkeypatch, fake_llm, fake_node)
        _correr_loop(gn_, _ORDEN_18331)

        assert capturas[0].startswith("habeas data uruguay 18331")

    def test_sin_instruccion_cae_a_la_orden_original(self, monkeypatch):
        """Args solo estructurados (fs_read) → base = orden original, y el
        path viaja pegado (comportamiento previo conservado)."""
        capturas: list = []

        def fake_llm(messages, tools, min_predict=None):
            if fake_llm.ronda == 0:
                fake_llm.ronda = 1
                return {"content": "", "tool_calls": [{"function": {
                    "name": "fs_read",
                    "arguments": {"path": "/tmp/datos.txt"},
                }}]}
            return {"content": "listo", "tool_calls": []}
        fake_llm.ronda = 0

        def fake_node(estado):
            capturas.append(estado.get("orden", ""))
            return {"fs_result": "datos"}

        gn_ = _preparar_loop(monkeypatch, fake_llm, fake_node)
        _correr_loop(gn_, "mirá este archivo")

        assert capturas[0].startswith("mirá este archivo")
        assert "[path] /tmp/datos.txt" in capturas[0]


# ══════════════════════════════════════════════════════════════════════
# TC-02 · F-3: parafrasear no relee la misma URL
# ══════════════════════════════════════════════════════════════════════

class TestF3DeduplicacionUrls:

    def test_misma_busqueda_parafraseada_no_relee(self, monkeypatch):
        """El escenario del log: la 2ª búsqueda parafraseada NO relee la
        URL ya leída y devuelve [SIN RESULTADOS NUEVOS], clasificado
        'sin_resultados' (alimenta registro + tope duro + guía)."""
        _, lecturas = _fake_web_tools(monkeypatch, _RESULTADOS_1URL)
        monkeypatch.setattr(gn, "_generar_query_busqueda",
                            lambda o: "ley 18331 uruguay")

        gn_ = _preparar_loop(
            monkeypatch, _llm_web_calls(_INSTRUCCIONES_PARAFRASEADAS[:2]))
        state = _correr_loop(gn_, _ORDEN_18331)

        pasos = state["agent_pasos_log"]
        assert [p["tool"] for p in pasos] == ["web", "web"]
        # la URL se leyó UNA vez: el paso 2 no volvió a leerla
        assert lecturas == [_URL_IMPO]
        assert pasos[0]["cls"] == "ok"
        assert pasos[1]["cls"] == "sin_resultados"
        assert "SIN RESULTADOS NUEVOS" in pasos[1]["resultado"]
        # la guía de cierre viaja con el resultado negativo
        assert "[SISTEMA]" in pasos[1]["resultado"]
        assert "resultado válido" in pasos[1]["resultado"]

    def test_paraphrase_avanza_a_url_nueva(self, monkeypatch):
        """Si los resultados traen una URL NUEVA además de la ya leída,
        el paso la lee (progreso real), y solo agotadas TODAS marca sin
        resultados nuevos."""
        _, lecturas = _fake_web_tools(monkeypatch, _RESULTADOS_2URLS)
        monkeypatch.setattr(gn, "_generar_query_busqueda",
                            lambda o: "ley 18331 uruguay")

        gn_ = _preparar_loop(
            monkeypatch, _llm_web_calls(_INSTRUCCIONES_PARAFRASEADAS[:3]))
        state = _correr_loop(gn_, _ORDEN_18331, max_rondas=4)

        pasos = state["agent_pasos_log"]
        # paso 1 → IMPO, paso 2 → BCU (nueva), paso 3 → todo ya leído
        assert lecturas == [_URL_IMPO, _URL_BCU]
        assert [p["cls"] for p in pasos] == ["ok", "ok", "sin_resultados"]

    def test_urls_ya_leidas_solo_en_agent_loop(self):
        """La dedup de URLs es solo del agent loop (agent_activo); el
        camino plan_executor conserva el comportamiento previo."""
        pasos = [{"tool": "web", "args": {}, "cls": "ok",
                  "resultado": f"[CONTENIDO LEÍDO]\n[CONTENIDO DE {_URL_IMPO}]\ntexto"}]
        assert gn._urls_ya_leidas_del_turno(
            {"agent_activo": True, "agent_pasos_log": pasos}) == {_URL_IMPO}
        # sin agent_activo (plan executor) → sin dedup
        assert gn._urls_ya_leidas_del_turno({"agent_pasos_log": pasos}) == set()
        # paso no-web no aporta URLs
        assert gn._urls_ya_leidas_del_turno(
            {"agent_activo": True,
             "agent_pasos_log": [{"tool": "shell", "cls": "ok",
                                  "resultado": "[CONTENIDO DE x]"}]}) == set()


# ══════════════════════════════════════════════════════════════════════
# TC-03 · F-4: las repeticiones agotan el tope duro y cierran el loop
# ══════════════════════════════════════════════════════════════════════

class TestF4TopeDuroConRepeticiones:

    def test_cuatro_parafrases_ciergan_el_loop(self, monkeypatch):
        """3 búsquedas repetidas clasifican 'sin_resultados' (F-4) y
        agotan el tope: el loop cierra forzado con lo que hay en vez de
        repetirse para siempre. Solo la PRIMERA hace fetch de la URL."""
        _, lecturas = _fake_web_tools(monkeypatch, _RESULTADOS_1URL)
        monkeypatch.setattr(gn, "_generar_query_busqueda",
                            lambda o: "ley 18331 uruguay")

        gn_ = _preparar_loop(
            monkeypatch, _llm_web_calls(_INSTRUCCIONES_PARAFRASEADAS[:4]))
        state = _correr_loop(gn_, _ORDEN_18331, max_rondas=6)

        assert state["done"] is True
        assert "No pude conseguir resultados" in state["final_response"]
        pasos = state["agent_pasos_log"]
        assert len(pasos) == 4
        # paso 1 ok (leyó la URL), pasos 2-4 repetidos sin resultado
        assert [p["cls"] for p in pasos] == ["ok", "sin_resultados",
                                             "sin_resultados", "sin_resultados"]
        assert gn._fallos_de_tool(pasos, "web") == 3
        # solo el paso 1 hizo fetch: los repetidos no gastaron red
        assert lecturas == [_URL_IMPO]


# ══════════════════════════════════════════════════════════════════════
# TC-04 · F-5: saneo del condensador de queries
# ══════════════════════════════════════════════════════════════════════

class TestF5SaneoDeQueries:

    def test_el_condensador_no_ve_el_contexto_previo(self, monkeypatch):
        """El bloque [CONTEXTO DE PASOS PREVIOS] no entra al condensador
        (era la fuente del fragmento '[contexto' en las queries)."""
        capturado: dict = {}

        def fake_llm_chat(system="", user="", min_predict=None, **kw):
            capturado["user"] = user
            return "ley 18331 uruguay habeas data"

        monkeypatch.setattr(gn, "_llm_chat", fake_llm_chat)
        orden = ("texto del artículo 25 de la ley 18331\n"
                 "[CONTEXTO DE PASOS PREVIOS]\n"
                 "  - Paso 1: [SEARXNG] Ley N° 18331 - IMPO ... (relleno)")
        q = gn._generar_query_busqueda(orden)

        assert gn._MARCADOR_CTX_PREVIO not in capturado["user"]
        assert "Paso 1" not in capturado["user"]
        assert q == "ley 18331 uruguay habeas data"

    def test_query_degenerada_cae_al_fallback(self, monkeypatch):
        """El eco 'aether, que dice ... [contexto' se descarta; el
        fallback de keywords mantiene el tema SIN wake word ni corchetes."""
        monkeypatch.setattr(
            gn, "_llm_chat",
            lambda **kw: "aether, que dice ley numero 18331 uruguay? [contexto")
        q = gn._generar_query_busqueda(
            "buscar el texto del artículo 18331 del Código Penal uruguayo")
        assert "[" not in q
        assert "aether" not in q
        assert "18331" in q          # el fallback mantiene el tema real
        assert len(q.split()) >= 2

    def test_wake_word_fuera_del_passthrough_corto(self, monkeypatch):
        """'Aether, ley 18331' (≤3 palabras) pasa directo pero SIN la
        wake word — antes se buscaba literalmente 'Aether, ...'."""

        def _boom(**kw):
            raise AssertionError("una query corta no debería llamar al condensador")

        monkeypatch.setattr(gn, "_llm_chat", _boom)
        assert gn._generar_query_busqueda("Aether, ley 18331") == "ley 18331"

    def test_query_de_una_palabra_rechazada(self, monkeypatch):
        """'18331' sola es demasiado vaga: se rechaza y cae al fallback."""
        monkeypatch.setattr(gn, "_llm_chat", lambda **kw: "18331")
        q = gn._generar_query_busqueda(
            "buscar el contenido del artículo 25 de la ley 18331 uruguay")
        assert q != "18331"
        assert len(q.split()) >= 2
        assert "18331" in q

    def test_sanear_query_llm_casos_borde(self):
        assert gn._sanear_query_llm("aether, que dice ley 18331? [contexto") == ""
        assert gn._sanear_query_llm('"ley 18331 uruguay"') == "ley 18331 uruguay"
        assert gn._sanear_query_llm("aether") == ""            # solo wake word
        assert gn._sanear_query_llm("18331") == ""             # una palabra
        assert gn._sanear_query_llm("") == ""


# ══════════════════════════════════════════════════════════════════════
# TC-05 · F-2: node_web respeta query/url explícitos del modelo
# ══════════════════════════════════════════════════════════════════════

class TestF2ArgsEstructurados:

    def _estado_web(self, tool_args, pasos=None, activo=True,
                    orden="buscar la ley 18331"):
        return {"orden": orden, "_tool_args": tool_args,
                "agent_activo": activo, "agent_pasos_log": pasos or []}

    def test_url_explicita_se_lee_sin_buscar(self, monkeypatch):
        busquedas: list = []
        lecturas: list = []

        def _buscar(q):
            busquedas.append(q)
            raise AssertionError("con url explícita no se debe buscar")

        monkeypatch.setattr(gn, "buscar_web", SimpleNamespace(invoke=_buscar))
        monkeypatch.setattr(
            gn, "leer_url",
            SimpleNamespace(invoke=lambda u: lecturas.append(u)
                            or f"[CONTENIDO DE {u}]\nTexto del PDF de la ley..."))

        out = gn.node_web(self._estado_web({"url": _URL_BCU}))

        assert lecturas == [_URL_BCU]
        assert busquedas == []
        assert f"[CONTENIDO DE {_URL_BCU}]" in out["web_results"]

    def test_query_explicita_no_se_recondensa(self, monkeypatch):
        busquedas: list = []
        monkeypatch.setattr(
            gn, "buscar_web",
            SimpleNamespace(invoke=lambda q: busquedas.append(q)
                            or _RESULTADOS_1URL))
        monkeypatch.setattr(
            gn, "leer_url",
            SimpleNamespace(invoke=lambda u: f"[CONTENIDO DE {u}]\ntexto"))

        def _boom(**kw):
            raise AssertionError("con query explícita no se debe condensar")

        monkeypatch.setattr(gn, "_llm_chat", _boom)
        gn.node_web(self._estado_web({"query": "habeas data uruguay 18331"}))
        assert busquedas == ["habeas data uruguay 18331"]

    def test_url_explicita_ya_leida_no_se_relee(self, monkeypatch):
        lecturas: list = []
        monkeypatch.setattr(
            gn, "leer_url",
            SimpleNamespace(invoke=lambda u: lecturas.append(u) or "no debe llegar"))
        pasos = [{"tool": "web", "args": {}, "cls": "ok",
                  "resultado": f"[CONTENIDO LEÍDO]\n[CONTENIDO DE {_URL_IMPO}]\ntexto"}]

        out = gn.node_web(self._estado_web({"url": _URL_IMPO}, pasos=pasos))

        assert lecturas == []
        assert "SIN RESULTADOS NUEVOS" in out["web_results"]

    def test_sin_tool_args_condensa_como_antes(self, monkeypatch):
        """Regresión (camino plan_executor): sin _tool_args ni
        agent_activo, node_web condensa la orden y lee urls[0] igual
        que antes — incluso si esa URL figura en pasos viejos."""
        busquedas: list = []
        lecturas: list = []
        monkeypatch.setattr(
            gn, "buscar_web",
            SimpleNamespace(invoke=lambda q: busquedas.append(q)
                            or _RESULTADOS_1URL))
        monkeypatch.setattr(
            gn, "leer_url",
            SimpleNamespace(invoke=lambda u: lecturas.append(u)
                            or f"[CONTENIDO DE {u}]\ntexto"))
        monkeypatch.setattr(gn, "_generar_query_busqueda",
                            lambda o: "query condensada")

        state = {"orden": "buscar la ley 18331", "_tool_args": {},
                 "agent_pasos_log": [{"tool": "web", "args": {}, "cls": "ok",
                                      "resultado": f"[CONTENIDO DE {_URL_IMPO}]\n"}]}
        out = gn.node_web(state)

        assert busquedas == ["query condensada"]
        assert lecturas == [_URL_IMPO]     # sin dedup fuera del agent loop
        assert "CONTENIDO LEÍDO" in out["web_results"]

    def test_schema_web_declarado_para_el_modelo(self):
        """El catálogo le dice al modelo que puede mandar query/url, y la
        validación acepta un call solo-url (F-2)."""
        assert tr.TOOL_REGISTRY["web"]["instruccion_requerida"] is False
        assert set(tr.TOOL_PARAMETROS["web"]["properties"]) == {"instruccion", "query", "url"}
        assert tr.TOOL_PARAMETROS["web"]["required"] == []

        ok, motivo = tr.validar_tool_call("web", {"url": _URL_BCU})
        assert ok, motivo
        ok, motivo = tr.validar_tool_call("web", {"query": "habeas data"})
        assert ok, motivo
        ok, motivo = tr.validar_tool_call("web", {"instruccion": "buscar la ley"})
        assert ok, motivo


# ══════════════════════════════════════════════════════════════════════
# Regresión · _construir_orden_paso reenvía 'url' estructurado
# ══════════════════════════════════════════════════════════════════════

def test_construir_orden_paso_reenvia_url():
    orden = gn._construir_orden_paso("base", {"url": _URL_BCU, "path": "/a"}, [])
    assert f"[url] {_URL_BCU}" in orden
    assert "[path] /a" in orden
    # y no duplica la clave si ya está en la base
    orden2 = gn._construir_orden_paso(f"leer {_URL_BCU}", {"url": _URL_BCU}, [])
    assert orden2.count("[url]") == 0
