"""Tests del canal de RAZONAMIENTO separado (backend).

Cubre los requisitos de la UI unificada status+reasoning:

- Separación REAL reasoning/respuesta: la hace el BACKEND en streaming
  (_ThinkingStreamSplitter + campo nativo "thinking" de Ollama). El cliente
  (Web UI) JAMÁS parsea texto para detectar reasoning.
- Reasoning por streaming: ReasoningEvent fragmento a fragmento + completo
  en DoneEvent (por si el cliente pierde fragmentos).
- Reasoning AUSENTE de logs: node_text / node_plan_synthesizer ya no
  imprimen el razonamiento (stdout = canal de logs).
- Transportes (SSE y WS) exponen {"type":"reasoning"} y done.reasoning.
- Respuesta sin reasoning → cero eventos de reasoning (la UI no crea bloque).
- Sin regresiones: el flujo normal de tokens → respuesta queda intacto.

Los marcadores se construyen con chr() para que el archivo sea robusto a
cualquier mangling de tags en tooling/diffs.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

import core.agent.graph_nodes as g
from core.agent.streaming import (
    clear_reasoning_sink,
    clear_token_sink,
    set_reasoning_sink,
    set_token_sink,
)

# Marcadores inline del formato Ornith/QwQ (construidos por partes).
T_OPEN = chr(60) + "think" + chr(62)
T_CLOSE = chr(60) + "/think" + chr(62)


@pytest.fixture
def sinks(monkeypatch):
    """Registra sinks de tokens y reasoning; devuelve (tokens, razonamiento)."""
    tokens: list[str] = []
    razonamiento: list[str] = []
    set_token_sink(tokens.append)
    set_reasoning_sink(razonamiento.append)
    yield tokens, razonamiento
    clear_token_sink()
    clear_reasoning_sink()


def _fake_ollama_stream(monkeypatch, chunks):
    """ollama.chat(stream=True) → generador con los chunks provistos."""
    def _chat(**kwargs):
        def _gen():
            for c in chunks:
                yield c
        return _gen()
    monkeypatch.setattr(g.ollama, "chat", _chat)


# ══════════════════════════════════════════════════════════════════════
# 1. Splitter: separación en streaming, tolerante a chunks
# ══════════════════════════════════════════════════════════════════════

class TestThinkingStreamSplitter:
    def test_formato_completo_en_chunks(self):
        s = g._ThinkingStreamSplitter()
        razon, contenido = [], []
        for tok in [T_OPEN + "Voy a revisar", " el filesystem",
                    " y comparar" + T_CLOSE + "Respuesta", " final"]:
            r, c = s.feed(tok)
            if r:
                razon.append(r)
            if c:
                contenido.append(c)
        assert s.flush() == ("", "")
        assert "".join(razon) == "Voy a revisar el filesystem y comparar"
        assert "".join(contenido) == "Respuesta final"

    def test_marcadores_partidos_entre_chunks(self):
        s = g._ThinkingStreamSplitter()
        assert s.feed("texto " + T_OPEN[:2]) == ("", "texto ")
        assert s.feed(T_OPEN[2:] + "razonando") == ("razonando", "")
        assert s.feed(T_CLOSE[:3]) == ("", "")
        assert s.feed(T_CLOSE[3:] + "respuesta") == ("", "respuesta")

    def test_sin_marcadores_todo_es_respuesta(self):
        s = g._ThinkingStreamSplitter()
        assert s.feed("respuesta directa") == ("", "respuesta directa")
        assert s.feed(" sin razonamiento") == ("", " sin razonamiento")
        assert s.flush() == ("", "")

    def test_stream_cortado_dentro_de_think(self):
        s = g._ThinkingStreamSplitter()
        assert s.feed(T_OPEN + "pensando incompleto...") == ("pensando incompleto...", "")
        assert s.flush() == ("", "")

    def test_carry_de_content_se_recupera_en_flush(self):
        s = g._ThinkingStreamSplitter()
        assert s.feed("respuesta <thi") == ("", "respuesta ")
        assert s.flush() == ("", "<thi")

    def test_falso_prefijo_no_corta_texto(self):
        s = g._ThinkingStreamSplitter()
        assert s.feed("hola <th") == ("", "hola ")
        # El carry se antepone: el texto final se reconstruye EXACTO.
        assert s.feed("e browser es libre") == ("", "<the browser es libre")

    def test_dos_bloques_de_thinking(self):
        s = g._ThinkingStreamSplitter()
        razon, contenido = [], []
        for tok in [T_OPEN + "uno" + T_CLOSE, "respuesta ",
                    T_OPEN + "dos" + T_CLOSE, "final"]:
            r, c = s.feed(tok)
            if r:
                razon.append(r)
            if c:
                contenido.append(c)
        assert "".join(razon) == "unodos"
        assert "".join(contenido) == "respuesta final"


# ══════════════════════════════════════════════════════════════════════
# 2. _llm_chat: reasoning → canal propio; respuesta → tokens limpios
# ══════════════════════════════════════════════════════════════════════

class TestLLMChatCanalSeparado:
    def test_streaming_con_thinking_inline(self, monkeypatch, sinks):
        _, razonamiento = sinks
        tokens: list[str] = []
        colector: list[str] = []
        _fake_ollama_stream(monkeypatch, [
            {"message": {"content": T_OPEN + "Voy a revisar"}},
            {"message": {"content": " el filesystem" + T_CLOSE}},
            {"message": {"content": "Respuesta final"}},
        ])
        salida = g._llm_chat(system="s", user="u",
                             on_token=tokens.append, on_reasoning=colector.append)

        assert salida == "Respuesta final"                       # respuesta limpia
        assert tokens == ["Respuesta final"]                      # respuesta: SOLO content
        assert razonamiento == ["Voy a revisar", " el filesystem"]
        assert colector == razonamiento                           # collector del nodo

    def test_campo_nativo_thinking_de_ollama(self, monkeypatch, sinks):
        _, razonamiento = sinks
        tokens: list[str] = []
        _fake_ollama_stream(monkeypatch, [
            {"message": {"thinking": "razonando nativo", "content": "respuesta"}},
        ])
        salida = g._llm_chat(system="s", user="u", on_token=tokens.append)
        assert salida == "respuesta"
        assert razonamiento == ["razonando nativo"]
        assert tokens == ["respuesta"]

    def test_sin_reasoning_cero_eventos(self, monkeypatch, sinks):
        """Respuesta sin razonamiento → NINGÚN fragmento al canal reasoning."""
        _, razonamiento = sinks
        tokens: list[str] = []
        _fake_ollama_stream(monkeypatch, [
            {"message": {"content": "Hola "}},
            {"message": {"content": "todo bien?"}},
        ])
        salida = g._llm_chat(system="s", user="u", on_token=tokens.append)
        assert salida == "Hola todo bien?"
        assert tokens == ["Hola ", "todo bien?"]
        assert razonamiento == []

    def test_marcador_partido_en_el_stream(self, monkeypatch, sinks):
        _, razonamiento = sinks
        tokens: list[str] = []
        _fake_ollama_stream(monkeypatch, [
            {"message": {"content": T_OPEN[:3]}},
            {"message": {"content": T_OPEN[3:] + "reflexionando"}},
            {"message": {"content": T_CLOSE[:4]}},
            {"message": {"content": T_CLOSE[4:] + "La respuesta"}},
        ])
        salida = g._llm_chat(system="s", user="u", on_token=tokens.append)
        assert salida == "La respuesta"
        assert "".join(razonamiento) == "reflexionando"
        assert tokens == ["La respuesta"]


# ══════════════════════════════════════════════════════════════════════
# 3. Agent loop con STREAMING: respuesta en vivo + tool calls reensambladas
#    (estilo Open WebUI: el reasoning es opcional y NUNCA bloquea el answer)
# ══════════════════════════════════════════════════════════════════════

class TestAgenteLoopReasoning:
    """Casos A-I del spec sobre _llm_chat_agente (streaming)."""

    def _fake_stream(self, monkeypatch, chunks):
        def _chat(**kwargs):
            def _gen():
                for c in chunks:
                    yield c
            return _gen()
        monkeypatch.setattr(g.ollama, "chat", _chat)

    # ── A/G: modelo SIN reasoning → respuesta directa, inmediata ──
    def test_A_sin_reasoning_respuesta_directa_en_vivo(self, monkeypatch, sinks):
        """Gemma-4 no emite thinking: el contenido aparece apenas se genera,
        sin esperar tags ni reasoning. Cero eventos de reasoning."""
        tokens, razonamiento = sinks
        self._fake_stream(monkeypatch, [
            {"message": {"content": "Hola, "}},
            {"message": {"content": "soy Gemma."}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert out == {"content": "Hola, soy Gemma.", "tool_calls": []}
        assert tokens == ["Hola, ", "soy Gemma."]     # EN VIVO, chunk a chunk
        assert razonamiento == []

    # ── B: modelo CON reasoning → [Pensó] + respuesta ──
    def test_B_con_reasoning_canales_separados(self, monkeypatch, sinks):
        tokens, razonamiento = sinks
        self._fake_stream(monkeypatch, [
            {"message": {"content": T_OPEN + "Analizo la pregunta."}},
            {"message": {"content": T_CLOSE + "La respuesta es X."}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert out["content"] == "La respuesta es X."
        assert out["tool_calls"] == []
        assert "".join(razonamiento) == "Analizo la pregunta."
        assert tokens == ["La respuesta es X."]       # answer NO espera reasoning

    # ── C/D: tags divididos entre chunks ──
    def test_C_tag_apertura_dividido(self, monkeypatch, sinks):
        tokens, razonamiento = sinks
        self._fake_stream(monkeypatch, [
            {"message": {"content": "<thi"}},
            {"message": {"content": "nk>razonando</thi"}},
            {"message": {"content": "nk> respuesta"}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert "".join(razonamiento) == "razonando"
        assert out["content"] == " respuesta"

    def test_D_tag_cierre_dividido(self, monkeypatch, sinks):
        tokens, razonamiento = sinks
        self._fake_stream(monkeypatch, [
            {"message": {"content": T_OPEN + "pienso"}},
            {"message": {"content": "algo</thi"}},
            {"message": {"content": "nk>listo"}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert "".join(razonamiento) == "piensoalgo"
        assert out["content"] == "listo"

    # ── E: reasoning ultra corto es válido (sin mínimo de duración) ──
    def test_E_reasoning_ultra_corto(self, monkeypatch, sinks):
        tokens, razonamiento = sinks
        self._fake_stream(monkeypatch, [
            {"message": {"content": T_OPEN + "." + T_CLOSE + "Ok"}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert "".join(razonamiento) == "."
        assert out["content"] == "Ok"

    # ── H: backend con reasoning y content como CAMPOS SEPARADOS ──
    def test_H_campos_separados_nativos(self, monkeypatch, sinks):
        tokens, razonamiento = sinks
        self._fake_stream(monkeypatch, [
            {"message": {"thinking": "razono nativo", "content": "respuesta"}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert razonamiento == ["razono nativo"]
        assert out["content"] == "respuesta"
        assert tokens == ["respuesta"]

    # ── I: stream sin reasoning pero CON answer ──
    def test_I_stream_termina_con_answer_sin_reasoning(self, monkeypatch, sinks):
        tokens, razonamiento = sinks
        self._fake_stream(monkeypatch, [
            {"message": {"content": "texto parcial"}},
            {"message": {"content": " y final"}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert razonamiento == []
        assert out["content"] == "texto parcial y final"


    # ── Tool calls: reensamblado del stream (no se rompe el tool calling) ──

    def test_tool_call_dict_completo_en_un_chunk(self, monkeypatch, sinks):
        self._fake_stream(monkeypatch, [
            {"message": {"content": "Voy a buscar", "tool_calls": [
                {"index": 0, "function": {"name": "web", "arguments": {"instruccion": "buscar"}}},
            ]}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert out["content"] == "Voy a buscar"
        assert out["tool_calls"] == [
            {"function": {"name": "web", "arguments": {"instruccion": "buscar"}}},
        ]

    def test_tool_call_json_partida_entre_chunks(self, monkeypatch, sinks):
        """arguments como STRING JSON cortada por el límite de chunks."""
        self._fake_stream(monkeypatch, [
            {"message": {"tool_calls": [
                {"index": 0, "function": {"name": "fs_read",
                                          "arguments": '{"path": "/tmp/a'}},
            ]}},
            {"message": {"tool_calls": [
                {"index": 0, "function": {"arguments": '.txt"}'}},
            ]}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert out["tool_calls"] == [
            {"function": {"name": "fs_read", "arguments": {"path": "/tmp/a.txt"}}},
        ]

    def test_dos_tool_calls_con_indices(self, monkeypatch, sinks):
        self._fake_stream(monkeypatch, [
            {"message": {"tool_calls": [
                {"index": 0, "function": {"name": "web", "arguments": {"q": "a"}}},
                {"index": 1, "function": {"name": "shell", "arguments": {"c": "b"}}},
            ]}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert [tc["function"]["name"] for tc in out["tool_calls"]] == ["web", "shell"]
        assert out["tool_calls"][1]["function"]["arguments"] == {"c": "b"}

    def test_json_cortada_degrada_a_args_vacios(self, monkeypatch, sinks):
        """JSON incompleta (num_predict cortado) → args {} (criterio legado)."""
        self._fake_stream(monkeypatch, [
            {"message": {"tool_calls": [
                {"index": 0, "function": {"name": "shell",
                                          "arguments": '{"instruccion": "algo'}},
            ]}},
        ])
        out = g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        assert out["tool_calls"][0]["function"]["arguments"] == {}

    def test_cancel_durante_el_stream(self, monkeypatch, sinks):
        """El Stop del usuario sigue cortando la generación a mitad de chunk."""
        from core.agent.streaming import InferenceCancelled, request_cancel, reset_cancel
        reset_cancel()
        try:
            def _chat(**kwargs):
                def _gen():
                    yield {"message": {"content": "uno "}}
                    request_cancel()          # el usuario apretó Stop
                    yield {"message": {"content": "dos "}}
                return _gen()
            monkeypatch.setattr(g.ollama, "chat", _chat)

            with pytest.raises(InferenceCancelled):
                g._llm_chat_agente([{"role": "user", "content": "u"}], [])
        finally:
            reset_cancel()




# ══════════════════════════════════════════════════════════════════════
# 4. Nodos: reasoning en su canal, NUNCA en stdout (logs)
# ══════════════════════════════════════════════════════════════════════

class TestNodosSinReasoningEnLogs:
    def test_node_text_sin_print_de_reasoning(self, monkeypatch, sinks, capsys):
        _fake_ollama_stream(monkeypatch, [
            {"message": {"content": T_OPEN + "Razonamiento del modelo."}},
            {"message": {"content": T_CLOSE + "La respuesta visible."}},
        ])
        state = {"orden": "explicame algo", "mem": {"conversacion": []}}
        update = g.node_text(state)

        assert update["final_response"] == "La respuesta visible."
        assert update["_ornith_reasoning"] == "Razonamiento del modelo."
        capturado = capsys.readouterr()
        # El razonamiento NO va a logs (stdout) — requisito explícito.
        assert "Razonamiento del modelo" not in capturado.out
        assert "Ornith thinking" not in capturado.out

    def test_plan_synthesizer_sin_print_de_reasoning(self, monkeypatch, sinks, capsys):
        _fake_ollama_stream(monkeypatch, [
            {"message": {"content": T_OPEN + "Sintetizando los datos." + T_CLOSE}},
            {"message": {"content": "Resultado sintetizado."}},
        ])
        state = {
            "orden": "busca y resume",
            "mem": {"conversacion": []},
            "plan_pasos": [{"tool": "web"}],
            "plan_resultados": ["datos de la web"],
        }
        update = g.node_plan_synthesizer(state)

        assert update["final_response"] == "Resultado sintetizado."
        assert update["_ornith_reasoning"] == "Sintetizando los datos."
        capturado = capsys.readouterr()
        assert "Sintetizando los datos" not in capturado.out
        assert "Ornith thinking" not in capturado.out

    def test_node_text_sin_reasoning_no_llena_el_campo(self, monkeypatch, sinks):
        _fake_ollama_stream(monkeypatch, [
            {"message": {"content": "Respuesta sin thinking."}},
        ])
        state = {"orden": "hola", "mem": {"conversacion": []}}
        update = g.node_text(state)
        assert update["final_response"] == "Respuesta sin thinking."
        assert update["_ornith_reasoning"] == ""


# ══════════════════════════════════════════════════════════════════════
# 5. Transportes: SSE y WS exponen {"type":"reasoning"} + done.reasoning
# ══════════════════════════════════════════════════════════════════════

class TestTransportePayloads:
    def test_payload_sse_de_cada_evento(self):
        from backend.api.routes.chat import _payload_de_evento
        from backend.core.aether_service import (
            TokenEvent, ReasoningEvent, NodeUpdateEvent, StdoutLineEvent,
            DoneEvent, ErrorEvent,
        )

        assert _payload_de_evento(TokenEvent("a")) == {"type": "token", "data": "a"}
        assert _payload_de_evento(ReasoningEvent("x")) == {"type": "reasoning", "data": "x"}

        nodo = _payload_de_evento(NodeUpdateEvent("planner", {}))
        assert nodo["type"] == "node" and nodo["node"] == "planner"
        assert "estado" in nodo and "delta" in nodo

        assert _payload_de_evento(StdoutLineEvent("log line")) == {"type": "log", "data": "log line"}

        done = _payload_de_evento(DoneEvent("La respuesta.", "El razonamiento."))
        assert done == {"type": "done", "response": "La respuesta.",
                        "reasoning": "El razonamiento."}

        assert _payload_de_evento(ErrorEvent("boom")) == {"type": "error", "message": "boom"}
        assert _payload_de_evento(object()) is None  # evento desconocido → skip

    def test_payload_ws_mismo_contrato(self, monkeypatch):
        from backend.api.routes import ws_routes
        from backend.core.aether_service import (
            AetherService, ReasoningEvent, TokenEvent, DoneEvent,
        )

        def _eventos_fake(orden, imagenes=None):
            yield TokenEvent("resp")
            yield ReasoningEvent("razon")
            yield DoneEvent("resp completa", "razon completa")

        monkeypatch.setattr(AetherService, "iter_eventos", staticmethod(_eventos_fake))
        payloads = list(ws_routes._iter_payloads("orden"))

        assert payloads[0] == {"type": "token", "data": "resp"}
        assert payloads[1] == {"type": "reasoning", "data": "razon"}
        assert payloads[2]["type"] == "done"
        assert payloads[2]["reasoning"] == "razon completa"
        assert payloads[2]["response"] == "resp completa"

    def test_done_sin_reasoning_retrocompatible(self):
        """DoneEvent legado (solo respuesta) → reasoning="" en el payload."""
        from backend.api.routes.chat import _payload_de_evento
        from backend.core.aether_service import DoneEvent
        done = _payload_de_evento(DoneEvent("solo respuesta"))
        assert done["reasoning"] == ""


# ══════════════════════════════════════════════════════════════════════
# 6. Servicio: ReasoningEvent a la cola + acumulado en DoneEvent
# ══════════════════════════════════════════════════════════════════════

class TestServicioReasoning:
    def test_hilo_del_grafo_emite_reasoning_y_lo_acumula(self, monkeypatch):
        import queue as _queue

        import backend.core.aether_service as svc
        import core.agent.graph_builder as gb
        import core.agent.graph_state as gs
        import core.memory.memory_manager as mm
        import core.memory.consolidator as cons
        from core.agent.streaming import emit_reasoning

        monkeypatch.setattr(svc, "_AETHER_MEMORY", {"conversacion": []})
        monkeypatch.setattr(mm, "registrar_turno", lambda *a, **k: None)
        monkeypatch.setattr(cons, "programar_consolidacion", lambda *a, **k: None)
        monkeypatch.setattr(
            gs, "crear_estado_inicial",
            lambda orden, mem, modo_autonomo=True, imagenes=None: {"orden": orden, "mem": mem},
        )

        class GrafoFake:
            def stream(self, estado, stream_mode=None):
                emit_reasoning("Voy a ")
                emit_reasoning("revisar el filesystem.")
                yield {"text": {"final_response": "Listo."}}

        monkeypatch.setattr(gb, "get_graph", lambda: GrafoFake())

        q: "queue.Queue" = _queue.Queue()
        svc._correr_grafo_en_hilo("revisá el sistema", q)
        eventos = []
        while not q.empty():
            eventos.append(q.get())

        fragmentos = [e.fragmento for e in eventos if isinstance(e, svc.ReasoningEvent)]
        assert fragmentos == ["Voy a ", "revisar el filesystem."]

        dones = [e for e in eventos if isinstance(e, svc.DoneEvent)]
        assert len(dones) == 1
        assert dones[0].respuesta == "Listo."
        assert dones[0].reasoning == "Voy a revisar el filesystem."

        # El reasoning NUNCA viaja como log (stdout capturado por el writer).
        logs = [e.texto for e in eventos if isinstance(e, svc.StdoutLineEvent)]
        assert all("Voy a revisar" not in texto for texto in logs)


# ══════════════════════════════════════════════════════════════════════
# 7. Sin regresiones: flujo normal de chat y helpers legados
# ══════════════════════════════════════════════════════════════════════

class TestSinRegresiones:
    def test_parse_ornith_thinking_sigue_separando_posthoc(self):
        """La red de seguridad post-hoc de los nodos sigue intacta."""
        razonamiento, respuesta = g._parse_ornith_thinking(
            T_OPEN + "r" + T_CLOSE + "resp")
        assert razonamiento == "r"
        assert respuesta == "resp"

    def test_parse_sin_marcadores_devuelve_todo(self):
        assert g._parse_ornith_thinking("respuesta común") == ("", "respuesta común")

    def test_llm_chat_llamadas_legadas_son_compatibles(self, monkeypatch, sinks):
        """Firmas viejas (sin on_reasoning) siguen funcionando igual."""
        _fake_ollama_stream(monkeypatch, [{"message": {"content": "ok"}}])
        assert g._llm_chat(system="s", user="u") == "ok"



