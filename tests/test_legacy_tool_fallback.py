"""Tests QA del fallback de tool calls en texto legado (agent loop).

Regresión reportada con gemma4:e4b: el modelo escribía

    [vision]
    {"instruccion": "Describe el contenido de la imagen adjunta: ..."}
    <tool_call|>

en el CONTENT en vez de usar tool calling nativo, y la visión (o el
comando) nunca se ejecutaba: el agent loop cerraba con "Respuesta final
tras 0 pasos".
"""
import pytest

from core.agent.graph_nodes import (
    _limpiar_artefactos_tool_call,
    _tool_calls_desde_texto,
)


# ── Reproducción exacta del log del usuario ────────────────────────────────

def test_vision_bloque_abierto_con_json_truncado_al_final():
    contenido = ('[vision]\n{\n  "instruccion": "Describe el contenido de la '
                 'imagen adjunta: /home/thomi/Aether/adjuntos/foto.jpg"\n}'
                 '<tool_call|>')
    calls = _tool_calls_desde_texto(contenido)
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "vision"
    assert "foto.jpg" in fn["arguments"]["instruccion"]


def test_bloque_cerrado_formato_simple():
    calls = _tool_calls_desde_texto("[vision] mirar la pantalla [/vision]")
    assert calls[0]["function"]["name"] == "vision"
    assert calls[0]["function"]["arguments"]["instruccion"] == "mirar la pantalla"


def test_bloque_shell_con_comando():
    calls = _tool_calls_desde_texto("[SHELL] ls -la /tmp [/SHELL]")
    assert calls[0]["function"]["name"] == "shell"
    assert "ls -la" in calls[0]["function"]["arguments"]["instruccion"]


def test_bloques_desconocidos_se_ignoran():
    assert _tool_calls_desde_texto("[pirulo] nada [/pirulo]") == []
    assert _tool_calls_desde_texto("texto común sin tools") == []
    assert _tool_calls_desde_texto("") == []


def test_json_en_bloque_toma_prioridad_sobre_texto_plano():
    body = '{"instruccion": "buscar X", "top": 3}'
    calls = _tool_calls_desde_texto(f"[web] {body} [/web]")
    assert calls[0]["function"]["arguments"]["instruccion"] == "buscar X"


def test_json_roto_en_bloque_cae_a_texto_plano():
    calls = _tool_calls_desde_texto("[vision] {esto no es json} [/vision]")
    assert calls[0]["function"]["name"] == "vision"
    assert "esto no es json" in calls[0]["function"]["arguments"]["instruccion"]


# ── Limpieza de artefactos en la respuesta final ─────────────────────────

def test_limpieza_de_artefactos_tool_call():
    assert _limpiar_artefactos_tool_call("respuesta buena <tool_call|>") == "respuesta buena"
    assert _limpiar_artefactos_tool_call("ok <|function=x|>") == "ok"
    assert _limpiar_artefactos_tool_call("sin artefactos") == "sin artefactos"


def test_limpieza_no_toca_comparaciones_legitimas():
    # Un '<' común en texto no es un artefacto.
    assert _limpiar_artefactos_tool_call("si a < b entonces...") == "si a < b entonces..."


# ── Integración con el agent loop (sin Ollama) ───────────────────────────

def test_agent_loop_ejecuta_tool_escrita_como_texto(monkeypatch):
    """El contenido legado se convierte en tool call real y se ejecuta."""
    import core.agent.graph_nodes as gn
    import core.agent.tool_registry as tr

    ejecutado = {}

    def fake_llm(messages, tools, min_predict=None):
        if not ejecutado:
            return {"content": "[vision] describir foto.png [/vision]",
                    "tool_calls": []}
        return {"content": "La imagen muestra un gato.", "tool_calls": []}

    def fake_node(state):
        ejecutado["args"] = state.get("_tool_args")
        return {"vision_result": "un gato", "final_response": "es un gato"}

    monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
    monkeypatch.setattr(tr, "get_node_func", lambda _tool: fake_node)

    state = {
        "orden": "describí la imagen",
        "mem": {},
        "context_agent_messages": None,
        "agent_messages": None,
        "agent_pasos_log": None,
    }
    out1 = gn.node_agent_loop(dict(state))
    assert ejecutado, "la tool debió ejecutarse pese a venir como texto"
    assert ejecutado["args"]["instruccion"] == "describir foto.png"

    state2 = dict(state)
    state2.update(out1)
    out2 = gn.node_agent_loop(state2)
    assert out2["done"] is True
    assert "gato" in out2["final_response"]


# ── Rescate de tool anidada (shell vision("…") → vision) ───────────────

def test_rescata_tool_anidada_en_shell():
    from core.agent.graph_nodes import _rescatar_llamada_confundida
    tool, args = _rescatar_llamada_confundida(
        "shell", {"instruccion": 'vision("Describe la imagen /tmp/a.jpg")'})
    assert tool == "vision"
    assert "/tmp/a.jpg" in args["instruccion"]


def test_no_rescata_shell_legitimo():
    from core.agent.graph_nodes import _rescatar_llamada_confundida
    tool, args = _rescatar_llamada_confundida(
        "shell", {"instruccion": "ls -la /tmp"})
    assert tool == "shell" and args["instruccion"] == "ls -la /tmp"


def test_no_rescata_tools_que_no_son_shell():
    from core.agent.graph_nodes import _rescatar_llamada_confundida
    tool, args = _rescatar_llamada_confundida(
        "web", {"instruccion": 'vision("x")'})
    assert tool == "web"


# ── Anti-bucle: misma tool + mismos args no se re-ejecuta ──────────────

def test_anti_bucle_no_repite_la_misma_llamada(monkeypatch):
    import core.agent.graph_nodes as gn
    import core.agent.tool_registry as tr

    ejecuciones = []

    def fake_llm(messages, tools, min_predict=None):
        if len([m for m in messages if m.get("role") == "tool"]) >= 3:
            return {"content": "Listo, descrita la imagen.", "tool_calls": []}
        return {"content": '[vision] describir /tmp/foto.jpg [/vision]',
                "tool_calls": []}

    def fake_node(state):
        ejecuciones.append(state.get("_tool_args"))
        return {"vision_result": "un gato"}

    monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
    monkeypatch.setattr(tr, "get_node_func", lambda _tool: fake_node)

    state = {"orden": "describí la imagen", "mem": {},
             "agent_messages": None, "agent_pasos_log": None}
    # Primeras dos pasadas: ejecuta; tercera: anti-bucle frena y pide cerrar.
    for _ in range(3):
        out = gn.node_agent_loop(state)
        state.update(out)

    assert len(ejecuciones) <= 2
    assert any("NO la repitas" in str(p["resultado"])
               for p in state["agent_pasos_log"])


# ── node_vision: camino de archivo (adjuntos) ──────────────────────────

def test_extraer_path_imagen(tmp_path):
    from core.agent.graph_nodes import _extraer_path_imagen
    img = tmp_path / "foto gato.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    orden = f'Describe el contenido de la imagen adjunta: {img}'
    assert _extraer_path_imagen(orden) == str(img)
    assert _extraer_path_imagen("no hay imágenes acá") is None
    assert _extraer_path_imagen("/tmp/inexistente.png") is None


def test_node_vision_describe_archivo_sin_screenshot(tmp_path, monkeypatch):
    import core.agent.graph_nodes as gn
    import backend.core.attachments as att
    img = tmp_path / "foto.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    llamadas = []

    def fake_describir(ruta, pregunta=None):
        llamadas.append(str(ruta))
        return "una foto de un gato"

    monkeypatch.setattr(att, "describir_imagen", fake_describir)
    out = gn.node_vision({"orden": f"Describe la imagen adjunta: {img}"})
    assert llamadas == [str(img)]
    assert "gato" in out["vision_result"]


def test_node_vision_sin_path_sigue_flujo_screenshot(tmp_path, monkeypatch):
    """Sin path de archivo, node_vision intenta el flujo de captura
    (grim mockeado: no debe llamar a describir_imagen)."""
    import core.agent.graph_nodes as gn
    import core.tools.vision as vision_tool
    monkeypatch.setattr(gn.time, "sleep", lambda _s: None)  # salta countdown

    capturas = []

    def fake_ver_pantalla(pregunta, crop=None):
        capturas.append(pregunta)
        return "descripción de la pantalla"

    monkeypatch.setattr(gn, "ver_pantalla", fake_ver_pantalla)
    out = gn.node_vision({"orden": "mirá la pantalla y decime qué hay"})
    assert capturas, "debió intentar capturar la pantalla"
    assert "pantalla" in out["vision_result"]

