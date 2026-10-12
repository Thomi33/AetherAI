"""Tests QA de la regresión "la Web UI no muestra el razonamiento del modelo".

Bug reproducido: el SSE /api/chat/stream tenía una cadena isinstance propia
que omitía ReasoningEvent (y su payload `done` no traía `reasoning`) → el
motor emitía el razonamiento por su canal dedicado, pero moría en el
serializador: la Web UI jamás lo recibía y el botón del bloque de
razonamiento no desplegaba nada (se descartaba al terminar).

Fix verificado acá: _payload_de_evento (chat.py) es la ÚNICA vía de
serialización de eventos, compartida por el SSE (_consumir) y el WS
(ws_routes._iter_payloads). Además el SSE ahora le pasa las imágenes al
grafo — antes llamaba iter_eventos(orden) sin ellas (visión inline rota por
SSE, a diferencia del WS).

Hermético: no toca Ollama ni el grafo (iter_eventos fakedeado con eventos
fijos); usa la app FastAPI mínima del router (patrón de
test_thumbnail_y_agent_loop_qa.py).
"""
import json

import pytest

pytest.importorskip("fastapi", reason="el backend web usa otro venv")

from backend.api.routes.chat import _payload_de_evento
from backend.core.aether_service import (
    AetherService,
    DoneEvent,
    ErrorEvent,
    NodeUpdateEvent,
    ReasoningEvent,
    StdoutLineEvent,
    TokenEvent,
)


def _app_con_chat():
    """App mínima con SOLO el router de chat (sin lifespan ni estáticos)."""
    from fastapi import FastAPI
    from backend.api.routes import chat as chat_routes
    app = FastAPI()
    app.include_router(chat_routes.router, prefix="/api")
    return app


def _parsear_sse(texto: str) -> list:
    """data: {...} → dicts (ignora keep-alives y el [DONE] final)."""
    eventos = []
    for linea in texto.splitlines():
        if not linea.startswith("data: ") or linea == "data: [DONE]":
            continue
        try:
            eventos.append(json.loads(linea[len("data: "):]))
        except json.JSONDecodeError:
            pass
    return eventos


# ── _payload_de_evento: serializa TODOS los tipos de evento ──────────────────

def test_payload_reasoning_viaja_por_canal_propio():
    assert _payload_de_evento(ReasoningEvent(fragmento="pienso ")) == {
        "type": "reasoning", "data": "pienso "}


def test_payload_done_limpia_respuesta_y_trae_razonamiento():
    payload = _payload_de_evento(DoneEvent(
        respuesta="[SHELL]rm -rf /[/SHELL]\nHola!",
        reasoning="razoné antes de responder",
    ))
    assert payload["type"] == "done"
    assert "[SHELL]" not in payload["response"]     # limpieza aplicada
    assert "Hola!" in payload["response"]
    assert payload["reasoning"] == "razoné antes de responder"


def test_payload_done_sin_razonamiento_devuelve_vacio():
    payload = _payload_de_evento(DoneEvent(respuesta="Hola"))
    assert payload["reasoning"] == ""


def test_payload_resto_de_eventos_y_desconocidos():
    assert _payload_de_evento(TokenEvent(fragmento="hola")) == {
        "type": "token", "data": "hola"}
    log = _payload_de_evento(StdoutLineEvent(texto="x" * 5000))
    assert log["type"] == "log" and len(log["data"]) == 2000   # truncado
    assert _payload_de_evento(ErrorEvent(mensaje="boom")) == {
        "type": "error", "message": "boom"}
    nodo = _payload_de_evento(NodeUpdateEvent(nodo="node_text", delta={}))
    assert nodo["type"] == "node" and nodo["node"] == "node_text"
    assert nodo["delta"] == {}
    assert isinstance(nodo["estado"], (str, type(None)))      # estado humano
    assert _payload_de_evento(object()) is None                # desconocido


# ── Regresión end-to-end: el SSE debe streamear el razonamiento ──────────────

def test_sse_incluye_reasoning_y_done_con_razonamiento(monkeypatch):
    """El bug: _consumir descartaba ReasoningEvent (else: continue) y su
    payload done no tenía 'reasoning'. El stream debe llevar ambos."""
    from fastapi.testclient import TestClient

    eventos = [
        ReasoningEvent(fragmento="pienso "),
        ReasoningEvent(fragmento="en voz alta"),
        TokenEvent(fragmento="Hola"),
        DoneEvent(respuesta="[SHELL]x[/SHELL]Hola",
                  reasoning="pienso en voz alta"),
    ]
    monkeypatch.setattr(
        AetherService, "iter_eventos",
        staticmethod(lambda orden, imagenes=None: iter(eventos)))

    client = TestClient(_app_con_chat())
    r = client.post("/api/chat/stream", json={"message": "hola"})
    assert r.status_code == 200
    evs = _parsear_sse(r.text)

    assert evs[0]["type"] == "start"            # handshake primero
    assert "reasoning" in [e["type"] for e in evs]   # ← la regresión
    razonamiento = "".join(e["data"] for e in evs if e["type"] == "reasoning")
    assert razonamiento == "pienso en voz alta"
    done = next(e for e in evs if e["type"] == "done")
    assert done["reasoning"] == "pienso en voz alta"
    assert "[SHELL]" not in done["response"]
    assert "data: [DONE]" in r.text             # cierre del stream


def test_sse_pasa_las_imagenes_al_grafo(monkeypatch):
    """Paridad SSE/WS: el SSE debe pasar los thumbnails (imagenes) al grafo
    — antes llamaba iter_eventos(orden) sin ellos (visión inline rota)."""
    from fastapi.testclient import TestClient

    vistos = {}

    def _fake_iter(orden, imagenes=None):
        vistos["orden"] = orden
        vistos["imagenes"] = imagenes
        return iter([DoneEvent(respuesta="ok", reasoning="")])

    monkeypatch.setattr(AetherService, "iter_eventos",
                         staticmethod(_fake_iter))
    client = TestClient(_app_con_chat())
    r = client.post("/api/chat/stream", json={"message": "hola"})
    assert r.status_code == 200
    # Con el bug el default None quedaba pegado; ahora viaja la lista
    # (vacía cuando no hay adjuntos de imagen).
    assert vistos["imagenes"] == []


def test_ws_incluye_reasoning_y_done_con_razonamiento(monkeypatch):
    """El WS ya serializaba el razonamiento; con la vía compartida no puede
    volver a divergir del SSE."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.routes import ws_routes

    eventos = [
        ReasoningEvent(fragmento="pienso"),
        DoneEvent(respuesta="Hola", reasoning="pienso"),
    ]
    monkeypatch.setattr(
        AetherService, "iter_eventos",
        staticmethod(lambda mensaje, imagenes=None: iter(eventos)))

    app = FastAPI()
    app.include_router(ws_routes.router)
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"message": "hola"})
        tipos, done = [], None
        while True:
            msg = ws.receive_json()
            tipos.append(msg.get("type"))
            if msg.get("type") in ("done", "error"):
                done = msg
                break
        assert "start" in tipos
        assert "reasoning" in tipos
        assert done["reasoning"] == "pienso"
        assert done["response"] == "Hola"
