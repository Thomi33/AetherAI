"""
Tests QA de la integración "Web UI resistente a refresh/reinicio del
navegador" (re-attach del stream con sesión live).

Escenario reproducido (sin tocar Ollama ni el grafo):
1. _live_reset crea la sesión live del turno; _publish apila cada evento
   en la cola del consumidor actual Y en el buffer secuenciado de la
   sesión (los tokens ya emitidos no se pierden al caer la conexión).
2. live_snapshot expone el estado (active/done/estado/seq/preview/
   reasoning) y purga por TTL.
3. iter_live_eventos(since) re-emite desde seq > since y sigue en vivo
   hasta done: un cliente que hizo F5 recibe el turno COMPLETO (replay
   del pasado + stream del resto), sin duplicar lo que ya tenía.
4. GET /api/chat/live/status y /api/chat/live/stream: mismo contrato de
   payloads que /api/chat/stream (+ seq), live_start/live_end de marco y
   live_none cuando no hay sesión (backend reiniciado / TTL vencido).
5. El turno terminado PERO bufferizado (done) también se puede re-emitar
   completo: la tab que refrescó después de que terminara el turno recibe
   la respuesta final igual.

Hermético (patrón de test_reasoning_sse_qa.py): la sesión live se arma
por la API interna del módulo; app FastAPI mínima con SOLO el router de
chat (sin lifespan ni estáticos).
"""
import queue
import threading
import time

import pytest

pytest.importorskip("fastapi", reason="el backend web usa otro venv")

import backend.core.aether_service as aether_service
from backend.api.routes.chat import _payload_de_evento
from backend.core.aether_service import (
    AetherService,
    DoneEvent,
    ErrorEvent,
    NodeUpdateEvent,
    ReasoningEvent,
    TokenEvent,
)


@pytest.fixture(autouse=True)
def _live_limpia():
    """La sesión live es global: cada test arranca y termina sin una."""
    aether_service._LIVE_SESSION = None
    yield
    aether_service._LIVE_SESSION = None


def _app_con_chat():
    """App mínima con SOLO el router de chat (sin lifespan ni estáticos)."""
    from fastapi import FastAPI
    from backend.api.routes import chat as chat_routes
    app = FastAPI()
    app.include_router(chat_routes.router, prefix="/api")
    return app


def _parsear_sse(texto: str) -> list:
    """data: {...} → dicts (ignora keep-alives y el [DONE] final)."""
    import json
    eventos = []
    for linea in texto.splitlines():
        if not linea.startswith("data: ") or linea == "data: [DONE]":
            continue
        try:
            eventos.append(json.loads(linea[len("data: "):]))
        except json.JSONDecodeError:
            pass
    return eventos


def _turno_demo(terminado: bool = True) -> None:
    """Simula el hilo del grafo publicando el turno típico de un chat."""
    q = queue.Queue()
    aether_service._live_reset("inf-demo", "haceme un resumen largo", True)
    aether_service._publish(q, ReasoningEvent(fragmento="pienso "))
    aether_service._publish(q, TokenEvent(fragmento="Hola"))
    aether_service._publish(q, TokenEvent(fragmento=" mundo"))
    if terminado:
        aether_service._publish(q, DoneEvent(respuesta="Hola mundo",
                                             reasoning="pienso "))
        aether_service._live_fin("COMPLETED", "inf-demo")


# ── _publish: cola del consumidor + buffer live (a la par) ────────────────────

def test_publish_apila_en_cola_y_en_sesion_live():
    q = queue.Queue()
    aether_service._live_reset("inf-1", "orden demo", True)
    aether_service._publish(q, TokenEvent(fragmento="hola "))
    aether_service._publish(q, TokenEvent(fragmento="mundo"))

    # Cola del consumidor actual: comportamiento de siempre, intacto.
    assert q.get_nowait() == TokenEvent(fragmento="hola ")
    assert q.get_nowait() == TokenEvent(fragmento="mundo")

    # Buffer live secuenciado (la copia para el cliente que se reconecte).
    snap = AetherService.live_snapshot()
    assert snap["active"] is True and snap["done"] is False
    assert snap["seq"] == 2 and snap["eventos"] == 2
    assert snap["estado"] == "RUNNING"
    assert snap["reasoning"] is True
    assert snap["preview"] == "orden demo"


def test_publish_tras_done_no_apila():
    """El buffer de la sesión cerrada queda congelado (turno terminado)."""
    q = queue.Queue()
    _turno_demo(terminado=True)
    aether_service._publish(q, TokenEvent(fragmento="tardío"))
    assert AetherService.live_snapshot()["seq"] == 4  # no llegó a 5


# ── live_snapshot: TTL y defensas ─────────────────────────────────────────────

def test_snapshot_sesion_terminada_sigue_bufferizada():
    _turno_demo(terminado=True)
    snap = AetherService.live_snapshot()
    assert snap["active"] is False and snap["done"] is True
    assert snap["estado"] == "COMPLETED"


def test_snapshot_sin_sesion():
    assert AetherService.live_snapshot() == {
        "active": False, "done": False, "seq": 0, "eventos": 0}


def test_snapshot_purga_sesion_terminada_vencida():
    _turno_demo(terminado=True)
    aether_service._LIVE_SESSION["fin_ts"] = (
        time.time() - (aether_service._LIVE_TTL_SEG + 60))
    assert AetherService.live_snapshot() == {
        "active": False, "done": False, "seq": 0, "eventos": 0}


def test_live_fin_de_otra_inferencia_no_cierra():
    """Defensivo: el finally de un hilo viejo no puede cerrar el turno nuevo."""
    aether_service._live_reset("inf-a", "x", True)
    aether_service._live_fin("COMPLETED", "inf-otra")
    assert AetherService.live_snapshot()["active"] is True


# ── iter_live_eventos: replay + follow en vivo ──────────────────────────────

def test_replay_completo_desde_cero():
    _turno_demo(terminado=True)
    pares = list(AetherService.iter_live_eventos(0))
    assert [s for s, _ in pares] == [1, 2, 3, 4]
    assert [p[1].__class__ for p in pares] == [
        ReasoningEvent, TokenEvent, TokenEvent, DoneEvent]
    assert pares[-1][1].respuesta == "Hola mundo"   # terminal incluido


def test_replay_desde_since_solo_eventos_nuevos():
    """F5 con seq ya vista: el cliente pide solo lo que le falta."""
    _turno_demo(terminado=True)
    assert [s for s, _ in AetherService.iter_live_eventos(2)] == [3, 4]
    assert [s for s, _ in AetherService.iter_live_eventos(4)] == []


def test_replay_sigue_eventos_nuevos_y_termina():
    """El replay de un turno EN CURSO sigue streameando hasta el done."""
    aether_service._live_reset("inf-thread", "orden live", True)
    aether_service._publish(queue.Queue(), TokenEvent(fragmento="a"))

    def _hilo():
        time.sleep(0.25)
        aether_service._publish(queue.Queue(), TokenEvent(fragmento="b"))
        aether_service._live_fin("COMPLETED", "inf-thread")

    threading.Thread(target=_hilo, daemon=True).start()
    eventos = [ev for _, ev in AetherService.iter_live_eventos(0)]
    assert [e.fragmento for e in eventos if isinstance(e, TokenEvent)] == ["a", "b"]


def test_replay_sin_sesion_no_emite_nada():
    assert list(AetherService.iter_live_eventos(0)) == []


# ── Endpoints: /api/chat/live/status y /api/chat/live/stream ──────────────────

def test_live_status_route():
    from fastapi.testclient import TestClient
    _turno_demo(terminado=True)
    r = TestClient(_app_con_chat()).get("/api/chat/live/status")
    assert r.status_code == 200
    st = r.json()
    assert st["active"] is False and st["done"] is True
    assert st["estado"] == "COMPLETED" and st["seq"] == 4
    assert st["reasoning"] is True and st["preview"] == "haceme un resumen largo"


def test_live_stream_replay_completo_con_contrato_de_chat():
    """El re-attach emite el MISMO contrato de payloads que /chat/stream
    (+ seq), enmarcado por live_start/live_end: la burbuja restaurada se
    reconstruye igual que la del stream original."""
    from fastapi.testclient import TestClient
    _turno_demo(terminado=True)

    r = TestClient(_app_con_chat()).get("/api/chat/live/stream?since=0")
    assert r.status_code == 200
    evs = _parsear_sse(r.text)

    assert evs[0]["type"] == "live_start"          # handshake primero
    assert evs[0]["reasoning"] is True             # flag del modelo (bloque eager)
    assert {"type": "reasoning", "data": "pienso ", "seq": 1} in evs
    assert {"type": "token", "data": "Hola", "seq": 2} in evs
    assert {"type": "token", "data": " mundo", "seq": 3} in evs
    done = next(e for e in evs if e["type"] == "done")
    assert done["response"] == "Hola mundo"
    assert done["reasoning"] == "pienso "
    assert done["seq"] == 4
    assert evs[-1]["type"] == "live_end"           # cierre del canal live
    assert "data: [DONE]" in r.text


def test_live_stream_sin_sesion_emite_live_none():
    """Backend reiniciado / TTL vencido: el frontend finaliza con el parcial
    que le quedó en localStorage (nota de interrupción)."""
    from fastapi.testclient import TestClient
    r = TestClient(_app_con_chat()).get("/api/chat/live/stream?since=0")
    assert r.status_code == 200
    assert _parsear_sse(r.text) == [{"type": "live_none"}]
    assert "data: [DONE]" in r.text


def test_live_stream_replay_desde_since_no_duplica():
    from fastapi.testclient import TestClient
    _turno_demo(terminado=True)
    r = TestClient(_app_con_chat()).get("/api/chat/live/stream?since=2")
    evs = _parsear_sse(r.text)
    # live_start.seq_total es metadata del buffer, no un evento: excluirlo.
    assert evs[0]["type"] == "live_start" and evs[0]["seq_total"] == 4
    # Solo lo que el cliente no vio: seq 3 y 4 (token " mundo" + done).
    assert [e["seq"] for e in evs
            if e["type"] not in ("live_start", "live_end")] == [3, 4]


def test_live_stream_replay_de_error_del_turno():
    """Turno que murió con error: el refresh recibe el error del turno,
    no un stream colgado para siempre."""
    from fastapi.testclient import TestClient
    q = queue.Queue()
    aether_service._live_reset("inf-err", "explotá", False)
    aether_service._publish(q, TokenEvent(fragmento="parcial"))
    aether_service._publish(q, ErrorEvent(mensaje="boom"))
    aether_service._live_fin("FAILED", "inf-err")

    r = TestClient(_app_con_chat()).get("/api/chat/live/stream?since=0")
    evs = _parsear_sse(r.text)
    assert {"type": "token", "data": "parcial", "seq": 1} in evs
    assert {"type": "error", "message": "boom", "seq": 2} in evs
    assert evs[-1]["type"] == "live_end"


# ── _payload_de_evento: el contrato del re-attach es el del chat ─────────────

def test_payload_live_usa_la_unica_via_de_serializacion():
    """Paridad con /chat/stream y /ws/chat: el re-attach pasando por otra
    cadena sería la regresión del bug de ReasoningEvent (test_reasoning_sse_qa)."""
    assert _payload_de_evento(DoneEvent(respuesta="x", reasoning="r"))["type"] == "done"
    assert _payload_de_evento(NodeUpdateEvent(nodo="node_text", delta={}))["type"] == "node"

