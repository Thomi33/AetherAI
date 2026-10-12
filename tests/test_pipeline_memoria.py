"""Tests de los fixes al pipeline de memoria (consolidador, context builder
y registro de turnos contra el store JSON). Solo los casos reportados:

1. Consolidador: la salida se toma DESPUÉS del último cierre de thinking
   (falte o no el tag de apertura); think=False en la llamada; salida sin
   'RESUMEN CONSOLIDADO' conserva el resumen anterior; el [RESUMEN PREVIO]
   del prompt es texto (antes se inyectaba el dict entero).
2. Inyección: sin tema, los últimos N turnos por recencia (sin filtro de
   tema), incluyendo turnos de usuario guardados sin tema.
3. Orden y metadatos: el turno del usuario se guarda ANTES que la respuesta
   y con tema + sesion_id en AMBOS turnos (node_finalize).
4. Cursor: ultimo_turno_id avanza al id del último turno consolidado.
5. Prompt: sin perfil psicológico/estado de ánimo; solo hechos verificables,
   decisiones y pendientes; sin duplicados; los turnos [AETHER] no son
   hechos del usuario.
"""
from __future__ import annotations

import pytest

import core.memory.memoria_store as store
import core.memory.memory_manager as mm


@pytest.fixture
def mem_tmp(tmp_path, monkeypatch):
    """Aísla el store en tmp_path (mismo patrón que test_memoria_json)."""
    ruta = tmp_path / "db" / "memoria.json"
    monkeypatch.setattr(store, "RUTA_JSON", ruta)
    monkeypatch.setattr(store, "RUTA_BAK", ruta.with_suffix(".json.bak"))
    monkeypatch.setattr(store, "BASE_AETHER", tmp_path)
    return ruta


class _FakeResp:
    status_code = 200
    text = ""

    def __init__(self, body: str):
        self._body = body

    def json(self):
        return {"response": self._body}


def _fake_ollama(monkeypatch, body: str) -> list[dict]:
    """Patea requests.post; devuelve la lista de payloads capturados."""
    payloads: list[dict] = []

    def _post(url, json=None, timeout=None):
        payloads.append(json or {})
        return _FakeResp(body)

    monkeypatch.setattr("requests.post", _post)
    return payloads


# Marcadores de thinking construidos POR PARTES (misma convención que
# _ThinkingStreamSplitter en core/agent/graph_nodes.py): el tag literal se
# pierde entre capas de renderizado y los tests dejarían de probar el caso.
_THINK_OPEN = "<" + "think" + ">"
_THINK_CLOSE = "<" + "/think" + ">"


# ══════════════════════════════════════════════════════════════════════
# 1) CONSOLIDADOR: thinking inline, validación de la salida y payload
# ══════════════════════════════════════════════════════════════════════

def test_consolidador_descarta_thinking_con_apertura_y_cierre(mem_tmp, monkeypatch):
    mem = mm.cargar_memoria()
    mm.registrar_turno(mem, "usuario", "arranqué el proyecto X", sesion_id="s")
    body = (_THINK_OPEN + " el usuario menciona un proyecto. " + _THINK_CLOSE
            + "RESUMEN CONSOLIDADO:\nProyecto X activo.")
    _fake_ollama(monkeypatch, body)
    assert mm.consolidar_resumen(forzar=True) is not None
    texto = mm.obtener_resumen()["texto"]
    assert texto.startswith("RESUMEN CONSOLIDADO")
    assert "el usuario menciona" not in texto      # el razonamiento NO entra
    assert "Proyecto X activo" in texto


def test_consolidador_funciona_sin_tag_de_apertura(mem_tmp, monkeypatch):
    # El server se comió el tag de apertura: queda razonamiento suelto
    # terminado en el tag de cierre → tomar lo posterior al ÚLTIMO cierre.
    mem = mm.cargar_memoria()
    mm.registrar_turno(mem, "usuario", "dato nuevo", sesion_id="s")
    body = ("analizo los turnos... " + _THINK_CLOSE
            + "RESUMEN CONSOLIDADO:\nSolo hechos verificables.")
    _fake_ollama(monkeypatch, body)
    mm.consolidar_resumen(forzar=True)
    assert mm.obtener_resumen()["texto"].startswith("RESUMEN CONSOLIDADO")


def test_consolidador_output_sin_marca_conserva_resumen_anterior(mem_tmp, monkeypatch):
    mem = mm.cargar_memoria()
    mm.registrar_turno(mem, "usuario", "turno nuevo", sesion_id="s")   # id 1
    mm.guardar_resumen("RESUMEN CONSOLIDADO:\nResumen viejo válido.", ultimo_turno_id=0)
    _fake_ollama(monkeypatch, "El usuario parecía entusiasmado y dijo cosas.")
    assert mm.consolidar_resumen(forzar=True) is None
    resumen = mm.obtener_resumen()
    assert resumen["texto"].startswith("RESUMEN CONSOLIDADO:\nResumen viejo")
    assert resumen["ultimo_turno_id"] == 0         # el cursor tampoco avanzó


def test_consolidador_pide_think_false_y_prompt_previo_es_texto(mem_tmp, monkeypatch):
    mem = mm.cargar_memoria()
    mm.registrar_turno(mem, "usuario", "turno", sesion_id="s")
    mm.guardar_resumen("RESUMEN CONSOLIDADO:\nPrevio.", ultimo_turno_id=0)
    payloads = _fake_ollama(monkeypatch, "RESUMEN CONSOLIDADO:\nNuevo.")
    mm.consolidar_resumen(forzar=True)

    payload = payloads[0]
    assert payload["think"] is False               # sin thinking en la llamada
    prompt = payload["prompt"]
    assert "[RESUMEN PREVIO]" in prompt
    assert "RESUMEN CONSOLIDADO:\nPrevio." in prompt
    assert "ultimo_turno_id" not in prompt         # antes iba el dict entero


# ══════════════════════════════════════════════════════════════════════
# 4) CURSOR: avanza al último turno consolidado y no re-procesa
# ══════════════════════════════════════════════════════════════════════

def test_consolidador_avanza_cursor_al_ultimo_turno(mem_tmp, monkeypatch):
    mem = mm.cargar_memoria()
    mm.registrar_turno(mem, "usuario", "t1", sesion_id="s")
    mm.registrar_turno(mem, "aether", "r1", sesion_id="s")
    mm.registrar_turno(mem, "usuario", "t2", sesion_id="s")
    _fake_ollama(monkeypatch, "RESUMEN CONSOLIDADO:\nSíntesis.")

    assert mm.consolidar_resumen(forzar=True) is not None
    turnos = store.cargar()["conversaciones"]
    assert mm.obtener_resumen()["ultimo_turno_id"] == turnos[-1]["id"]
    assert mm.obtener_turnos_pendientes_de_resumen() == []   # no re-procesa


# ══════════════════════════════════════════════════════════════════════
# 2) INYECCIÓN SIN TEMA: recencia pura sobre el store
# ══════════════════════════════════════════════════════════════════════

def test_contexto_sin_tema_inyecta_ultimos_por_recencia(mem_tmp):
    from core.memory.context_builder import construir_contexto_memoria
    mem = mm.cargar_memoria()
    # Turnos del usuario guardados como se guardaban antes del fix:
    # sin tema ni sesion_id (así los registraban los bridges).
    mm.registrar_turno(mem, "usuario", "pregunta vieja")
    mm.registrar_turno(mem, "aether", "respuesta vieja", sesion_id="s1", tema="web")
    mm.registrar_turno(mem, "usuario", "pregunta nueva")

    contexto = construir_contexto_memoria(mem, tema="", sesion_id="s1")
    assert "[CONVERSACIÓN" in contexto
    # Sin tema: últimos N por recencia, incluyendo los turnos del usuario
    # guardados sin tema (antes quedaban afuera del historial).
    assert "pregunta nueva" in contexto
    assert "pregunta vieja" in contexto


# ══════════════════════════════════════════════════════════════════════
# 2b+3) node_finalize: usuario antes que respuesta, tema/sesión en ambos
# ══════════════════════════════════════════════════════════════════════

def test_finalize_guarda_usuario_antes_con_tema_y_sesion(mem_tmp):
    import core.agent.graph_nodes as gn
    mem = mm.cargar_memoria()
    mem["sesion_id"] = "ses-1"
    state = {
        "mem": mem,
        "orden": "buscá el precio del oro",
        "final_response": "costó 100",
        "sesion_id": "ses-1",
        "tema_actual": "web",
    }

    out = gn.node_finalize(state)

    assert out["done"] is True
    turnos = store.cargar()["conversaciones"]
    assert [t["rol"] for t in turnos] == ["usuario", "aether"]   # orden
    assert turnos[0]["id"] < turnos[1]["id"]                      # cronológico
    assert all(t["sesion_id"] == "ses-1" for t in turnos)        # ambos turnos
    assert all(t["tema"] == "web" for t in turnos)
    assert turnos[0]["texto"] == "buscá el precio del oro"
    assert turnos[1]["texto"] == "costó 100"


# ══════════════════════════════════════════════════════════════════════
# 5) PROMPT del consolidador
# ══════════════════════════════════════════════════════════════════════

def test_prompt_consolidador_hechos_sin_perfil_psicologico():
    p = mm._SYSTEM_PROMPT_CONSOLIDATOR.lower()
    assert mm._MARCA_RESUMEN.lower() in p          # exige la marca de arranque
    assert "psicológic" in p and "ánimo" in p      # prohibe perfil / ánimo
    assert "hechos verificables" in p              # solo hechos verificables
    assert "decisiones" in p and "pendientes" in p
    assert "duplicad" in p                          # sin duplicados
    assert "aether" in p                            # dichos del asistente ≠ hechos
