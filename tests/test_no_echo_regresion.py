"""
test_no_echo_regresion.py — Regresión del bug "Aether ecoa el mensaje del usuario".

Causa raíz (verificada en vivo, 2026-09-26): _correr_grafo_en_hilo (API y TUI)
registraba el turno del usuario en el store PERSISTENTE *antes* de correr el
grafo. obtener_turnos_por_tema() lee directo de ese store, así que el mensaje
recién llegado entraba al slot CONVERSACIÓN del contexto. El modelo veía su
input dos veces (historial + prompt actual) y devolvía un eco literal del
mensaje del usuario, timestamps incluidos, en vez de procesarlo.

Fix: registrar_turno("usuario", orden) se movió al bloque finally (después de
la inferencia) en backend/core/aether_service.py y tui/engine_bridge.py.

Estos tests reproducen el mecanismo del bug SIN llamar a Ollama:
1. Contexto: si el turno se registra ANTES de construir el contexto, el
   mensaje del usuario aparece en CONVERSACIÓN (mecanismo del eco).
2. Contexto: si NO se registró todavía (comportamiento fixed), el mensaje
   NO aparece.
3. Fuente: _correr_grafo_en_hilo (API y TUI) no llama registrar_turno con
   rol usuario antes de grafo.stream().
"""
from __future__ import annotations

import inspect
import re

import pytest

from core.memory import memoria_store
from core.memory.context_builder import construir_contexto_memoria
from core.memory.memory_manager import (
    inicializar_db,
    normalizar_mem,
    registrar_turno,
)


MSG = ("Quita todas las marcas de tiempo: (0:00) hola (0:10) como estás? "
       "(0:19) probando (0:25) el eco")


def _mem_fresca() -> dict:
    return normalizar_mem({"conversacion": [], "core": {}})


@pytest.fixture(autouse=True)
def _store_limpio(tmp_path, monkeypatch):
    """Aísla el store JSON en tmp_path: los tests no tocan la memoria real."""
    import core.memory.memoria_store as ms
    monkeypatch.setattr(ms, "RUTA_JSON", tmp_path / "memoria.json")
    monkeypatch.setattr(ms, "RUTA_BAK", tmp_path / "memoria.json.bak")
    inicializar_db()
    yield


class TestMecanismoDelEco:
    """El mecanismo exacto que producía el eco, verificado sin LLM."""

    def test_registrar_antes_del_contexto_met_el_mensaje_en_el_historial(self):
        """Comportamiento VIEJO (bug): registrar antes → el mensaje del turno
        actual aparece en el slot CONVERSACIÓN del contexto. Es la dupla
        historial+prompt que hacía que el modelo devuelva el texto tal cual."""
        mem = _mem_fresca()
        registrar_turno(mem, "usuario", MSG, tema="text")
        contexto = construir_contexto_memoria(mem, tema="text")
        assert MSG[:40] in contexto, (
            "Si el turno está registrado, el contexto DEBERÍA incluirlo "
            "(esto reproduce el mecanismo del eco)"
        )

    def test_contexto_sin_registrar_no_incluye_el_mensaje(self):
        """Comportamiento FIXED: mientras la inferencia corre, el turno del
        usuario todavía no está en el store → el contexto no lo incluye →
        el modelo solo ve el mensaje como prompt actual (una sola vez)."""
        mem = _mem_fresca()
        # NO registrar: esto es lo que pasa ahora durante la inferencia.
        contexto = construir_contexto_memoria(mem, tema="text")
        assert MSG[:40] not in contexto, (
            "El mensaje del turno actual no debe estar en el contexto "
            "mientras la inferencia está en curso"
        )
        # Después de la inferencia sí queda registrado (para el PRÓXIMO turno).
        registrar_turno(mem, "usuario", MSG, tema="text")
        contexto_siguiente = construir_contexto_memoria(mem, tema="text")
        assert MSG[:40] in contexto_siguiente, (
            "Al turno siguiente el historial SÍ debe incluir el mensaje "
            "previo (continuidad conversacional)"
        )


class TestRegistroPosteriorALaInferencia:
    """_correr_grafo_en_hilo (API y TUI) debe registrar el turno del usuario
    recién DESPUÉS de que el grafo termina (bloque finally), nunca antes."""

    @staticmethod
    def _cuerpo(ruta: str) -> str:
        with open(ruta, encoding="utf-8") as f:
            return f.read()

    def test_api_registra_usuario_despues_del_grafo(self):
        import backend.core.aether_service as svc
        src = inspect.getsource(svc._correr_grafo_en_hilo)
        # registrar_turno(..."usuario"...) debe estar en el finally, es decir
        # por debajo del bloque "finally:" del cuerpo de la función.
        idx_finally = src.find("finally:")
        assert idx_finally != -1, "el hilo del grafo debe tener bloque finally"
        zona_pre = src[:idx_finally]
        zona_post = src[idx_finally:]
        assert not re.search(
            r'registrar_turno\(\s*_AETHER_MEMORY\s*,\s*"usuario"', zona_pre
        ), (
            "registrar_turno(usuario) NO debe ejecutarse antes del finally: "
            "sucediendo antes del grafo, el context_manager lo inyecta en "
            "CONVERSACIÓN y el modelo ecoa el mensaje del usuario"
        )
        assert re.search(
            r'registrar_turno\(\s*_AETHER_MEMORY\s*,\s*"usuario"', zona_post
        ), "el turno del usuario debe registrarse en el finally (persistencia)"

    def test_tui_registra_usuario_despues_del_grafo(self):
        import tui.engine_bridge as eb
        src = inspect.getsource(eb._correr_grafo_en_hilo)
        idx_finally = src.find("finally:")
        assert idx_finally != -1, "el hilo del grafo debe tener bloque finally"
        zona_pre = src[:idx_finally]
        zona_post = src[idx_finally:]
        assert not re.search(
            r'registrar_turno\(\s*_motor\.mem\s*,\s*"usuario"', zona_pre
        ), "TUI: registrar_turno(usuario) NO debe ir antes del grafo (eco)"
        assert re.search(
            r'registrar_turno\(\s*_motor\.mem\s*,\s*"usuario"', zona_post
        ), "TUI: el turno del usuario debe registrarse en el finally"
