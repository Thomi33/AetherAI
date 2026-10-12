"""Tests de core/config/capabilities.py — fuente única de verdad de la
matriz de capacidades por modelo.

Dos capas (no confundir):
  - capacidades_modelo(): las DECLARADAS por Ollama (/api/show);
  - funciones_disponibles(): las EFECTIVAS de Aether (declaradas +
    entorno + config).

Los tests mockean requests.post: no requieren Ollama corriendo. El
formato de `capabilities` contra la Ollama real instalada (v0.40.0) se
verificó manualmente el 09/10/2026 y está documentado en el docstring
del módulo (representación de thinking: string "thinking" en la lista).
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest
import requests

import backend.core.attachments as attachments
from core.config import capabilities


def _resp(status=200, payload=None):
    class _R:
        status_code = status

        def json(self):
            return payload or {}

    return _R()


class _FakePost:
    """Stub de requests.post que responde según el "model" del payload.

    valores: {modelo: respuesta | lista_de_respuestas_en_orden |
    excepción_a_lanzar}. Registra cada consulta para verificar caches.
    """

    def __init__(self, valores: dict):
        self.valores = {m: (list(v) if isinstance(v, list) else [v])
                        for m, v in valores.items()}
        self.consultas: list[str] = []

    def __call__(self, url, json=None, timeout=None):
        modelo = (json or {}).get("model")
        self.consultas.append(modelo)
        cola = self.valores.get(modelo)
        if not cola:
            raise ConnectionError(f"sin mock para el modelo {modelo!r}")
        resp = cola.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _parchar(monkeypatch, valores: dict) -> _FakePost:
    fake = _FakePost(valores)
    monkeypatch.setattr(requests, "post", fake)
    return fake


@pytest.fixture(autouse=True)
def _cache_limpio():
    """Los tests no se contaminan entre sí (ni con el entorno)."""
    capabilities._CACHE_CAPACIDADES.clear()
    yield
    capabilities._CACHE_CAPACIDADES.clear()


def _deps(monkeypatch, ffmpeg=True, whisper=True, grim=True):
    """Fija el entorno (dependencias del sistema) para la matriz."""
    monkeypatch.setattr(capabilities, "_ffmpeg_disponible", lambda: ffmpeg)
    monkeypatch.setattr(capabilities, "_whisper_disponible", lambda: whisper)
    monkeypatch.setattr(capabilities, "_grim_disponible", lambda: grim)


# Payloads reales verificados contra Ollama v0.40.0 (09/10/2026)
_R1 = {"capabilities": ["tools", "thinking", "completion"]}
_GEMMA4 = {"capabilities": ["tools", "completion", "vision", "audio"]}
_LLAVA = {"capabilities": ["completion", "vision"]}
_MOONDREAM = {"capabilities": ["completion", "vision"]}
_TEXT_ONLY = {"capabilities": ["completion", "tools"]}


# ════════════════════════════════════════════════════════════════════════
# 1. Capacidades DECLARADAS (capacidades_modelo)
# ════════════════════════════════════════════════════════════════════════

class TestCapacidadesModelo:

    def test_formato_lista_ollama_040_thinking(self, monkeypatch):
        """deepseek-r1:14b real → thinking declarado, sin visión."""
        _parchar(monkeypatch, {"deepseek-r1:14b": _resp(payload=_R1)})
        caps = capabilities.capacidades_modelo("deepseek-r1:14b")
        assert caps["thinking"] is True
        assert caps["vision"] is False
        assert caps["desconocido"] is False

    def test_formato_lista_gemma4_real(self, monkeypatch):
        """MODELO actual del usuario: visión+audio, SIN thinking."""
        _parchar(monkeypatch, {"gemma4": _resp(payload=_GEMMA4)})
        caps = capabilities.capacidades_modelo("gemma4")
        assert caps["vision"] is True
        assert caps["audio"] is True
        assert caps["thinking"] is False

    def test_formato_dict_intermedio_con_multimodal(self, monkeypatch):
        """Ollamas intermedias reportaban un dict con `multimodal`."""
        _parchar(monkeypatch, {"viejo:1b": _resp(payload={
            "capabilities": {"completion": True,
                             "multimodal": ["vision"]}})})
        caps = capabilities.capacidades_modelo("viejo:1b")
        assert caps["vision"] is True
        assert caps["completion"] is True
        assert caps["tools"] is False

    def test_projector_fallback_con_caps_vacias(self, monkeypatch):
        """caps=[] + projector CLIP → visión (regresión "con-projector")."""
        _parchar(monkeypatch, {"con-projector": _resp(payload={
            "capabilities": [],
            "projector_info": {"clip.has_vision_encoder": True}})})
        caps = capabilities.capacidades_modelo("con-projector")
        assert caps["vision"] is True
        assert caps["desconocido"] is False

    def test_projector_fallback_sin_capabilities(self, monkeypatch):
        """Respuesta SIN capabilities pero con projector → visión,
        no desconocido."""
        _parchar(monkeypatch, {"solo-projector": _resp(payload={
            "projector_info": {"clip.has_vision_encoder": True}})})
        caps = capabilities.capacidades_modelo("solo-projector")
        assert caps["vision"] is True
        assert caps["desconocido"] is False

    def test_text_only_respuesta_valida_se_cachea(self, monkeypatch):
        fake = _parchar(monkeypatch, {"text-only": [_resp(payload=_TEXT_ONLY)]})
        assert capabilities.capacidades_modelo("text-only")["vision"] is False
        assert capabilities.capacidades_modelo("text-only")["vision"] is False
        assert len(fake.consultas) == 1  # respuesta válida → cacheada

    def test_cache_por_modelo(self, monkeypatch):
        fake = _parchar(monkeypatch, {"a:1b": _resp(payload=_LLAVA),
                                      "b:1b": _resp(payload=_LLAVA)})
        capabilities.capacidades_modelo("a:1b")
        capabilities.capacidades_modelo("a:1b")   # sale del cache
        capabilities.capacidades_modelo("b:1b")
        assert len(fake.consultas) == 2

    def test_error_http_no_cachea(self, monkeypatch):
        """HTTP != 200 → desconocido SIN cachear: reintenta."""
        fake = _parchar(monkeypatch, {"m:1b": [_resp(500, {}),
                                               _resp(payload=_LLAVA)]})
        caps = capabilities.capacidades_modelo("m:1b")
        assert caps["desconocido"] is True
        assert caps["vision"] is False      # sin info fiable
        caps = capabilities.capacidades_modelo("m:1b")
        assert caps["vision"] is True       # reintentó y recuperó
        assert len(fake.consultas) == 2

    def test_excepcion_no_cachea_reintenta(self, monkeypatch):
        """Regresión del contrato "retry-qa": caída → desconocido sin
        cachear; recuperación → datos reales."""
        _parchar(monkeypatch, {"retry-qa": [ConnectionError("ollama abajo"),
                                            _resp(payload=_LLAVA)]})
        caps = capabilities.capacidades_modelo("retry-qa")
        assert caps["desconocido"] is True
        caps = capabilities.capacidades_modelo("retry-qa")
        assert caps["desconocido"] is False
        assert caps["vision"] is True

    def test_modelo_vacio_no_consulta(self, monkeypatch):
        fake = _FakePost({})
        monkeypatch.setattr(requests, "post", fake)
        caps = capabilities.capacidades_modelo("")
        assert caps["desconocido"] is True
        assert fake.consultas == []


# ════════════════════════════════════════════════════════════════════════
# 2. Wrapper de compatibilidad (attachments.modelo_admite_imagenes)
# ════════════════════════════════════════════════════════════════════════

class TestCompatibilidadAttachments:

    def test_alias_del_cache(self):
        """_CACHE_MODELO_VISION es EL MISMO dict del cache real: el
        fixture de test_thumbnail_y_agent_loop_qa (.clear()) sigue
        limpiando el cache de capabilities tras la delegación."""
        assert (attachments._CACHE_MODELO_VISION
                is capabilities._CACHE_CAPACIDADES)

    def test_wrapper_delega(self, monkeypatch):
        fake = _parchar(monkeypatch, {"con-vision": _resp(payload=_LLAVA),
                                      "text-only": _resp(payload=_TEXT_ONLY)})
        assert attachments.modelo_admite_imagenes("con-vision") is True
        assert attachments.modelo_admite_imagenes("text-only") is False
        assert len(fake.consultas) == 2

    def test_sin_info_confiable_da_false(self, monkeypatch):
        _parchar(monkeypatch, {"caido": [ConnectionError("x")]})
        assert attachments.modelo_admite_imagenes("caido") is False


# ════════════════════════════════════════════════════════════════════════
# 3. Reasoning verificable (ajuste #3)
# ════════════════════════════════════════════════════════════════════════

class TestModeloConReasoning:

    def test_thinking_declarado_da_true(self, monkeypatch):
        _parchar(monkeypatch, {"qwen3:4b": _resp(payload={
            "capabilities": ["completion", "tools", "thinking"]})})
        assert capabilities.modelo_con_reasoning("qwen3:4b") is True

    def test_sin_thinking_declarado_da_false_ante_heuristico(self, monkeypatch):
        """REGRESIÓN CLAVE (ajuste #3): Gemma-4-E4B no declara thinking
        → False, aunque el heurístico por nombre diría True ("gemma").
        Las capacidades REALES de Ollama mandan; el heurístico es solo
        fallback cuando no se puede consultar."""
        _parchar(monkeypatch, {"hf.co/X/Gemma-4-E4B:Q6": _resp(payload=_GEMMA4)})
        assert capabilities.modelo_con_reasoning(
            "hf.co/X/Gemma-4-E4B:Q6") is False

    def test_fallback_heuristico_cuando_ollama_caida(self, monkeypatch):
        _parchar(monkeypatch, {"deepseek-r1:14b": [ConnectionError("x")]})
        assert capabilities.modelo_con_reasoning("deepseek-r1:14b") is True

    def test_fallback_heuristico_sin_keywords_da_false(self, monkeypatch):
        _parchar(monkeypatch, {"llama3.1:8b": [ConnectionError("x")]})
        assert capabilities.modelo_con_reasoning("llama3.1:8b") is False

    def test_default_usa_modelo_principal(self, monkeypatch):
        monkeypatch.setattr(capabilities, "_modelo_principal",
                            lambda: "m:9b")
        _parchar(monkeypatch, {"m:9b": _resp(payload=_R1)})
        assert capabilities.modelo_con_reasoning() is True


# ════════════════════════════════════════════════════════════════════════
# 4. Matriz de capacidades EFECTIVAS (funciones_disponibles)
# ════════════════════════════════════════════════════════════════════════

class TestFuncionesDisponibles:

    def test_todo_habilitado(self, monkeypatch):
        """Config real del usuario: Gemma-4-E4B (visión) + moondream
        (MODELO_VISION) + deps completas → todo True."""
        _deps(monkeypatch)
        _parchar(monkeypatch, {"principal": _resp(payload=_GEMMA4),
                               "visionm": _resp(payload=_MOONDREAM)})
        m = capabilities.funciones_disponibles("principal", "visionm")
        assert m == {"imagen_inline": True,
                     "caption_fallback": True,
                     "video_transcripcion": True,
                     "video_frames": True,
                     "vision_pantalla": True}

    def test_principal_text_only_con_modelo_vision(self, monkeypatch):
        """Principal text-only: sin inline ni pantalla, pero los
        captions de MODELO_VISION habilitan los frames de video."""
        _deps(monkeypatch)
        _parchar(monkeypatch, {"principal": _resp(payload=_TEXT_ONLY),
                               "visionm": _resp(payload=_MOONDREAM)})
        m = capabilities.funciones_disponibles("principal", "visionm")
        assert m["imagen_inline"] is False
        assert m["caption_fallback"] is True
        assert m["video_frames"] is True          # por caption_fallback
        assert m["vision_pantalla"] is False      # grim sí, visión no

    def test_ambos_modelos_text_only(self, monkeypatch):
        _deps(monkeypatch)
        _parchar(monkeypatch, {"principal": _resp(payload=_TEXT_ONLY),
                               "visionm": _resp(payload=_TEXT_ONLY)})
        m = capabilities.funciones_disponibles("principal", "visionm")
        assert m["imagen_inline"] is False
        assert m["caption_fallback"] is False
        assert m["video_frames"] is False
        assert m["video_transcripcion"] is True   # audio independiente

    def test_sin_ffmpeg(self, monkeypatch):
        _deps(monkeypatch, ffmpeg=False)
        _parchar(monkeypatch, {"principal": _resp(payload=_GEMMA4)})
        m = capabilities.funciones_disponibles("principal", "principal")
        assert m["imagen_inline"] is True        # imágenes no requieren ffmpeg
        assert m["video_transcripcion"] is False
        assert m["video_frames"] is False

    def test_sin_whisper(self, monkeypatch):
        _deps(monkeypatch, whisper=False)
        _parchar(monkeypatch, {"principal": _resp(payload=_GEMMA4)})
        m = capabilities.funciones_disponibles("principal", "principal")
        assert m["video_transcripcion"] is False
        assert m["video_frames"] is True         # frames no requieren whisper

    def test_sin_grim(self, monkeypatch):
        _deps(monkeypatch, grim=False)
        _parchar(monkeypatch, {"principal": _resp(payload=_GEMMA4)})
        m = capabilities.funciones_disponibles("principal", "principal")
        assert m["vision_pantalla"] is False
        assert m["video_frames"] is True

    def test_mismo_modelo_consulta_una_sola_vez(self, monkeypatch):
        _deps(monkeypatch)
        fake = _parchar(monkeypatch, {"m": [_resp(payload=_GEMMA4)]})
        capabilities.funciones_disponibles("m", "m")
        assert len(fake.consultas) == 1


# ════════════════════════════════════════════════════════════════════════
# 5. Wrapper de compatibilidad (aether_service._modelo_con_reasoning)
# ════════════════════════════════════════════════════════════════════════

class TestCompatibilidadAetherService:

    def test_delega_en_capabilities(self, monkeypatch):
        """El SSE 'start' y la sesión live usan este wrapper: refleja
        lo que decida la fuente única de verdad."""
        from backend.core import aether_service
        monkeypatch.setattr(capabilities, "modelo_con_reasoning",
                            lambda modelo=None: True)
        assert aether_service._modelo_con_reasoning() is True
        monkeypatch.setattr(capabilities, "modelo_con_reasoning",
                            lambda modelo=None: False)
        assert aether_service._modelo_con_reasoning() is False

    def test_cadena_completa_con_caps_reales(self, monkeypatch):
        """Integración aether_service → capabilities → /api/show:
        thinking declarado → True (sin heurístico)."""
        from backend.core import aether_service
        monkeypatch.setattr(capabilities, "_modelo_principal",
                            lambda: "m:9b")
        _parchar(monkeypatch, {"m:9b": _resp(payload=_R1)})
        assert aether_service._modelo_con_reasoning() is True

    def test_cambio_de_comportamiento_gemma_sin_thinking(self, monkeypatch):
        """REGRESIÓN (ajuste #3): antes el heurístico por nombre decía
        True para cualquier modelo con "gemma"; ahora manda Ollama.
        Gemma-4-E4B no declara thinking → False (aunque se llame gemma)."""
        from backend.core import aether_service
        monkeypatch.setattr(capabilities, "_modelo_principal",
                            lambda: "hf.co/X/Gemma-4-E4B:Q6")
        _parchar(monkeypatch, {"hf.co/X/Gemma-4-E4B:Q6": _resp(payload=_GEMMA4)})
        assert aether_service._modelo_con_reasoning() is False

    def test_nunca_revierta(self, monkeypatch):
        """Defensa en profundidad: si capabilities revienta, el turno
        sigue (False) — nunca frenar un turno por esto."""
        from backend.core import aether_service

        def _boom(modelo=None):
            raise RuntimeError("boom")

        monkeypatch.setattr(capabilities, "modelo_con_reasoning", _boom)
        assert aether_service._modelo_con_reasoning() is False
