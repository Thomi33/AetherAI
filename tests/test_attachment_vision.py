"""Tests del pipeline de visión de adjuntos (backend/core/attachments.py).

Regresión verificada contra Ollama real (16/09/2026): moondream:latest
degenera (loop 'มามามา…' o string vacío) cuando el prompt incluye la
pregunta del usuario ("User request: …" / "Answer this question…").
El fix usa siempre prompts de captioning simples + detector de salida
degenerada con un retry y fallback a None.

Los tests mockean requests.post: no requieren Ollama corriendo.
"""

from __future__ import annotations

import base64
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

import requests

from backend.core import attachments




@pytest.fixture()
def imagen_tmp(tmp_path):
    p = tmp_path / "foto.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)  # bytes cualesquiera
    return p


def _resp(status=200, payload=None):
    class _R:
        status_code = status

        def json(self):
            return payload or {}

    return _R()


class _FakeRequests:
    """Stub del módulo requests que registra los prompts enviados."""

    def __init__(self, respuestas: list):
        self._respuestas = list(respuestas)
        self.prompts: list[str] = []

    def post(self, url, json=None, timeout=None):
        self.prompts.append((json or {}).get("prompt", ""))
        if not self._respuestas:
            raise AssertionError("más llamadas de las esperadas")
        return self._respuestas.pop(0)


def _describir(monkeypatch, imagen_tmp, respuestas):
    fake = _FakeRequests(respuestas)
    # describir_imagen hace `import requests` dentro de la función: en la
    # práctica termina usando el módulo global, así que se parcha acá.
    monkeypatch.setattr(requests, "post", fake.post)
    resultado = attachments.describir_imagen(imagen_tmp, pregunta="qué ves?")
    return resultado, fake


LOOP_TAILANDES = "มามารามามารามามารามามารามามารามามารามามารามามารามามารามามารามามา"
DESC_OK = "The image shows a computer screen with a dark chat interface."


class TestSalidaDegenerada:
    @pytest.mark.parametrize("texto", ["", "   ", None, LOOP_TAILANDES,
                                       "la la la la la la la la la la la la la la"])
    def test_degeneradas(self, texto):
        assert attachments._salida_degenerada(texto) is True

    @pytest.mark.parametrize("texto", [
        DESC_OK,
        "Un gato negro duerme sobre un teclado mecánico retroiluminado ✓",
        "Screenshot of a desktop with three windows open: terminal, browser, editor.",
    ])
    def test_sanas(self, texto):
        assert attachments._salida_degenerada(texto) is False


class TestDescribirImagen:
    def test_descripcion_normal(self, monkeypatch, imagen_tmp):
        res, fake = _describir(monkeypatch, imagen_tmp,
                               [_resp(200, {"response": DESC_OK})])
        assert res == DESC_OK
        assert len(fake.prompts) == 1

    def test_prompt_nunca_incluye_pregunta_del_usuario(self, monkeypatch, imagen_tmp):
        """Regresión: sufijar la pregunta degeneraba moondream."""
        _describir(monkeypatch, imagen_tmp, [_resp(200, {"response": DESC_OK})])
        res, fake = _describir(monkeypatch, imagen_tmp,
                               [_resp(200, {"response": DESC_OK})])
        for prompt in fake.prompts:
            assert "User request" not in prompt
            assert "qué ves" not in prompt.lower()

    def test_reintenta_si_la_primera_sale_degenerada(self, monkeypatch, imagen_tmp):
        res, fake = _describir(monkeypatch, imagen_tmp, [
            _resp(200, {"response": LOOP_TAILANDES}),
            _resp(200, {"response": DESC_OK}),
        ])
        assert res == DESC_OK
        assert len(fake.prompts) == 2

    def test_devuelve_none_si_ambas_salen_degeneradas(self, monkeypatch, imagen_tmp):
        res, _ = _describir(monkeypatch, imagen_tmp, [
            _resp(200, {"response": LOOP_TAILANDES}),
            _resp(200, {"response": ""}),
        ])
        assert res is None

    def test_http_error_devuelve_none(self, monkeypatch, imagen_tmp):
        res, _ = _describir(monkeypatch, imagen_tmp, [_resp(500, {})])
        assert res is None

    def test_error_en_payload_200_devuelve_none(self, monkeypatch, imagen_tmp):
        res, _ = _describir(monkeypatch, imagen_tmp,
                            [_resp(200, {"error": "model not found"})])
        assert res is None

    def test_archivo_inexistente_devuelve_none_sin_llamar(self, monkeypatch, tmp_path):
        fake = _FakeRequests([])
        monkeypatch.setattr(requests, "post", fake.post)
        res = attachments.describir_imagen(tmp_path / "no_existe.png")
        assert res is None
        assert fake.prompts == []
