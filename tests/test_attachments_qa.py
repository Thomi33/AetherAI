"""Tests QA del pipeline de adjuntos del backend web (bug "adjunto vacío").

Cubre la regresión reportada: la Web UI sube los archivos por
/api/attachments/upload y luego los referencia por path en /chat/stream;
el backend NO debe descartarlos por "vino vacío".
"""
import base64

import pytest

import backend.core.attachments as attachments
from backend.core.attachments import (
    _decodificar, adjuntos_ya_guardados, componer_orden,
)


@pytest.fixture
def adjuntos_dir(tmp_path, monkeypatch):
    """Redirige el directorio de adjuntos a tmp_path."""
    d = tmp_path / "adjuntos"
    d.mkdir()
    monkeypatch.setattr(attachments, "_directorio_adjuntos", lambda: d)
    return d


# ── Regresión del bug reportado ──────────────────────────────────────────────

def _preparar_orden():
    """Importa _preparar_orden solo si el stack web (fastapi) está."""
    pytest.importorskip("fastapi", reason="el backend web usa otro venv")
    from backend.api.routes.chat import _preparar_orden as prep
    return prep


def test_meta_de_upload_sin_data_no_se_descarta(adjuntos_dir):
    """Bug: /attachments/upload devuelve metas SIN data; al reenviarlas al
    chat con data=undefined el backend las descartaba como 'vino vacío'."""
    img = adjuntos_dir / "20260916-000000_0_foto.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 100)

    orden, metas = _preparar_orden()({
        "message": "qué es esto",
        "attachments": [{"name": "foto.png", "mime": "image/png",
                         "path": str(img)}],  # sin data: como vienen de upload
    })
    assert len(metas) == 1
    assert metas[0]["kind"] == "image"
    assert "vino vacío" not in orden
    assert str(img) in orden


def test_adjunto_con_data_sigue_funcionando(adjuntos_dir):
    data = base64.b64encode(b"contenido de texto").decode()
    orden, metas = _preparar_orden()({
        "message": "mirá",
        "attachments": [{"name": "nota.txt", "mime": "text/plain",
                         "data": data}],
    })
    assert len(metas) == 1
    assert metas[0]["size"] == len(b"contenido de texto")
    assert "nota.txt" in orden


# ── adjuntos_ya_guardados: validación de paths ───────────────────────────────

def test_ya_guardados_rechaza_path_fuera_del_directorio(adjuntos_dir, tmp_path):
    fuera = tmp_path / "fuera.png"
    fuera.write_bytes(b"x" * 10)
    metas = adjuntos_ya_guardados([{"name": "f.png", "mime": "image/png",
                                    "path": str(fuera)}])
    assert metas == []


def test_ya_guardados_rechaza_inexistente_y_vacio(adjuntos_dir):
    vacio = adjuntos_dir / "vacio.png"
    vacio.write_bytes(b"")
    metas = adjuntos_ya_guardados([
        {"name": "a.png", "mime": "image/png", "path": str(adjuntos_dir / "no.png")},
        {"name": "b.png", "mime": "image/png", "path": str(vacio)},
    ])
    assert metas == []


# ── Decodificación tolerante a data URLs ─────────────────────────────────────

def test_decodificar_acepta_data_url_completa():
    raw = b"imagen falsa"
    data_url = "data:image/png;base64," + base64.b64encode(raw).decode()
    meta, err = _decodificar({"name": "f.png", "mime": "image/png",
                              "data": data_url}, 0)
    assert err is None
    assert meta["bytes"] == raw


def test_decodificar_rechaza_vacio_y_base64_invalido():
    meta, err = _decodificar({"name": "x.png", "data": ""}, 0)
    assert meta is None and "vacío" in err
    meta, err = _decodificar({"name": "x.png", "data": "!!!no-base64!!!"}, 0)
    assert meta is None and "base64" in err


# ── La visión corre una sola vez (describir=False en upload) ─────────────────

def test_componer_orden_sin_describir_no_llama_al_modelo(adjuntos_dir, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("describir_imagen no debió llamarse")
    monkeypatch.setattr(attachments, "describir_imagen", _boom)
    orden = componer_orden("mirá", [{
        "name": "f.png", "mime": "image/png", "kind": "image",
        "path": str(adjuntos_dir / "f.png"), "size": 10,
    }], [], describir=False)
    assert "f.png" in orden
