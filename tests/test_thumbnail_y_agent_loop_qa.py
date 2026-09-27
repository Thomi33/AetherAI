"""QA de los dos cambios pedidos para la Web UI:

1. FOTO + PROMPT → THUMBNAIL: la imagen adjunta viaja al modelo como
   thumbnail (b64 JPEG chico), incrustado en el mensaje del usuario del
   grafo cuando el MODELO admite visión (agent loop y text). El caption
   genérico de MODELO_VISION queda como fallback para modelos text-only y
   también consume el thumbnail (ya no la foto completa).

2. AGENT LOOP SIN LÍMITE DE PASOS: el corte automático ("Detuve el
   razonamiento tras 6 pasos…") se eliminó. Quién detiene el loop es el
   usuario: Stop de la Web UI (/api/stop) → request_cancel() →
   InferenceCancelled al inicio del siguiente paso.
"""
from __future__ import annotations

import base64
import io
import shutil
import struct
import zlib

import pytest

import backend.core.attachments as attachments
import core.agent.graph_nodes as gn
import core.agent.tool_registry as tr
from core.agent.streaming import (
    InferenceCancelled,
    request_cancel,
    reset_cancel,
)


def _png_rgb(w: int = 64, h: int = 32) -> bytes:
    """PNG RGB válido sin dependencias (zlib + struct)."""

    def _chunk(tipo: bytes, data: bytes) -> bytes:
        cuerpo = tipo + data
        return (struct.pack(">I", len(data)) + cuerpo
                + struct.pack(">I", zlib.crc32(cuerpo)))

    crudo = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + _chunk(b"IDAT", zlib.compress(crudo))
            + _chunk(b"IEND", b""))


@pytest.fixture
def adjuntos_dir(tmp_path, monkeypatch):
    """Redirige el directorio de adjuntos a tmp_path (patrón de
    test_attachments_qa) y limpia las cachés del módulo."""
    d = tmp_path / "adjuntos"
    d.mkdir()
    monkeypatch.setattr(attachments, "_directorio_adjuntos", lambda: d)
    attachments._THUMB_CACHE.clear()
    attachments._CACHE_MODELO_VISION.clear()
    return d


# ════════════════════════════════════════════════════════════════════════
# 1. THUMBNAILS
# ════════════════════════════════════════════════════════════════════════

class TestGenerarThumbnail:

    def test_cadena_rota_cae_a_bytes_originales(self, tmp_path, monkeypatch):
        """Sin PIL/cv2/ffmpeg: mejor mandar la imagen completa que perder
        el análisis (mismo comportamiento que antes del cambio)."""
        monkeypatch.setattr(attachments, "_thumb_pil", lambda r, m: None)
        monkeypatch.setattr(attachments, "_thumb_cv2", lambda r, m: None)
        monkeypatch.setattr(attachments, "_thumb_ffmpeg", lambda r, m: None)
        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG-fake")
        assert attachments.generar_thumbnail_b64(img) == \
            base64.b64encode(b"\x89PNG-fake").decode()

    def test_cache_por_path_y_mtime(self, tmp_path, monkeypatch):
        llamadas = []

        def fake_pil(ruta, max_lado):
            llamadas.append(ruta)
            return base64.b64encode(b"THUMB").decode()

        monkeypatch.setattr(attachments, "_thumb_pil", fake_pil)
        monkeypatch.setattr(attachments, "_thumb_cv2", lambda r, m: None)
        monkeypatch.setattr(attachments, "_thumb_ffmpeg", lambda r, m: None)
        img = tmp_path / "a.png"
        img.write_bytes(b"img")
        attachments.generar_thumbnail_b64(img)
        attachments.generar_thumbnail_b64(img)
        assert len(llamadas) == 1
        img.write_bytes(b"img-cambiada")  # mtime nueva → regenera
        attachments.generar_thumbnail_b64(img)
        assert len(llamadas) == 2

    def test_pil_genera_jpeg_downscaleado(self, tmp_path):
        Image = pytest.importorskip("PIL.Image", reason="Pillow no instalado")
        img = tmp_path / "grande.png"
        Image.new("RGB", (3000, 1500), (255, 0, 0)).save(img)
        b64 = attachments.generar_thumbnail_b64(img, max_lado=512)
        assert b64, "el thumbnail no se generó"
        raw = base64.b64decode(b64)
        assert raw[:2] == b"\xff\xd8"           # JPEG
        assert len(raw) < img.stat().st_size     # mucho más liviano
        with Image.open(io.BytesIO(raw)) as thumb:
            assert max(thumb.size) <= 512        # downscale respetado

    @pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg no instalado")
    def test_ffmpeg_como_fallback(self, tmp_path, monkeypatch):
        """PIL/cv2 caídos → ffmpeg genera el thumbnail igual."""
        monkeypatch.setattr(attachments, "_thumb_pil", lambda r, m: None)
        monkeypatch.setattr(attachments, "_thumb_cv2", lambda r, m: None)
        img = tmp_path / "f.png"
        img.write_bytes(_png_rgb())
        b64 = attachments.generar_thumbnail_b64(img, max_lado=24)
        assert b64, "ffmpeg no generó el thumbnail"
        assert base64.b64decode(b64)[:2] == b"\xff\xd8"


class TestDescribirImagenUsaThumbnail:

    def test_el_payload_lleva_el_thumbnail(self, tmp_path, monkeypatch):
        """describir_imagen manda el THUMBNAIL, no los bytes completos."""
        import requests

        capturado = {}

        class _Resp:
            status_code = 200

            def json(self):
                return {"response": "un gato naranja"}

        def fake_post(url, json=None, timeout=None):
            capturado["payload"] = json
            return _Resp()

        monkeypatch.setattr(requests, "post", fake_post)
        monkeypatch.setattr(attachments, "generar_thumbnail_b64",
                            lambda ruta, max_lado=1024: "VEVTVE8=")
        img = tmp_path / "foto.png"
        img.write_bytes(b"\x89PNG-original-enorme")
        assert attachments.describir_imagen(img) == "un gato naranja"
        assert capturado["payload"]["images"] == ["VEVTVE8="]


class TestModeloAdmiteImagenes:

    @staticmethod
    def _resp(caps=None, projector=None):
        data = {}
        if caps is not None:
            data["capabilities"] = caps
        if projector is not None:
            data["projector_info"] = projector

        class _Resp:
            status_code = 200

            def json(self):
                return data

        return _Resp()

    def test_capabilities_vision_y_cache(self, monkeypatch):
        import requests
        llamadas = []

        def fake_post(url, json=None, timeout=None):
            llamadas.append(json.get("model"))
            return self._resp(caps=["completion", "vision"])

        monkeypatch.setattr(requests, "post", fake_post)
        assert attachments.modelo_admite_imagenes("con-vision") is True
        assert attachments.modelo_admite_imagenes("con-vision") is True
        assert len(llamadas) == 1  # cacheado por modelo

    def test_projector_info_cuenta_como_vision(self, monkeypatch):
        import requests
        monkeypatch.setattr(
            requests, "post",
            lambda url, json=None, timeout=None: self._resp(
                caps=[], projector={"clip.has_vision_encoder": True}))
        assert attachments.modelo_admite_imagenes("con-projector") is True

    def test_modelo_text_only_da_false(self, monkeypatch):
        import requests
        monkeypatch.setattr(
            requests, "post",
            lambda url, json=None, timeout=None: self._resp(caps=["completion"]))
        assert attachments.modelo_admite_imagenes("text-only-qa") is False

    def test_error_de_ollama_no_cachea_false(self, monkeypatch):
        import requests

        def fake_post(url, json=None, timeout=None):
            if fake_post.intentos == 0:
                fake_post.intentos += 1
                raise ConnectionError("ollama abajo")
            return self._resp(caps=["vision"])

        fake_post.intentos = 0
        monkeypatch.setattr(requests, "post", fake_post)
        assert attachments.modelo_admite_imagenes("retry-qa") is False  # caída
        assert attachments.modelo_admite_imagenes("retry-qa") is True   # recuperó


class TestThumbnailsInline:

    @staticmethod
    def _metas(path):
        return [{"kind": "image", "path": str(path), "name": "f.png",
                 "mime": "image/png"}]

    def test_todo_o_nada(self, tmp_path, monkeypatch):
        img = tmp_path / "f.png"
        img.write_bytes(b"img")
        monkeypatch.setattr(attachments, "modelo_admite_imagenes", lambda m: True)
        monkeypatch.setattr(attachments, "generar_thumbnail_b64",
                            lambda ruta, max_lado=1024: "QQ==")
        assert attachments.thumbnails_inline(self._metas(img), "m") == ["QQ=="]
        # un thumbnail fallido → todo-o-nada: []
        monkeypatch.setattr(attachments, "generar_thumbnail_b64",
                            lambda ruta, max_lado=1024: None)
        assert attachments.thumbnails_inline(self._metas(img), "m") == []
        # modelo sin visión → []
        monkeypatch.setattr(attachments, "modelo_admite_imagenes", lambda m: False)
        assert attachments.thumbnails_inline(self._metas(img), "m") == []

    def test_sin_imagenes_no_consulta_ollama(self, monkeypatch):
        monkeypatch.setattr(attachments, "modelo_admite_imagenes",
                            lambda m: pytest.fail("no debía consultar Ollama"))
        assert attachments.thumbnails_inline(
            [{"kind": "text", "path": "/x", "name": "n"}], "m") == []


class TestComponerOrdenInline:

    @staticmethod
    def _meta():
        return [{"name": "foto.jpg", "kind": "image", "path": "/tmp/f.jpg",
                 "mime": "image/jpeg", "size": 10}]

    def test_inline_marca_el_thumb_adjunto(self):
        orden = attachments.componer_orden(
            "qué se ve?", self._meta(), [], describir=False, inline=True)
        assert "qué se ve?" in orden
        assert "thumbnail adjunto" in orden

    def test_sin_inline_no_lo_menciona(self):
        orden = attachments.componer_orden(
            "mirá", self._meta(), [], describir=False)
        assert "thumbnail adjunto" not in orden


# ════════════════════════════════════════════════════════════════════════
# 2. AGENT LOOP SIN LÍMITE DE PASOS
# ════════════════════════════════════════════════════════════════════════

def _estado_loop(**extra):
    return {"orden": "trabajá", "mem": {}, "agent_messages": None,
            "agent_pasos_log": None, **extra}


class TestAgentLoopSinLimite:

    def test_mas_de_seis_pasos_sigue_hasta_que_cierra_el_modelo(self, monkeypatch):
        """Regresión del pedido: el corte automático a los 6 pasos
        (MAX_AGENT_STEPS) ya no existe. Acá el loop ejecuta 8 pasos y lo
        cierra el MODELO con texto, no el sistema."""
        rondas = []

        def fake_llm(messages, tools, min_predict=None):
            rondas.append(1)
            if len(rondas) <= 8:
                return {"content": "", "tool_calls": [{"function": {
                    "name": "shell",
                    "arguments": {"instruccion": f"paso {len(rondas)}"}}}]}
            return {"content": "listo, terminé", "tool_calls": []}

        def fake_node(estado):
            return {"shell_output": "ok"}

        monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
        monkeypatch.setattr(tr, "construir_tools_ollama", lambda: [])
        monkeypatch.setattr(tr, "validar_tool_call", lambda t, a: (True, ""))
        monkeypatch.setattr(tr, "get_node_func", lambda t: fake_node)

        state = _estado_loop()
        for _ in range(30):  # tope defensivo del TEST, no del sistema
            out = gn.node_agent_loop(state)
            state.update(out)
            if out.get("done"):
                break
        assert state["final_response"] == "listo, terminé"
        assert len(state["agent_pasos_log"]) == 8, (
            "el loop se cortó antes de que el modelo terminara")
        assert "Detuve el razonamiento" not in str(state["final_response"])

    def test_stop_cancela_el_loop(self):
        """El Stop del usuario (request_cancel) corta el loop al inicio del
        siguiente paso, con la misma excepción del streaming de texto."""
        request_cancel()
        try:
            with pytest.raises(InferenceCancelled):
                gn.node_agent_loop(_estado_loop())
        finally:
            reset_cancel()

    def test_tras_reset_el_loop_vuelve_a_correr(self, monkeypatch):
        request_cancel()
        reset_cancel()
        monkeypatch.setattr(gn, "_llm_chat_agente",
                            lambda m, t, min_predict=None: {
                                "content": "hola de nuevo", "tool_calls": []})
        monkeypatch.setattr(tr, "construir_tools_ollama", lambda: [])
        out = gn.node_agent_loop(_estado_loop())
        assert out["final_response"] == "hola de nuevo"
        assert out["done"] is True

    def test_config_sin_max_agent_steps(self):
        """La clave se eliminó por completo de la config."""
        from core.config.config_manager import ConfigManager
        assert "MAX_AGENT_STEPS" not in ConfigManager.VALIDATORS
        assert "MAX_AGENT_STEPS" not in ConfigManager.DEFAULTS


class TestAgentLoopConThumb:

    def test_primer_mensaje_lleva_las_imagenes(self, monkeypatch):
        capturado = {}

        def fake_llm(messages, tools, min_predict=None):
            capturado["messages"] = list(messages)
            return {"content": "veo un gato naranja", "tool_calls": []}

        monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
        monkeypatch.setattr(tr, "construir_tools_ollama", lambda: [])
        out = gn.node_agent_loop(_estado_loop(agent_images=["QQ=="]))
        assert out["done"] is True
        user_msg = capturado["messages"][1]
        assert user_msg["content"] == "trabajá"
        assert user_msg["images"] == ["QQ=="], (
            "el thumbnail no llegó al mensaje del usuario")

    def test_sin_imagenes_no_agrega_la_clave(self, monkeypatch):
        capturado = {}

        def fake_llm(messages, tools, min_predict=None):
            capturado["messages"] = list(messages)
            return {"content": "hola", "tool_calls": []}

        monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
        monkeypatch.setattr(tr, "construir_tools_ollama", lambda: [])
        gn.node_agent_loop(_estado_loop())
        assert "images" not in capturado["messages"][1]


class TestEstadoConImagenes:

    def test_crear_estado_inicial_las_propaga(self):
        from core.agent.graph_state import crear_estado_inicial, _claves_estado
        est = crear_estado_inicial("mirá", {"conversacion": []}, True,
                                    imagenes=["QQ=="])
        assert est["agent_images"] == ["QQ=="]
        assert "agent_images" in _claves_estado()
        # default: sin imágenes → lista vacía (compat TUI/CLI)
        assert crear_estado_inicial(
            "hola", {"conversacion": []})["agent_images"] == []


class TestLlmChatConImagenes:

    def test_imagenes_al_mensaje_del_usuario(self, monkeypatch):
        capturado = {}

        def fake_chat(**kwargs):
            capturado.update(kwargs)
            yield {"message": {"content": "hola"}}

        monkeypatch.setattr(gn.ollama, "chat", fake_chat)
        out = gn._llm_chat(system="s", user="u", imagenes=["QQ=="])
        assert out == "hola"
        msgs = capturado["messages"]
        assert msgs[0] == {"role": "system", "content": "s"}
        assert msgs[1] == {"role": "user", "content": "u", "images": ["QQ=="]}

    def test_sin_imagenes_no_agrega_la_clave(self, monkeypatch):
        capturado = {}

        def fake_chat(**kwargs):
            capturado.update(kwargs)
            yield {"message": {"content": "hola"}}

        monkeypatch.setattr(gn.ollama, "chat", fake_chat)
        gn._llm_chat(system="s", user="u")
        assert "images" not in capturado["messages"][1]


class TestPrepararOrdenConThumb:
    """_preparar_orden devuelve (orden, metas, imagenes_b64)."""

    def test_modelo_con_vision_manda_thumb_y_saltea_el_caption(
            self, adjuntos_dir, monkeypatch):
        pytest.importorskip("fastapi", reason="el backend web usa otro venv")
        from backend.api.routes.chat import _preparar_orden
        img = adjuntos_dir / "20260925-000000_0_foto.png"
        img.write_bytes(_png_rgb())
        monkeypatch.setattr(attachments, "modelo_admite_imagenes", lambda m: True)
        monkeypatch.setattr(attachments, "generar_thumbnail_b64",
                            lambda ruta, max_lado=1024: "QQ==")
        monkeypatch.setattr(attachments, "describir_imagen",
                            lambda ruta, pregunta=None: pytest.fail(
                                "el caption de MODELO_VISION es redundante"))
        orden, metas, imagenes = _preparar_orden({
            "message": "qué se ve en la foto?",
            "attachments": [{"name": "foto.png", "mime": "image/png",
                             "path": str(img)}],
        })
        assert imagenes == ["QQ=="]
        assert "thumbnail adjunto" in orden
        assert "Descripción por visión" not in orden

    def test_modelo_text_only_cae_al_caption(self, adjuntos_dir, monkeypatch):
        pytest.importorskip("fastapi", reason="el backend web usa otro venv")
        from backend.api.routes.chat import _preparar_orden
        img = adjuntos_dir / "20260925-000000_0_foto.png"
        img.write_bytes(_png_rgb())
        monkeypatch.setattr(attachments, "modelo_admite_imagenes", lambda m: False)
        monkeypatch.setattr(attachments, "describir_imagen",
                            lambda ruta, pregunta=None: "un gato")
        orden, metas, imagenes = _preparar_orden({
            "message": "qué se ve en la foto?",
            "attachments": [{"name": "foto.png", "mime": "image/png",
                             "path": str(img)}],
        })
        assert imagenes == []
        assert "Descripción por visión" in orden


class TestEndpointAdjuntosFile:
    """GET /api/attachments/file: sirve los adjuntos guardados para que la
    Web UI muestre el thumbnail en el historial (dataURL → path → nombre)."""

    @staticmethod
    def _client():
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routes import chat as chat_routes
        app = FastAPI()
        app.include_router(chat_routes.router, prefix="/api")
        return TestClient(app)

    def test_thumb_por_path(self, adjuntos_dir):
        pytest.importorskip("fastapi", reason="el backend web usa otro venv")
        client = self._client()
        img = adjuntos_dir / "20260925-000000_0_foto.png"
        img.write_bytes(_png_rgb())
        r = client.get("/api/attachments/file",
                       params={"path": str(img), "thumb": 1})
        assert r.status_code == 200
        assert r.content[:2] == b"\xff\xd8"          # JPEG liviano

    def test_fallback_por_nombre_para_mensajes_viejos(self, adjuntos_dir):
        pytest.importorskip("fastapi", reason="el backend web usa otro venv")
        client = self._client()
        img = adjuntos_dir / "20260925-000000_0_foto.png"
        img.write_bytes(_png_rgb())
        r = client.get("/api/attachments/file",
                       params={"name": "foto.png", "thumb": 1})
        assert r.status_code == 200
        assert r.content[:2] == b"\xff\xd8"

    def test_sin_thumb_sirve_el_original(self, adjuntos_dir):
        pytest.importorskip("fastapi", reason="el backend web usa otro venv")
        client = self._client()
        img = adjuntos_dir / "20260925-000000_0_foto.png"
        img.write_bytes(_png_rgb())
        r = client.get("/api/attachments/file", params={"path": str(img)})
        assert r.status_code == 200
        assert r.content == _png_rgb()

    def test_paths_arbitrarios_y_faltantes_da_404(self, adjuntos_dir):
        pytest.importorskip("fastapi", reason="el backend web usa otro venv")
        client = self._client()
        # Nunca file disclosure: fuera del directorio → 404
        assert client.get("/api/attachments/file",
                          params={"path": "/etc/passwd"}).status_code == 404
        assert client.get("/api/attachments/file",
                          params={"name": "inexistente.png"}).status_code == 404
        assert client.get("/api/attachments/file").status_code == 404
        assert client.get("/api/attachments/file",
                          params={"path": str(adjuntos_dir / "no.png")}).status_code == 404
