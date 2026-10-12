"""QA del pipeline de FRAMES DE VIDEO (Paso 2 de la matriz de
capacidades por modelo):

  - _extraer_frames_video(): extracción UNA sola vez, reuso persistente
    (frames junto al video), límites de degradación (duración,
    presupuesto de ffmpeg, tope de config).
  - componer_orden(): la sección de video agrega "Frames analizados"
    con timestamps + captions, o degrada a solo-transcripción.
  - thumbnails_frames_inline(): presupuesto compartido de imágenes
    inline (las imágenes del usuario tienen prioridad).

Unitarios herméticos (mock de subprocess.run; sin Ollama ni ffmpeg
reales) + integración con ffmpeg real (skipif) usando un video
sintético (lavfi testsrc).
"""

from __future__ import annotations

import pathlib
import shutil
import struct
import subprocess
import sys
import time as _time_mod
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

import backend.core.attachments as attachments


def _png_rgb(w: int = 24, h: int = 16) -> bytes:
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


THUMB_FALSO = "VEhVTUI="  # b64 de "THUMB"


@pytest.fixture
def adjuntos_dir(tmp_path, monkeypatch):
    """Adjuntos en tmp_path (patrón del repo) + thumbnail determinista:
    el mock de subprocess solo debe ver ffprobe/ffmpeg de EXTRACCIÓN."""
    d = tmp_path / "adjuntos"
    d.mkdir()
    monkeypatch.setattr(attachments, "_directorio_adjuntos", lambda: d)
    attachments._THUMB_CACHE.clear()
    attachments._CACHE_MODELO_VISION.clear()
    monkeypatch.setattr(attachments, "_thumb_pil",
                        lambda r, m: THUMB_FALSO)
    monkeypatch.setattr(attachments, "_thumb_cv2", lambda r, m: None)
    monkeypatch.setattr(attachments, "_thumb_ffmpeg", lambda r, m: None)
    return d


class _FakeRun:
    """Stub de subprocess.run: 'ffprobe' devuelve duración fija;
    'ffmpeg' (extracción de frames) escribe un JPEG válido o falla N
    veces. Registra cada comando para verificar reuso/llamadas."""

    def __init__(self, duracion: float = 100.0, fallos_ffmpeg: int = 0):
        self.duracion = duracion
        self.fallos = fallos_ffmpeg
        self.comandos: list[list[str]] = []
        self.extracciones = 0

    def __call__(self, cmd, **kwargs):
        self.comandos.append(list(cmd))
        if cmd[0] == "ffprobe":

            class _R:
                returncode = 0
                stdout = f"{self.duracion}"

            return _R()
        if cmd[0] == "ffmpeg":
            if self.fallos > 0:
                self.fallos -= 1

                class _Fail:
                    returncode = 1
                    stdout = b""
                    stderr = b"boom"

                return _Fail()
            pathlib.Path(cmd[-1]).write_bytes(_png_rgb())
            self.extracciones += 1

            class _Ok:
                returncode = 0

            return _Ok()
        raise AssertionError(f"comando inesperado: {cmd[0]}")


def _video(adjuntos_dir, nombre: str = "clip.mp4") -> pathlib.Path:
    """Archivo de video falso (bytes cualquiera): _extraer_frames_video
    está mockeado en los tests de componer_orden; en los de extracción
    ffprobe/ffmpeg vienen del _FakeRun."""
    p = adjuntos_dir / nombre
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 128)
    return p


def _meta_video(path: pathlib.Path, texto: str | None = "hola mundo") -> dict:
    return {"name": path.name, "kind": "video", "path": str(path),
            "mime": "video/mp4", "size": 10, "texto_extraido": texto}


# ════════════════════════════════════════════════════════════════════════
# 1. Extracción de frames
# ════════════════════════════════════════════════════════════════════════

class TestFormatoMMSS:
    def test_formatos(self):
        assert attachments._formato_mmss(12.5) == "00:12"
        assert attachments._formato_mmss(62.5) == "01:02"
        assert attachments._formato_mmss(3720) == "01:02:00"

    def test_config_tope_y_default(self, monkeypatch):
        class _Cfg:
            def __init__(self, valor):
                self.valor = valor

            def get(self, key, default=None):
                return self.valor if key == "VIDEO_FRAMES" else default

        import core.config.config_manager as cm
        monkeypatch.setattr(cm, "get_config_manager", lambda: _Cfg(99))
        assert attachments._video_frames_n() == attachments._VIDEO_FRAMES_TOPE
        monkeypatch.setattr(cm, "get_config_manager", lambda: _Cfg(0))
        assert attachments._video_frames_n() == 1  # mínimo 1
        monkeypatch.setattr(cm, "get_config_manager",
                            lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert attachments._video_frames_n() == attachments._VIDEO_FRAMES


class TestExtraerFrames:
    def test_extrae_n_equiespaciados_y_persiste(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        fake = _FakeRun(duracion=100.0)
        monkeypatch.setattr(subprocess, "run", fake)
        frames = attachments._extraer_frames_video(video, n=4)
        # instantes: 100 * (i + 0.5) / 4 → 12.5, 37.5, 62.5, 87.5
        assert [round(f["t"], 3) for f in frames] == [12.5, 37.5, 62.5, 87.5]
        assert fake.extracciones == 4
        # Persisten junto al video en <video>.frames/
        dframes = video.with_name(video.name + ".frames")
        for i in range(4):
            f = dframes / f"frame_{i:02d}.jpg"
            assert f.is_file(), f"frame {i} no persistió"
        assert all(f["thumb"] == THUMB_FALSO for f in frames)
        # El comando ffmpeg recibió -ss con el instante correcto
        ss = [c[c.index("-ss") + 1] for c in fake.comandos if c[0] == "ffmpeg"]
        assert ss == ["12.500", "37.500", "62.500", "87.500"]

    def test_reusa_frames_sin_reextraer(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        fake = _FakeRun(duracion=100.0)
        monkeypatch.setattr(subprocess, "run", fake)
        frames_1 = attachments._extraer_frames_video(video, n=4)
        frames_2 = attachments._extraer_frames_video(video, n=4)
        assert fake.extracciones == 4          # UNA sola extracción por frame
        assert [f["t"] for f in frames_2] == [f["t"] for f in frames_1]

    def test_regenera_si_el_video_cambio(self, adjuntos_dir, monkeypatch):
        import os
        video = _video(adjuntos_dir)
        fake = _FakeRun(duracion=100.0)
        monkeypatch.setattr(subprocess, "run", fake)
        attachments._extraer_frames_video(video, n=4)
        # El video cambia (mtime más nuevo que los frames) → regenera
        os.utime(video, (_time_mod.time() + 20,) * 2)
        attachments._extraer_frames_video(video, n=4)
        assert fake.extracciones == 8

    def test_ffprobe_falla_degrada(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)

        def fake_run(cmd, **kw):
            class _R:
                returncode = 1
                stdout = ""
            return _R()

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert attachments._extraer_frames_video(video, n=4) == []

    def test_video_inexistente(self, adjuntos_dir, monkeypatch):
        fake = _FakeRun()
        monkeypatch.setattr(subprocess, "run", fake)
        assert attachments._extraer_frames_video(
            adjuntos_dir / "no_hay.mp4", n=4) == []
        assert fake.comandos == []

    def test_duracion_excesiva_sin_frames(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        fake = _FakeRun(duracion=5 * 3600)
        monkeypatch.setattr(subprocess, "run", fake)
        assert attachments._extraer_frames_video(video, n=4) == []
        # solo ffprobe corrió
        assert [c[0] for c in fake.comandos] == ["ffprobe"]

    def test_frame_fallido_se_salta(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        fake = _FakeRun(duracion=100.0, fallos_ffmpeg=2)
        monkeypatch.setattr(subprocess, "run", fake)
        frames = attachments._extraer_frames_video(video, n=4)
        assert len(frames) == 2                 # degradación parcial
        assert fake.extracciones == 4 - 2       # 2 exitosas... 4 - 2 fallas

    def test_presupuesto_de_ffmpeg_agotado(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        fake = _FakeRun(duracion=100.0)
        monkeypatch.setattr(subprocess, "run", fake)
        # monotonic() "avanza" 1000 s por llamada: el presupuesto de
        # _VIDEO_FRAME_TIMEOUT (30 s) se vence en el primer frame.
        llamadas = []

        def _mono():
            llamadas.append(1)
            return 10**9 + 1000 * len(llamadas)

        monkeypatch.setattr(attachments.time, "monotonic", _mono)
        assert attachments._extraer_frames_video(video, n=4) == []
        assert fake.extracciones == 0
        # solo corrió ffprobe; ninguna extracción ffmpeg
        assert [c[0] for c in fake.comandos] == ["ffprobe"]


# ════════════════════════════════════════════════════════════════════════
# 2. componer_orden: sección de video con frames
# ════════════════════════════════════════════════════════════════════════

def _parchar_frames(monkeypatch, frames, activa=True, descripciones=None):
    """Fija el pipeline para los tests de componer_orden (hermético:
    sin Ollama, sin ffmpeg)."""
    monkeypatch.setattr(attachments, "_vision_video_activa",
                        lambda: activa)
    monkeypatch.setattr(attachments, "_extraer_frames_video",
                        lambda ruta, n=None: list(frames))
    calls = []

    def _describir(ruta, pregunta=None):
        calls.append((ruta, pregunta))
        return (descripciones or {}).get(str(ruta))

    monkeypatch.setattr(attachments, "describir_imagen", _describir)
    return calls


class TestComponerOrdenVideo:

    def test_transcripcion_mas_frames_con_captions(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        frames = [{"t": 12.5, "path": adjuntos_dir / "f0.jpg",
                   "thumb": THUMB_FALSO},
                  {"t": 62.5, "path": adjuntos_dir / "f1.jpg",
                   "thumb": THUMB_FALSO}]
        descripciones = {str(adjuntos_dir / "f0.jpg"): "un auto rojo",
                         str(adjuntos_dir / "f1.jpg"): "un puente"}
        calls = _parchar_frames(monkeypatch, frames, activa=True,
                                descripciones=descripciones)
        orden = attachments.componer_orden(
            "qué pasa en el video", [_meta_video(video)], [],
            describir=True, inline=False)
        assert "Transcripción:\n```\nhola mundo\n```" in orden
        assert "Frames analizados (visión activa):" in orden
        assert "[t=00:12] un auto rojo" in orden
        assert "[t=01:02] un puente" in orden
        assert len(calls) == 2            # un caption por frame

    def test_sin_transcripcion_igual_analiza_frames(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        frames = [{"t": 12.5, "path": adjuntos_dir / "f0.jpg",
                   "thumb": THUMB_FALSO}]
        descripciones = {str(adjuntos_dir / "f0.jpg"): "una montaña"}
        _parchar_frames(monkeypatch, frames, activa=True,
                        descripciones=descripciones)
        orden = attachments.componer_orden(
            "mirá", [_meta_video(video, texto=None)], [],
            describir=True, inline=False)
        assert "(no se pudo transcribir)" in orden
        assert "[t=00:12] una montaña" in orden

    def test_inline_marca_frames_como_adjuntos(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        frames = [{"t": float(i * 10), "path": adjuntos_dir / f"f{i}.jpg",
                   "thumb": THUMB_FALSO} for i in range(6)]
        calls = _parchar_frames(monkeypatch, frames, activa=True)
        orden = attachments.componer_orden(
            "mirá", [_meta_video(video)], [],
            describir=False, inline=True)
        # describir=False (caption redundante con inline): no se llama
        assert calls == []
        # Hasta _MAX_IMAGENES_INLINE frames marcados como adjuntos
        assert orden.count("(frame adjunto como imagen") == \
            attachments._MAX_IMAGENES_INLINE
        assert orden.count("[t=") == 6           # los 6 igual se listan

    def test_vision_inactiva_degrada_a_transcripcion_only(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        llamadas = []

        def _no_deberia(ruta, n=None):
            llamadas.append(ruta)
            return []

        monkeypatch.setattr(attachments, "_extraer_frames_video", _no_deberia)
        monkeypatch.setattr(attachments, "_vision_video_activa", lambda: False)
        orden = attachments.componer_orden(
            "mirá", [_meta_video(video)], [], describir=True, inline=False)
        assert "Transcripción" in orden
        assert "Frames analizados" not in orden
        assert llamadas == []          # ni consultó frames

    def test_frames_vacios_degrada(self, adjuntos_dir, monkeypatch):
        video = _video(adjuntos_dir)
        _parchar_frames(monkeypatch, [], activa=True)
        orden = attachments.componer_orden(
            "mirá", [_meta_video(video)], [], describir=True, inline=False)
        assert "Frames analizados" not in orden
        assert "Transcripción" in orden


# ════════════════════════════════════════════════════════════════════════
# 3. thumbnails_frames_inline: presupuesto compartido
# ════════════════════════════════════════════════════════════════════════

class TestThumbnailsFramesInline:

    @staticmethod
    def _frames(n=4):
        return [{"t": float(i), "path": pathlib.Path(f"/tmp/f{i}.jpg"),
                 "thumb": f"thumb{i}"} for i in range(n)]

    def test_modelo_sin_vision_no_extrae(self, adjuntos_dir, monkeypatch):
        monkeypatch.setattr(attachments, "modelo_admite_imagenes",
                            lambda m: False)

        def _boom(ruta, n=None):
            raise AssertionError("no debía extraer frames")

        monkeypatch.setattr(attachments, "_extraer_frames_video", _boom)
        metas = [_meta_video(_video(adjuntos_dir))]
        assert attachments.thumbnails_frames_inline(metas, "text-only") == []

    def test_cupo_cero_no_extrae(self, adjuntos_dir, monkeypatch):
        monkeypatch.setattr(attachments, "modelo_admite_imagenes",
                            lambda m: True)

        def _boom(ruta, n=None):
            raise AssertionError("con cupo 0 no debía extraer")

        monkeypatch.setattr(attachments, "_extraer_frames_video", _boom)
        metas = [_meta_video(_video(adjuntos_dir))]
        assert attachments.thumbnails_frames_inline(metas, "vlm", cupo=0) == []

    def test_toma_solo_el_cupo(self, adjuntos_dir, monkeypatch):
        monkeypatch.setattr(attachments, "modelo_admite_imagenes",
                            lambda m: True)
        monkeypatch.setattr(attachments, "_extraer_frames_video",
                            lambda ruta, n=None: self._frames(4))
        metas = [_meta_video(_video(adjuntos_dir))]
        # cupo sobrante 2 (las imágenes del usuario ocuparon 2 de 4):
        # los frames llenan el resto del presupuesto compartido.
        assert attachments.thumbnails_frames_inline(
            metas, "vlm", cupo=2) == ["thumb0", "thumb1"]


# ════════════════════════════════════════════════════════════════════════
# 4. Integración con ffmpeg REAL (skipif): video sintético lavfi testsrc
# ════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not shutil.which("ffmpeg"),
                    reason="ffmpeg no instalado")
class TestIntegracionFfmpegReal:

    def test_video_sintetico_end_to_end(self, adjuntos_dir):
        """ffmpeg real: mp4 de 3 s (testsrc + tono) → ffprobe + extracción
        de frames + persistencia + reuso en la segunda llamada."""
        video = adjuntos_dir / "clip.mp4"
        subprocess.run(
            ["ffmpeg", "-y",
             "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=10",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-shortest", str(video)],
            capture_output=True, timeout=60, check=True)
        dur = attachments._duracion_video(video)
        assert 2.5 <= dur <= 3.5, f"duración rara: {dur}"
        frames = attachments._extraer_frames_video(video, n=2)
        assert len(frames) == 2
        # equiespaciado interior de 3 s: 0.75 y 2.25
        assert [round(f["t"], 2) for f in frames] == [0.75, 2.25]
        dframes = video.with_name(video.name + ".frames")
        assert (dframes / "frame_00.jpg").is_file()
        assert (dframes / "frame_01.jpg").is_file()
        assert all(f["thumb"] for f in frames)
        # reuso: segunda llamada, mismos resultados, sin re-extraer
        # (el mtime de los frames persistidos sigue >= al del video)
        frames_2 = attachments._extraer_frames_video(video, n=2)
        assert [f["path"] for f in frames_2] == [f["path"] for f in frames]


# ════════════════════════════════════════════════════════════════════════
# 5. _preparar_orden (chat.py): los 3 canales (/chat, /chat/stream,
#    /ws/chat) comparten esta función → frames inline en todos.
# ════════════════════════════════════════════════════════════════════════

def _prep():
    """Importa _preparar_orden solo si el stack web (fastapi) está
    (patrón de test_attachments_qa)."""
    pytest.importorskip("fastapi")
    from backend.api.routes.chat import _preparar_orden
    return _preparar_orden


class TestPrepararOrdenVideo:

    @staticmethod
    def _frames(n=4, dirpath=None):
        return [{"t": float(i), "path": dirpath / f"f{i}.jpg",
                 "thumb": f"thumb{i}"} for i in range(n)]

    @staticmethod
    def _adj_video(nombre="clip.mp4"):
        import base64
        return {"name": nombre, "mime": "video/mp4",
                "data": base64.b64encode(b"FAKEVIDEODATA").decode()}

    def test_frames_inline_con_modelo_vision(self, adjuntos_dir, monkeypatch):
        monkeypatch.setattr(attachments, "modelo_admite_imagenes",
                            lambda m: True)
        monkeypatch.setattr(attachments, "_vision_video_activa",
                            lambda: True)
        monkeypatch.setattr(attachments, "_extraer_frames_video",
                            lambda ruta, n=None: self._frames(4, adjuntos_dir))
        orden, metas, imagenes = _prep()({
            "message": "mirá el video",
            "attachments": [self._adj_video()]})
        assert len(imagenes) == 4            # los 4 frames van inline
        assert all(t.startswith("thumb") for t in imagenes)
        assert "(frame adjunto como imagen" in orden   # el texto lo dice
        assert orden.count("[t=") == 4

    def test_imagenes_del_usuario_tienen_prioridad(self, adjuntos_dir, monkeypatch):
        import base64
        monkeypatch.setattr(attachments, "modelo_admite_imagenes",
                            lambda m: True)
        monkeypatch.setattr(attachments, "_vision_video_activa",
                            lambda: True)
        monkeypatch.setattr(attachments, "generar_thumbnail_b64",
                            lambda ruta, max_lado=1024: "IMGTUMB")
        monkeypatch.setattr(attachments, "_extraer_frames_video",
                            lambda ruta, n=None: self._frames(4, adjuntos_dir))
        adjs = [self._adj_video()]
        adjs += [{"name": f"foto{i}.png", "mime": "image/png",
                  "data": base64.b64encode(_png_rgb()).decode()}
                 for i in range(3)]
        orden, metas, imagenes = _prep()({
            "message": "mirá", "attachments": adjs})
        # 3 imágenes primero + 1 frame (cupo 4-3): presupuesto compartido
        assert imagenes == ["IMGTUMB"] * 3 + ["thumb0"]
        # el texto marca EXACTO el cupo (cupo_inline): 1 solo frame
        assert orden.count("(frame adjunto como imagen") == 1
        assert orden.count("[t=") == 4

    def test_modelo_text_only_captions_no_inline(self, adjuntos_dir, monkeypatch):
        monkeypatch.setattr(attachments, "modelo_admite_imagenes",
                            lambda m: False)
        monkeypatch.setattr(attachments, "_vision_video_activa",
                            lambda: True)
        monkeypatch.setattr(attachments, "_extraer_frames_video",
                            lambda ruta, n=None: self._frames(2, adjuntos_dir))
        descripciones = {str(adjuntos_dir / "f0.jpg"): "un auto rojo"}
        monkeypatch.setattr(attachments, "describir_imagen",
                            lambda ruta, pregunta=None:
                            descripciones.get(str(ruta)))
        orden, metas, imagenes = _prep()({
            "message": "mirá", "attachments": [self._adj_video()]})
        assert imagenes == []                # sin visión → nada inline
        # captions por MODELO_VISION dentro de la orden
        assert "[t=00:00] un auto rojo" in orden
        assert "(frame adjunto" not in orden


# ════════════════════════════════════════════════════════════════════════
# 6. node_vision (grafo): la tool "vision" también acepta VIDEOS
# ════════════════════════════════════════════════════════════════════════

def _mp4_fake(d):
    p = d / "clip.mp4"
    p.write_bytes(b"\x00\x00\x00\x18ftyp" + b"0" * 64)
    return p


class TestNodeVisionVideo:

    def test_video_por_args_con_frames(self, adjuntos_dir, monkeypatch):
        """tool vision con path de video → análisis por frames con
        timestamps y captions (pipeline compartido con adjuntos)."""
        import core.agent.graph_nodes as gn
        monkeypatch.setattr(attachments, "_vision_video_activa", lambda: True)
        frames = [{"t": 12.5, "path": adjuntos_dir / "f0.jpg",
                   "thumb": THUMB_FALSO}]
        monkeypatch.setattr(attachments, "_extraer_frames_video",
                            lambda ruta, n=None: list(frames))
        monkeypatch.setattr(attachments, "describir_imagen",
                            lambda ruta, pregunta=None: "un auto rojo")
        out = gn.node_vision({"orden": "mirá el video",
                              "_tool_args": {"path": str(_mp4_fake(adjuntos_dir))}})
        assert "análisis visual por frames" in out["vision_result"]
        assert "[t=00:12] un auto rojo" in out["vision_result"]

    def test_video_mencionado_en_la_orden(self, adjuntos_dir, monkeypatch):
        """Sin _tool_args.path: el video aparece en la orden del usuario
        (camino 'otro' del nodo) → también se analiza."""
        import core.agent.graph_nodes as gn
        monkeypatch.setattr(attachments, "_vision_video_activa", lambda: True)
        frames = [{"t": 0.5, "path": adjuntos_dir / "f0.jpg",
                   "thumb": THUMB_FALSO}]
        monkeypatch.setattr(attachments, "_extraer_frames_video",
                            lambda ruta, n=None: list(frames))
        monkeypatch.setattr(attachments, "describir_imagen",
                            lambda ruta, pregunta=None: "mar de fondo")
        video = _mp4_fake(adjuntos_dir)
        out = gn.node_vision({"orden": f"mirá el video {video}"})
        assert "[t=00:00] mar de fondo" in out["vision_result"]

    def test_video_sin_capacidad_error_menciona_transcripcion(self, adjuntos_dir, monkeypatch):
        """Sin ffmpeg/modelo con visión (o video ilegible): [ERROR] con
        lo disponible (transcripción), igual contrato que los tests
        heredados (test_adjuntos_agent_loop_qa)."""
        import core.agent.graph_nodes as gn
        monkeypatch.setattr(attachments, "_vision_video_activa",
                            lambda: False)
        out = gn.node_vision({"orden": "mirá el video",
                              "_tool_args": {"path": str(_mp4_fake(adjuntos_dir))}})
        assert "[ERROR]" in out["vision_result"]
        assert "transcripción" in out["vision_result"]

    def test_pdf_sigue_dando_error_de_imagen(self, adjuntos_dir, monkeypatch):
        """Un archivo que no es imagen NI video conserva el mensaje
        original (regresión del contrato con adjuntos PDF)."""
        import core.agent.graph_nodes as gn
        pdf = adjuntos_dir / "informe.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(attachments, "_vision_video_activa", lambda: True)
        out = gn.node_vision({"orden": "describí el adjunto",
                              "_tool_args": {"path": str(pdf)}})
        assert "[ERROR]" in out["vision_result"]
        assert "no es una imagen" in out["vision_result"]

    def test_tool_registry_anuncia_video(self):
        """El planner tiene que SABER que vision acepta videos para
        elegirla ante un adjunto de video (Paso 4 del plan)."""
        from core.agent.tool_registry import TOOL_REGISTRY, TOOL_PARAMETROS
        assert "VIDEO" in TOOL_REGISTRY["vision"]["descripcion"].upper()
        assert "mp4" in TOOL_PARAMETROS["vision"]["properties"]["path"]["description"]


# ════════════════════════════════════════════════════════════════════════
# 7. Config: VIDEO_FRAMES versionado (Paso 6)
# ════════════════════════════════════════════════════════════════════════

class TestConfigVideoFrames:

    def test_config_json_versionado(self):
        """La clave vive en config.json (defaults versionados) con el
        valor aprobado del plan (4), sin necesitar config.local.json."""
        import json
        cfg_path = (pathlib.Path(__file__).resolve().parent.parent
                    / "core" / "config" / "config.json")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert cfg["VIDEO_FRAMES"] == 4

    def test_defaults_y_validador_del_config_manager(self):
        from core.config.config_manager import ConfigManager
        assert ConfigManager.DEFAULTS.get("VIDEO_FRAMES") == 4
        validar = ConfigManager.VALIDATORS["VIDEO_FRAMES"]
        assert validar(4) is True
        assert validar(1) is True
        assert validar(8) is True
        assert validar(0) is False
        assert validar(9) is False        # tope duro: el validador coincide
        assert validar("4") is False      # solo int


# ════════════════════════════════════════════════════════════════════════
# 8. Transcripción: ffmpeg→ndarray SIN PyAV (fix de versiones av 19)
# ════════════════════════════════════════════════════════════════════════

class TestTranscripcionAudio:

    def test_transcribe_recibe_el_array_no_el_path(self, adjuntos_dir, monkeypatch):
        """Regresión del conflicto faster-whisper × PyAV 19: el path
        revienta en av.open(metadata_errors); el ARRAY no toca PyAV."""
        import sys
        import types

        class _Seg:
            text = "hola mundo"

        class _Model:
            def __init__(self, modelo, device=None, compute_type=None):
                pass

            def transcribe(self, audio, language=None, vad_filter=None,
                           max_initial_timestamp=None):
                # recibió un ndarray-like, NO un path/str
                assert not isinstance(audio, (str, bytes, pathlib.Path))
                assert len(audio) > 0
                return [_Seg()], None

        fake_mod = types.ModuleType("faster_whisper")
        fake_mod.WhisperModel = _Model
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)

        import numpy as np
        monkeypatch.setattr(
            attachments, "_audio_a_numpy",
            lambda ruta: np.zeros(1600, dtype=np.float32))
        video = _mp4_fake(adjuntos_dir)
        assert attachments._transcribir_audio(video) == "hola mundo"

    def test_sin_audio_degrada_a_none(self, adjuntos_dir, monkeypatch):
        import sys
        import types
        llamadas = []

        class _Boom:
            def __init__(self, *a, **k):
                llamadas.append(a)  # no debe instanciarse

        fake_mod = types.ModuleType("faster_whisper")
        fake_mod.WhisperModel = _Boom
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_mod)
        monkeypatch.setattr(attachments, "_audio_a_numpy",
                            lambda ruta: None)
        assert attachments._transcribir_audio(_mp4_fake(adjuntos_dir)) is None
        assert llamadas == []       # ni cargó el modelo

    @pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("espeak")),
                        reason="ffmpeg/espeak no instalados")
    def test_audio_a_numpy_end_to_end(self, adjuntos_dir):
        """ffmpeg real + voz real (espeak): el clip con pista de voz
        decodifica a float32 16 kHz (≈ duración × 16000 muestras)."""
        import numpy as np
        video = adjuntos_dir / "voz.mp4"
        subprocess.run(
            ["ffmpeg", "-y",
             "-f", "lavfi", "-i", "testsrc=duration=3:size=160x120:rate=5",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
             "-shortest", str(video)],
            capture_output=True, timeout=60, check=True)
        audio = attachments._audio_a_numpy(video)
        assert audio is not None and len(audio) > 0
        assert audio.dtype == np.float32
        # ~3 s × 16000 Hz (tolerancia por -shortest)
        assert 2.0 * 16000 < len(audio) < 4.0 * 16000
        # sin pista de audio → None (video mudo)
        mudo = adjuntos_dir / "mudo.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi",
             "-i", "testsrc=duration=1:size=160x120:rate=5",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mudo)],
            capture_output=True, timeout=60, check=True)
        assert attachments._audio_a_numpy(mudo) is None


# ════════════════════════════════════════════════════════════════════════
# 9. Sidecar de texto extraído (fix del flujo upload→path de la Web UI)
# ════════════════════════════════════════════════════════════════════════

def _adj_data(nombre: str, mime: str, contenido: bytes) -> dict:
    import base64
    return {"name": nombre, "mime": mime,
            "data": base64.b64encode(contenido).decode()}


class TestSidecarTexto:

    def test_guardar_escribe_sidecar_y_path_lo_recupera(self, adjuntos_dir, monkeypatch):
        """Flujo UI: guardar_adjuntos (lo que hace /attachments/upload)
        persiste la transcripción junto al archivo; adjuntos_ya_guardados
        (lo que hace el turno con la referencia por path) la recupera."""
        monkeypatch.setattr(attachments, "_transcribir_audio",
                            lambda ruta: "hola mundo transcrita")
        metas, errores = attachments.guardar_adjuntos(
            [_adj_data("clip.mp4", "video/mp4", b"FAKEVIDEO")])
        assert errores == []
        video = pathlib.Path(metas[0]["path"])
        sidecar = attachments._path_texto_adjunto(video)
        assert sidecar.is_file()
        assert sidecar.read_text(encoding="utf-8") == "hola mundo transcrita"
        # El turno de la UI: referencia por path, SIN texto en el body
        referencias = [{"name": "clip.mp4", "mime": "video/mp4",
                        "path": str(video)}]
        metas2 = attachments.adjuntos_ya_guardados(referencias)
        assert metas2[0]["texto_extraido"] == "hola mundo transcrita"

    def test_pdf_tambien_recupera_por_path(self, adjuntos_dir, monkeypatch):
        monkeypatch.setattr(attachments, "_extraer_texto_pdf",
                            lambda ruta: "texto del pdf")
        metas, _ = attachments.guardar_adjuntos(
            [_adj_data("informe.pdf", "application/pdf", b"%PDF-1.4 fake")])
        metas2 = attachments.adjuntos_ya_guardados(
            [{"name": "informe.pdf", "mime": "application/pdf",
              "path": metas[0]["path"]}])
        assert metas2[0]["texto_extraido"] == "texto del pdf"

    def test_text_kind_no_necesita_sidecar(self, adjuntos_dir):
        """componer_orden RELEE el archivo de texto original: no se
        escribe sidecar para kind text (evita duplicar el contenido)."""
        metas, _ = attachments.guardar_adjuntos(
            [_adj_data("notas.txt", "text/plain", b"contenido original")])
        assert not attachments._path_texto_adjunto(
            pathlib.Path(metas[0]["path"])).exists()

    def test_sin_sidecar_degrada_como_antes(self, adjuntos_dir):
        video = _mp4_fake(adjuntos_dir)
        metas = attachments.adjuntos_ya_guardados(
            [{"name": video.name, "mime": "video/mp4", "path": str(video)}])
        assert metas[0]["kind"] == "video"
        assert "texto_extraido" not in metas[0]

    def test_sidecar_vacio_o_gigante_da_none(self, adjuntos_dir):
        video = _mp4_fake(adjuntos_dir)
        attachments._path_texto_adjunto(video).write_text(
            "   ", encoding="utf-8")
        assert attachments._leer_texto_adjunto(video) is None
        attachments._path_texto_adjunto(video).write_text(
            "x" * (attachments._MAX_TEXTO_SIDECAR + 1), encoding="utf-8")
        assert attachments._leer_texto_adjunto(video) is None

    def test_flujo_ui_completo_preparar_orden(self, adjuntos_dir, monkeypatch):
        """End-to-end del camino de la Web UI: upload (data) → turno con
        referencia por path → componer_orden con transcripción Y frames."""
        monkeypatch.setattr(attachments, "_transcribir_audio",
                            lambda ruta: "voz del video transcrita")
        metas, _ = attachments.guardar_adjuntos(
            [_adj_data("clip.mp4", "video/mp4", b"FAKEVIDEO")])
        monkeypatch.setattr(attachments, "_vision_video_activa",
                            lambda: True)
        frames = [{"t": 12.5, "path": adjuntos_dir / "f0.jpg",
                   "thumb": THUMB_FALSO}]
        monkeypatch.setattr(attachments, "_extraer_frames_video",
                            lambda ruta, n=None: list(frames))
        monkeypatch.setattr(attachments, "modelo_admite_imagenes",
                            lambda m: False)   # MODELO text-only → captions
        monkeypatch.setattr(attachments, "describir_imagen",
                            lambda ruta, pregunta=None: "un auto rojo")
        orden, metas_out, imagenes = _prep()({
            "message": "mirá el video",
            "attachments": [{"name": "clip.mp4", "mime": "video/mp4",
                             "path": metas[0]["path"]}]})
        # El fix: la transcripción del upload LLEGA al turno por el path
        assert "Transcripción:\n```\nvoz del video transcrita\n```" in orden
        assert "[t=00:12] un auto rojo" in orden
        assert imagenes == []    # text-only: sin inline
