"""
attachments.py — Soporte de imágenes y archivos adjuntos en el chat web.

La Web UI manda en el body del chat:
    attachments: [{"name": "foto.png", "mime": "image/png",
                  "data": "<base64 sin prefijo data:>"}]

Este módulo:
1. los guarda en <BASE_AETHER>/adjuntos/ (persisten: Aether los puede releer
   con sus tools de fs en turnos posteriores, y el usuario los tiene en disco),
2. compone el texto extra que se agrega a la orden del grafo:
   - archivos de texto → contenido embebido (acotado) inline,
   - PDF/DOCX → texto extraído,
   - audio/video → transcripción con faster-whisper,
   - imágenes → descripción generada con el modelo de visión de Ollama
     (mismo patrón que core/tools/vision.py) + la ruta en disco por si el
     agente quiere operar sobre el archivo.

3. imágenes → van al modelo como THUMBNAIL (lado mayor ≤ 1024px, JPEG):
   - si el MODELO principal admite visión (ver modelo_admite_imagenes), el
     thumbnail viaja incrustado en el mensaje del usuario del grafo y el
     modelo analiza la foto directamente con el prompt del usuario;
   - si no (modelo text-only), caption generado con MODELO_VISION (mismo
     patrón que core/tools/vision.py) + la ruta en disco por si el agente
     quiere operar sobre el archivo.

Todo best-effort: un adjunto roto nunca tira el request completo.
"""
from __future__ import annotations

import base64
import binascii
import io
import re
import subprocess
import tempfile
import time
from pathlib import Path

from core.config import capabilities as _capabilities

_MAX_BYTES = 10 * 1024 * 1024      # 10 MB por adjunto
_MAX_TEXTO_INLINE = 4000           # chars por archivo de texto embebido
_MAX_TEXTO_EMBEBIDOS = 4           # cuántos archivos de texto se embeben
_MAX_IMAGENES_DESCRITAS = 2        # cuántas imágenes se describen con visión
_MAX_AUDIO_SEGUNDOS = 600          # tope de 10 min para transcribir
# ── Pipeline de frames de video (análisis visual) ──
# Cada frame se extrae UNA sola vez (ffmpeg), persiste junto al video y
# se reusa para captions e inline (ver _extraer_frames_video). Los
# límites son de degradación: si se exceden, el video queda en el
# comportamiento anterior (solo transcripción), nunca rompe el turno.
_VIDEO_FRAMES = 4                  # default: frames equiespaciados por video
_VIDEO_FRAMES_TOPE = 8             # tope duro del config VIDEO_FRAMES (sanity)
_VIDEO_DURACION_MAX = 4 * 3600     # videos más largos → sin frames
_VIDEO_FRAME_TIMEOUT = 30          # presupuesto total (s) de ffmpeg por video
_THUMB_MAX_LADO = 1024             # lado mayor (px) del thumbnail para visión
_THUMB_CALIDAD_JPEG = 85           # calidad del JPEG del thumbnail
_MAX_IMAGENES_INLINE = 4           # cuántos thumbnails van inline al modelo
_THUMB_CACHE_MAX = 64              # thumbnails cacheados (path, mtime, lado)

_EXTS_TEXTO = {
    ".txt", ".md", ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".csv",
    ".log", ".sh", ".bash", ".zsh", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".xml", ".html", ".css", ".sql", ".env", ".c", ".h",
    ".cpp", ".hpp", ".rs", ".go", ".java", ".rb", ".php", ".lua", ".diff",
}

# Extensiones soportadas por categoría
_EXTS_PDF = {".pdf"}
_EXTS_DOCX = {".docx", ".doc"}
_EXTS_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".opus"}
_EXTS_VIDEO = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
_EXTS_VECTOR = {".svg"}

_RE_NOMBRE_SEGURO = re.compile(r"[^A-Za-z0-9._-]+")


def _directorio_adjuntos() -> Path:
    from core.config.settings import BASE_AETHER
    d = Path(BASE_AETHER) / "adjuntos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _kind(name: str, mime: str) -> str:
    """image | text | pdf | docx | audio | video | vector | other."""
    mime = (mime or "").lower()
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("text/") or mime in (
        "application/json", "application/javascript", "application/xml",
        "application/x-yaml", "application/toml",
    ):
        return "text"
    ext = Path(name).suffix.lower()
    if ext in _EXTS_PDF:
        return "pdf"
    if ext in _EXTS_DOCX:
        return "docx"
    if ext in _EXTS_AUDIO:
        return "audio"
    if ext in _EXTS_VIDEO:
        return "video"
    if ext in _EXTS_VECTOR:
        return "vector"
    if ext in _EXTS_TEXTO:
        return "text"
    return "other"


def _extraer_texto_pdf(ruta: Path) -> str | None:
    """Extrae texto de PDF usando pymupdf (fitz)."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(ruta)
        texto = []
        for page in doc:
            texto.append(page.get_text())
        doc.close()
        return "\n".join(texto).strip() or None
    except Exception:
        return None

def _extraer_texto_docx(ruta: Path) -> str | None:
    """Extrae texto de DOCX usando python-docx."""
    try:
        from docx import Document
        doc = Document(ruta)
        return "\n".join(p.text for p in doc.paragraphs).strip() or None
    except Exception:
        return None
def _extraer_texto_vector(ruta: Path) -> str | None:
    """Extrae texto de SVG (vector) - simple regex sobre XML."""
    try:
        contenido = ruta.read_text(encoding="utf-8", errors="replace")
        import re as _re
        textos = _re.findall(r">([^<]{2,})<", contenido)
        return " ".join(t.strip() for t in textos if t.strip()) or None
    except Exception:
        return None
def _audio_a_numpy(ruta: Path):
    """Decodifica el audio (de cualquier archivo de audio/VIDEO) a un
    ndarray float32 mono 16 kHz vía ffmpeg + PCM crudo, SIN PyAV.

    Por qué: faster-whisper decodifica con PyAV cuando recibe un PATH y
    llama av.open(..., metadata_errors=...) — kwarg que PyAV 19 eliminó
    (y PyAV 14 no publica wheels para Python 3.14): cualquier
    combinación de versiones revienta la transcripción. Pasando el
    ARRAY, faster-whisper no toca PyAV y ffmpeg — dependencia que el
    pipeline de video ya requiere — hace la decodificación.
    """
    try:
        import numpy as np
        r = subprocess.run(
            ["ffmpeg", "-i", str(ruta), "-vn", "-ac", "1", "-ar", "16000",
             "-f", "f32le", "-acodec", "pcm_f32le", "-"],
            capture_output=True, timeout=120)
        if r.returncode != 0 or not r.stdout:
            return None
        return np.frombuffer(r.stdout, dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None


def _transcribir_audio(ruta: Path) -> str | None:
    """Transcribe audio/video usando faster-whisper.

    El audio se decodifica UNA vez a ndarray con ffmpeg (_audio_a_numpy)
    y WhisperModel.transcribe recibe el ARRAY (no el path): PyAV queda
    fuera del pipeline, inmune al conflicto de versiones (ver
    _audio_a_numpy). Sin pista de audio / ffmpeg caído / whisper no
    instalado → None (el adjunto degrada a "no se pudo transcribir").
    """
    try:
        from faster_whisper import WhisperModel
        from core.config.config_manager import get_config_manager
        cfg = get_config_manager()
        modelo = cfg.get("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
        audio = _audio_a_numpy(ruta)
        if audio is None or len(audio) == 0:
            return None
        model = WhisperModel(modelo, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            audio, language=None,  # auto-detect
            vad_filter=True,
            max_initial_timestamp=_MAX_AUDIO_SEGUNDOS,
        )
        texto = " ".join(seg.text for seg in segments).strip()
        return texto or None
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════
# FRAMES DE VIDEO: análisis visual del contenido (matriz de capacidades
# core/config/capabilities — feature "video_frames": ffmpeg + algún
# modelo con visión). Cada frame se extrae UNA sola vez por procesamiento
# y se REUTILIZA para captions (describir_imagen) e imágenes inline
# (thumbnails_frames_inline): los JPEG persisten junto al video y
# generar_thumbnail_b64 cachea por path+mtime.
# ════════════════════════════════════════════════════════════════════════

def _vision_video_activa() -> bool:
    """True si el pipeline de frames está activo según la matriz de
    capacidades (fuente única: core/config/capabilities). Nunca revienta."""
    try:
        return bool(_capabilities.funciones_disponibles()["video_frames"])
    except Exception:  # noqa: BLE001
        return False


def _duracion_video(ruta: Path) -> float:
    """Duración en segundos vía ffprobe (0.0 si falla). Best-effort."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(ruta)],
            capture_output=True, text=True, timeout=15)
        if r.returncode != 0:
            return 0.0
        return max(0.0, float(r.stdout.strip()))
    except Exception:  # noqa: BLE001
        return 0.0


def _formato_mmss(seg: float) -> str:
    """'03:12' / '01:03:12' para etiquetar timestamps de frames."""
    s = max(0, int(round(seg)))
    h, resto = divmod(s, 3600)
    m, s = divmod(resto, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _video_frames_n() -> int:
    """VIDEO_FRAMES de config con default 4 (ajuste aprobado) y tope duro."""
    try:
        from core.config.config_manager import get_config_manager
        n = int(get_config_manager().get("VIDEO_FRAMES", _VIDEO_FRAMES))
    except Exception:  # noqa: BLE001
        n = _VIDEO_FRAMES
    return max(1, min(n, _VIDEO_FRAMES_TOPE))


def _extraer_frames_video(ruta: Path, n: int | None = None) -> list[dict]:
    """n frames equiespaciados del video, EXTRAÍDOS UNA SOLA VEZ.

    Los JPEG persisten junto al video (<video>.frames/frame_NN.jpg) y se
    reutilizan para todo: captions de MODELO_VISION, thumbnails inline y
    turnos futuros (generar_thumbnail_b64 cachea por path+mtime, así el
    mismo frame nunca se extrae ni re-escala dos veces). Solo se
    regeneran si el video cambió (mtime del frame < mtime del video).

    Límites (degradación a transcripción-only, nunca rompe el turno):
      - duración del video <= _VIDEO_DURACION_MAX;
      - presupuesto total de ffmpeg de _VIDEO_FRAME_TIMEOUT segundos
        (además del timeout individual de cada comando);
      - resolución acotada en la extracción (scale <= 1024px, mismo
        filtro que _thumb_ffmpeg): el JPEG persistido ya es liviano.

    Devuelve [{'t': seg, 'path': Path, 'thumb': b64}]; [] si no se pudo.
    """
    try:
        ruta = Path(ruta)
        if not ruta.is_file():
            return []
        n = n if n and n > 0 else _video_frames_n()
        dur = _duracion_video(ruta)
        if dur <= 0 or dur > _VIDEO_DURACION_MAX:
            return []
        try:
            mtime_video = ruta.stat().st_mtime
        except OSError:
            return []
        dframes = ruta.with_name(ruta.name + ".frames")
        dframes.mkdir(exist_ok=True)
        # Equiespaciado interior (evita frame negro de inicio y corte final)
        instantes = [dur * (i + 0.5) / n for i in range(n)]
        frames: list[dict] = []
        presupuesto = time.monotonic() + _VIDEO_FRAME_TIMEOUT
        for i, t in enumerate(instantes):
            out = dframes / f"frame_{i:02d}.jpg"
            try:
                reusar = out.is_file() and out.stat().st_mtime >= mtime_video
            except OSError:
                reusar = False
            if not reusar:
                if time.monotonic() > presupuesto:
                    break  # presupuesto agotado: degradar con lo que hay
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(ruta),
                         "-frames:v", "1",
                         "-vf", ("scale='min(iw,1024)':'min(ih,1024)'"
                                 ":force_original_aspect_ratio=decrease"),
                         "-q:v", "4", str(out)],
                        capture_output=True, timeout=15, check=True)
                except Exception:  # noqa: BLE001
                    continue
            thumb = generar_thumbnail_b64(out)
            if thumb:
                frames.append({"t": t, "path": out, "thumb": thumb})
        return frames
    except Exception:  # noqa: BLE001
        return []


def thumbnails_frames_inline(metas: list[dict], modelo: str,
                             cupo: int | None = None) -> list[str]:
    """Thumbnails (b64) de frames de videos adjuntos si `modelo` admite
    visión nativa (ver modelo_admite_imagenes).

    `cupo` = espacio restante del presupuesto compartido de imágenes
    inline (_MAX_IMAGENES_INLINE): las imágenes del usuario van primero
    (el caller _preparar_orden pasa el cupo sobrante; el overflow de
    frames queda solo-caption).

    Reusa los frames persistidos por _extraer_frames_video (una sola
    extracción por video; el thumbnail ya está cacheado).
    """
    if cupo is None:
        cupo = _MAX_IMAGENES_INLINE
    if cupo <= 0 or not modelo_admite_imagenes(modelo):
        return []
    out: list[str] = []
    for m in [x for x in metas if x.get("kind") == "video" and x.get("path")]:
        for f in _extraer_frames_video(Path(m["path"])):
            out.append(f["thumb"])
            if len(out) >= cupo:
                return out
    return out
# ════════════════════════════════════════════════════════════════════════
# THUMBNAILS: la foto se manda al modelo como thumbnail, no a resolución
# completa (una foto de 4000px son ~4 MB de base64; el mismo contenido en
# thumbnail JPEG de 1024px pesa ~100 KB y los modelos de visión re-escalan
# internamente igual). Se usa para:
#   - describir_imagen(): el caption de MODELO_VISION ve el thumbnail;
#   - thumbnails_inline(): el MODELO principal (si admite visión, ver
#     modelo_admite_imagenes) recibe el thumbnail incrustado en su propio
#     mensaje, junto con el prompt del usuario.
# Cadena de downscale: Pillow → OpenCV → ffmpeg → bytes originales (mejor
# mandar la imagen completa que perder el análisis).
# ════════════════════════════════════════════════════════════════════════

_THUMB_CACHE: dict[tuple[str, int, int], str] = {}


def _thumb_pil(ruta: Path, max_lado: int) -> str | None:
    """Thumbnail JPEG (b64) con Pillow."""
    try:
        from PIL import Image
        with Image.open(ruta) as im:
            if im.mode in ("RGBA", "LA", "P"):
                # JPEG no tiene alfa: compositar sobre fondo blanco para no
                # perder transparencias como canal negro.
                im = im.convert("RGBA")
                fondo = Image.new("RGB", im.size, (255, 255, 255))
                fondo.paste(im, mask=im.split()[-1])
                im = fondo
            else:
                im = im.convert("RGB")
            resampling = getattr(Image, "Resampling", Image)
            im.thumbnail((max_lado, max_lado), resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=_THUMB_CALIDAD_JPEG)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def _thumb_cv2(ruta: Path, max_lado: int) -> str | None:
    """Thumbnail JPEG (b64) con OpenCV."""
    try:
        import cv2
        img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        escala = max_lado / max(h, w)
        if escala < 1:  # nunca escalar hacia arriba
            img = cv2.resize(
                img, (max(1, int(w * escala)), max(1, int(h * escala))),
                interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(
            ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _THUMB_CALIDAD_JPEG])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        return None


def _thumb_ffmpeg(ruta: Path, max_lado: int) -> str | None:
    """Thumbnail JPEG (b64) con ffmpeg (sin dependencias de Python)."""
    out: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            out = Path(tmp.name)
        # min(iw, N)/min(ih, N) evita UPSCALAR imágenes chicas;
        # force_original_aspect_ratio=decrease preserva el aspecto.
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(ruta),
             "-vf", (f"scale='min(iw,{max_lado})':'min(ih,{max_lado})'"
                     ":force_original_aspect_ratio=decrease"),
             "-q:v", "4", str(out)],
            capture_output=True, timeout=30)
        if r.returncode == 0 and out.stat().st_size > 0:
            return base64.b64encode(out.read_bytes()).decode("utf-8")
        return None
    except Exception:
        return None
    finally:
        if out is not None:
            out.unlink(missing_ok=True)


def generar_thumbnail_b64(ruta: Path | str,
                          max_lado: int = _THUMB_MAX_LADO) -> str | None:
    """Thumbnail de una imagen como base64 (cacheado por path+mtime+lado).

    Nunca revienta: si ninguna vía de downscale está disponible cae a los
    bytes originales en base64 (el análisis sigue funcionando como antes).
    """
    ruta = Path(ruta)
    try:
        clave = (str(ruta), ruta.stat().st_mtime_ns, max_lado)
    except OSError:
        return None
    cacheado = _THUMB_CACHE.get(clave)
    if cacheado is not None:
        return cacheado
    data = (_thumb_pil(ruta, max_lado) or _thumb_cv2(ruta, max_lado)
            or _thumb_ffmpeg(ruta, max_lado))
    if data is None:
        try:
            data = base64.b64encode(ruta.read_bytes()).decode("utf-8")
        except OSError:
            return None
    _THUMB_CACHE[clave] = data
    if len(_THUMB_CACHE) > _THUMB_CACHE_MAX:
        _THUMB_CACHE.pop(next(iter(_THUMB_CACHE)))
    return data


# Cache delegado a core/config/capabilities (fuente única de verdad de
# la matriz de capacidades por modelo). ES EL MISMO objeto dict: el
# fixture de tests que hace attachments._CACHE_MODELO_VISION.clear()
# sigue limpiando el cache real tras la delegación.
_CACHE_MODELO_VISION: dict[str, dict] = _capabilities._CACHE_CAPACIDADES


def modelo_admite_imagenes(modelo: str) -> bool:
    """True si el modelo de Ollama acepta imágenes (visión nativa).

    DELEGADO en core/config/capabilities.modelo_admite_imagenes: consulta
    POST /api/show y mira `capabilities` (formato nuevo) o
    `projector_info` (encoder CLIP), cacheado por host+modelo. Este
    wrapper se conserva para no romper callers ni tests. Un modelo
    text-only rechaza mensajes con "images" (HTTP 400), así que el
    thumbnail inline SOLO se activa cuando esto devuelve True — si no,
    el análisis de la imagen cae al caption de MODELO_VISION.
    """
    return _capabilities.modelo_admite_imagenes(modelo)


def thumbnails_inline(metas: list[dict], modelo: str) -> list[str]:
    """Thumbnails (b64) de las imágenes adjuntas si `modelo` admite visión.

    Todo-o-nada: si alguna de las candidatas no genera thumbnail se
    devuelve [] y el caller cae al camino anterior (caption por
    MODELO_VISION dentro de componer_orden).
    """
    imagenes = [m for m in metas if m.get("kind") == "image" and m.get("path")]
    if not imagenes or not modelo_admite_imagenes(modelo):
        return []
    thumbs = [generar_thumbnail_b64(m["path"])
              for m in imagenes[:_MAX_IMAGENES_INLINE]]
    return thumbs if thumbs and all(thumbs) else []


def _decodificar(adj: dict, indice: int) -> tuple[dict | None, str | None]:
    """Valida y decodifica un adjunto. Devuelve (meta_guardable, error)."""
    nombre = _RE_NOMBRE_SEGURO.sub("_", str(adj.get("name") or f"adjunto_{indice}"))
    nombre = nombre.lstrip(".") or f"adjunto_{indice}"
    mime = str(adj.get("mime") or "application/octet-stream")
    data = adj.get("data") or ""
    # Tolerar data URLs completas ("data:image/png;base64,...."): solo la
    # parte base64 importa. Sin esto, un cliente que mande el dataURL entero
    # cae en "base64 inválido" y la imagen se descarta.
    if isinstance(data, str) and data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    if not data:
        return None, f"'{nombre}': vino vacío"
    try:
        crudo = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None, f"'{nombre}': base64 inválido"
    if len(crudo) > _MAX_BYTES:
        return None, (f"'{nombre}': excede {_MAX_BYTES // (1024 * 1024)} MB "
                      f"({len(crudo) // (1024 * 1024)} MB)")
    return {"name": nombre, "mime": mime, "bytes": crudo,
            "kind": _kind(nombre, mime)}, None


# ── SIDECAR DE TEXTO EXTRAÍDO (fix del flujo upload→path) ──
# La Web UI sube con /api/attachments/upload (que transcribe audio/video y
# extrae PDF/DOCX/SVG) pero el turno referencia el adjunto solo por PATH:
# el texto obtenido en el upload se perdía. Se persiste junto al archivo
# (<archivo>.txt) y adjuntos_ya_guardados lo relee — mismo patrón de
# persistencia+reuso que los frames de video (<video>.frames/).
# kind "text" no necesita sidecar: componer_orden relee el original.
_KINDS_TEXTO_SIDECAR = {"pdf", "docx", "vector", "audio", "video"}
_MAX_TEXTO_SIDECAR = 2 * 1024 * 1024   # sanity al leer (2 MB)


def _path_texto_adjunto(ruta: Path) -> Path:
    """Sidecar de texto extraído: <archivo>.txt junto al adjunto."""
    return ruta.with_name(ruta.name + ".txt")


def _guardar_texto_adjunto(ruta: Path, texto: str) -> None:
    """Persiste el texto extraído junto al adjunto (best-effort)."""
    try:
        _path_texto_adjunto(ruta).write_text(texto, encoding="utf-8")
    except Exception:  # noqa: BLE001 — nunca romper el guardado
        pass


def _leer_texto_adjunto(ruta: Path) -> str | None:
    """Texto extraído persistido junto al adjunto, o None. Best-effort."""
    try:
        sidecar = _path_texto_adjunto(ruta)
        if sidecar.stat().st_size > _MAX_TEXTO_SIDECAR:
            return None
        return sidecar.read_text(encoding="utf-8",
                                 errors="replace").strip() or None
    except Exception:  # noqa: BLE001
        return None


def guardar_adjuntos(attachments: list[dict]) -> tuple[list[dict], list[str]]:
    """Guarda los adjuntos en disco. Devuelve (metas, errores)."""
    metas: list[dict] = []
    errores: list[str] = []
    ts = time.strftime("%Y%m%d-%H%M%S")
    destino = _directorio_adjuntos()
    for i, adj in enumerate(attachments[:8]):  # tope duro de 8 por mensaje
        meta, err = _decodificar(adj, i)
        if err:
            errores.append(err)
            continue
        ext = Path(meta["name"]).suffix.lower()
        fname = f"{ts}_{i}_{meta['name']}"
        fpath = destino / fname
        fpath.write_bytes(meta["bytes"])
        meta["path"] = str(fpath)
        meta["size"] = len(meta["bytes"])
        metas.append(meta)
    for m in metas:
        kind = m["kind"]
        path = Path(m["path"])
        texto_extra = None
        if kind == "text":
            try:
                texto_extra = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        elif kind == "pdf":
            texto_extra = _extraer_texto_pdf(path)
        elif kind == "docx":
            texto_extra = _extraer_texto_docx(path)
        elif kind == "vector":
            texto_extra = _extraer_texto_vector(path)
        elif kind in ("audio", "video"):
            texto_extra = _transcribir_audio(path)
        if texto_extra:
            m["texto_extraido"] = texto_extra
            # Persistir junto al archivo (sidecar) para que los turnos
            # futuros vía upload→path lo recuperen: la UI referencia el
            # adjunto por path SIN re-mandar el texto (ver
            # _KINDS_TEXTO_SIDECAR).
            if kind in _KINDS_TEXTO_SIDECAR:
                _guardar_texto_adjunto(path, texto_extra)
    return metas, errores
def adjuntos_ya_guardados(adjuntos: list[dict]) -> list[dict]:
    """Reconstruye metas de adjuntos subidos previamente (upload multipart).

    La Web UI sube los archivos a /api/attachments/upload y luego los
    referencia por PATH en el body del chat (sin re-mandar el base64).
    Esta función valida esas referencias: el archivo debe existir, estar
    dentro del directorio de adjuntos y tener un tamaño sano — nunca se
    acepta un path arbitrario del cliente.
    """
    metas: list[dict] = []
    try:
        base = _directorio_adjuntos().resolve()
    except OSError:
        return metas
    for adj in adjuntos[:8]:
        try:
            p = Path(str(adj.get("path") or "")).expanduser().resolve()
            if not p.is_file() or base not in p.parents:
                continue
            size = p.stat().st_size
            if size == 0 or size > _MAX_BYTES:
                continue
            nombre = _RE_NOMBRE_SEGURO.sub(
                "_", str(adj.get("name") or p.name))
            mime = str(adj.get("mime") or "application/octet-stream")
            kind = _kind(nombre, mime)
            meta = {"name": nombre, "mime": mime, "kind": kind,
                     "path": str(p), "size": size}
            # Fix del flujo upload→path: recuperar el texto extraído que
            # el upload persistió junto al archivo (transcripción de
            # audio/video, texto de PDF/DOCX/SVG). Antes se perdía y el
            # video llegaba al turno como "no se pudo transcribir".
            if kind in _KINDS_TEXTO_SIDECAR:
                texto = _leer_texto_adjunto(p)
                if texto:
                    meta["texto_extraido"] = texto
            metas.append(meta)
        except OSError:
            continue
    return metas


def validar_path_adjunto(path: str) -> Path | None:
    """Path de UN adjunto válido (dentro del directorio de adjuntos,
    archivo existente, tamaño sano) o None. Misma regla que
    adjuntos_ya_guardados, para el endpoint que sirve archivos a la Web UI:
    nunca se acepta un path arbitrario del cliente.
    """
    try:
        base = _directorio_adjuntos().resolve()
        p = Path(str(path or "")).expanduser().resolve()
        if not p.is_file() or base not in p.parents:
            return None
        size = p.stat().st_size
        if size == 0 or size > _MAX_BYTES:
            return None
        return p
    except OSError:
        return None


def buscar_adjunto_por_nombre(nombre: str) -> Path | None:
    """Adjunto guardado más reciente cuyo archivo termina en `nombre`.

    Los archivos se guardan como <ts>_<i>_<nombre> (ver guardar_adjuntos):
    permite que la Web UI muestre el thumbnail de mensajes viejos que solo
    persistieron el nombre del archivo. Busca únicamente dentro del
    directorio de adjuntos (plano, sin recursión).
    """
    try:
        base = _directorio_adjuntos()
        objetivo = _RE_NOMBRE_SEGURO.sub("_", str(nombre or "")).lstrip(".")
        if not objetivo:
            return None
        candidatos = [p for p in base.iterdir()
                      if p.is_file() and p.name.endswith(f"_{objetivo}")]
        if not candidatos:
            return None
        return max(candidatos, key=lambda p: p.stat().st_mtime)
    except OSError:
        return None


def _salida_degenerada(texto: str) -> bool:
    """Detecta salidas degeneradas del modelo de visión (bug real con
    moondream): bucles de repetición tipo 'มามามา…' o string vacío.

    Heurísticas: (a) pocos caracteres distintos respecto a la longitud y
    (b) muy pocas palabras distintas entre muchas. Ambas apuntan a un loop
    de generación, no a una descripción real.
    """
    t = (texto or "").strip()
    if not t:
        return True
    if len(t) > 40 and len(set(t)) <= 8:
        return True
    palabras = t.split()
    if len(palabras) > 10 and len(set(palabras)) <= max(2, len(palabras) // 10):
        return True
    return False


def describir_imagen(ruta: Path, pregunta: str | None = None) -> str | None:
    """Describe una imagen usando el modelo de visión de Ollama.

    NOTA (bug verificado con moondream:latest): adjuntar la pregunta del
    usuario al prompt ("User request: …" o "Answer this question…") hace
    que moondream devuelva texto degenerado (loop tipo 'มามามา…') o un
    string vacío — el modelo espera prompts de captioning simples. La
    pregunta del usuario NO se pierde: sigue llegando al LLM principal,
    que la responde combinándola con esta descripción. Por eso el prompt
    acá es SIEMPRE caption simple, y si la salida es degenerada se
    reintenta una vez con el prompt mínimo antes de rendirse.

    La imagen viaja como THUMBNAIL (ver generar_thumbnail_b64), no a
    resolución completa: mismo contenido para el modelo de visión, una
    fracción del payload y del cómputo.
    """
    try:
        import requests
        from core.config.config_manager import get_config_manager
        from core.config.settings import OLLAMA_HOST
        cfg = get_config_manager()
        modelo = cfg.get("MODELO_VISION") or cfg.get("MODELO", "ornith:9b")
        imagen_b64 = generar_thumbnail_b64(ruta)
        if not imagen_b64:
            return None

        for prompt in ("Describe this image in detail.", "Describe this image."):
            r = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json={"model": modelo, "prompt": prompt, "images": [imagen_b64],
                      "stream": False, "options": {"temperature": 0.1, "num_ctx": 8192}},
                timeout=120,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            if "error" in data:
                return None
            resp = (data.get("response") or "").strip()
            if not _salida_degenerada(resp):
                return resp
        # Ambos intentos degenerados: mejor "sin descripción" que basura
        # inyectada en el contexto del LLM principal.
        return None
    except Exception:
        return None
def componer_orden(mensaje: str, metas: list[dict], errores: list[str],
                   describir: bool = True, inline: bool = False,
                   cupo_inline: int | None = None) -> str:
    """Mensaje + bloque [ADJUNTOS] con lo que el agente necesita:
    contenido de texto embebido, descripciones de imágenes y rutas en disco.

    describir=False saltea la descripción por visión (llamada al modelo,
    potencialmente decenas de segundos). El upload endpoint la usa porque su
    `orden` es preliminar: la visión corre UNA sola vez, al mandar el chat.

    inline=True marca las imágenes cuyo THUMBNAIL va incrustado en el
    mensaje del usuario del grafo (ver _preparar_orden): el modelo con
    visión nativa las analiza directamente y el caption de MODELO_VISION
    es redundante (el caller lo saltea con describir=False).

    cupo_inline (opcional): cuántos FRAMES de video van incrustados
    cuando inline=True — presupuesto compartido con las imágenes del
    usuario, que tienen prioridad (el caller _preparar_orden pasa el
    cupo exacto). Default: _MAX_IMAGENES_INLINE.
    """
    if not metas and not errores:
        return mensaje
    lineas = ["", "[ADJUNTOS DEL USUARIO]"]
    textos = [m for m in metas if m["kind"] == "text"]
    pdfs = [m for m in metas if m["kind"] == "pdf"]
    docxs = [m for m in metas if m["kind"] == "docx"]
    vectores = [m for m in metas if m["kind"] == "vector"]
    audios = [m for m in metas if m["kind"] == "audio"]
    videos = [m for m in metas if m["kind"] == "video"]
    imagenes = [m for m in metas if m["kind"] == "image"]
    otros = [m for m in metas if m["kind"] == "other"]
    for m in textos[:_MAX_TEXTO_EMBEBIDOS]:
        try:
            contenido = Path(m["path"]).read_text(encoding="utf-8", errors="replace")
        except OSError:
            contenido = "(no se pudo leer)"
        if len(contenido) > _MAX_TEXTO_INLINE:
            contenido = contenido[:_MAX_TEXTO_INLINE] + "\n…[truncado]"
        lineas.append(f"-- Archivo de texto '{m['name']}' (guardado en {m['path']}):\n" f"```\n{contenido}\n```")
    for m in textos[_MAX_TEXTO_EMBEBIDOS:]:
        lineas.append(f"-- Archivo de texto '{m['name']}' en {m['path']} " f"(leelo completo con fs_read).")
    for m in pdfs:
        txt = m.get("texto_extraido")
        if txt:
            if len(txt) > _MAX_TEXTO_INLINE:
                txt = txt[:_MAX_TEXTO_INLINE] + "\n…[truncado]"
            lineas.append(f"-- PDF '{m['name']}' (guardado en {m['path']}):\n" f"```\n{txt}\n```")
        else:
            lineas.append(f"-- PDF '{m['name']}' guardado en {m['path']} (no se pudo extraer texto).")
    for m in docxs:
        txt = m.get("texto_extraido")
        if txt:
            if len(txt) > _MAX_TEXTO_INLINE:
                txt = txt[:_MAX_TEXTO_INLINE] + "\n…[truncado]"
            lineas.append(f"-- DOCX '{m['name']}' (guardado en {m['path']}):\n" f"```\n{txt}\n```")
        else:
            lineas.append(f"-- DOCX '{m['name']}' guardado en {m['path']} (no se pudo extraer texto).")
    for m in vectores:
        txt = m.get("texto_extraido")
        if txt:
            if len(txt) > _MAX_TEXTO_INLINE:
                txt = txt[:_MAX_TEXTO_INLINE] + "\n…[truncado]"
            lineas.append(f"-- Vector '{m['name']}' (guardado en {m['path']}):\n" f"```\n{txt}\n```")
        else:
            lineas.append(f"-- Vector '{m['name']}' guardado en {m['path']} (no se pudo extraer texto).")
    for m in audios:
        txt = m.get("texto_extraido")
        if txt:
            if len(txt) > _MAX_TEXTO_INLINE:
                txt = txt[:_MAX_TEXTO_INLINE] + "\n…[truncado]"
            lineas.append(f"-- Audio '{m['name']}' (guardado en {m['path']}). Transcripción:\n" f"```\n{txt}\n```")
        else:
            lineas.append(f"-- Audio '{m['name']}' guardado en {m['path']} (no se pudo transcribir).")
    # Presupuesto de FRAMES que viajan inline, GLOBAL al mensaje (compartido
    # entre todos los videos; las imágenes del usuario ya ocuparon su parte
    # — el caller pasa el cupo exacto vía cupo_inline).
    marcar = ((cupo_inline if cupo_inline is not None
               else _MAX_IMAGENES_INLINE) if inline else 0)
    for m in videos:
        txt = m.get("texto_extraido")
        if txt:
            if len(txt) > _MAX_TEXTO_INLINE:
                txt = txt[:_MAX_TEXTO_INLINE] + "\n…[truncado]"
            encabezado = (f"-- Video '{m['name']}' (guardado en {m['path']}). "
                          f"Transcripción:\n```\n{txt}\n```")
        else:
            encabezado = (f"-- Video '{m['name']}' guardado en {m['path']} "
                          "(no se pudo transcribir).")
        # Análisis VISUAL por frames (feature "video_frames" de la matriz
        # de capacidades). El frame se extrae UNA vez y se reusa para el
        # caption (MODELO_VISION) y para el inline (ver _preparar_orden).
        # Si algo falla (sin ffmpeg, video corrupto, etc.) degrada al
        # comportamiento anterior: solo la transcripción.
        if _vision_video_activa():
            frames = _extraer_frames_video(Path(m["path"]))
            if frames:
                cuerpo: list[str] = []
                for f in frames:
                    desc = (describir_imagen(f["path"], pregunta=mensaje)
                            if describir else None)
                    linea = f"[t={_formato_mmss(f['t'])}]"
                    if desc:
                        linea += f" {desc}"
                    elif marcar > 0:
                        marcar -= 1
                        linea += " (frame adjunto como imagen; analizalo directamente)"
                    cuerpo.append(linea)
                if cuerpo:
                    encabezado += ("\n   Frames analizados (visión activa):\n   "
                                   + "\n   ".join(cuerpo))
        lineas.append(encabezado)
    for i, m in enumerate(imagenes):
        # inline=True → el thumbnail de esta imagen viaja incrustado en el
        # mensaje del usuario (ver _preparar_orden): el modelo la ve directo.
        adjunto = (" [thumbnail adjunto: analizala directamente]"
                   if inline and i < _MAX_IMAGENES_INLINE else "")
        if describir and i < _MAX_IMAGENES_DESCRITAS:
            desc = describir_imagen(m["path"], pregunta=mensaje)
            if desc:
                lineas.append(f"-- Imagen '{m['name']}' (guardada en {m['path']}){adjunto}. " f"Descripción por visión:\n{desc}")
                continue
            lineas.append(f"-- Imagen '{m['name']}' guardada en {m['path']}{adjunto} "
                          "(la descripción por visión no está disponible).")
        else:
            lineas.append(f"-- Imagen '{m['name']}' guardada en {m['path']}{adjunto}.")
    for m in otros:
        lineas.append(f"-- Archivo binario '{m['name']}' ({m['mime']}, {m['size']} bytes) " f"guardado en {m['path']}.")
    for e in errores:
        lineas.append(f"-- ⚠️ Adjunto descartado: {e}")
    base = mensaje.strip() or "Analizá los archivos adjuntos."
    return base + "\n" + "\n".join(lineas)
