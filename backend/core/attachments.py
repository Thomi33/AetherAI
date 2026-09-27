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

_MAX_BYTES = 10 * 1024 * 1024      # 10 MB por adjunto
_MAX_TEXTO_INLINE = 4000           # chars por archivo de texto embebido
_MAX_TEXTO_EMBEBIDOS = 4           # cuántos archivos de texto se embeben
_MAX_IMAGENES_DESCRITAS = 2        # cuántas imágenes se describen con visión
_MAX_AUDIO_SEGUNDOS = 600          # tope de 10 min para transcribir
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
def _transcribir_audio(ruta: Path) -> str | None:
    """Transcribe audio/video usando faster-whisper.
    Extrae audio si es video, luego transcribe.
    """
    try:
        from faster_whisper import WhisperModel
        from core.config.config_manager import get_config_manager
        cfg = get_config_manager()
        modelo = cfg.get("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
        if ruta.suffix.lower() in {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
                audio_path = Path(tmp_audio.name)
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(ruta),
                    "-ac", "1", "-ar", "16000", "-vn", str(audio_path)
                ], check=True, capture_output=True, timeout=60)
                ruta_audio = audio_path
            except Exception:
                if audio_path.exists():
                    audio_path.unlink(missing_ok=True)
                return None
        else:
            ruta_audio = ruta
        model = WhisperModel(modelo, device="cpu", compute_type="int8")
        segments, info = model.transcribe(
            str(ruta_audio),
            language=None,  # auto-detect
            vad_filter=True,
            max_initial_timestamp=_MAX_AUDIO_SEGUNDOS,
        )
        texto = " ".join(seg.text for seg in segments).strip()
        if ruta_audio != ruta:
            ruta_audio.unlink(missing_ok=True)
        return texto or None
    except Exception:
        return None
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


_CACHE_MODELO_VISION: dict[str, bool] = {}


def modelo_admite_imagenes(modelo: str) -> bool:
    """True si el modelo de Ollama acepta imágenes (visión nativa).

    Consulta POST /api/show y mira `capabilities` (formato nuevo) o
    `projector_info` (encoder CLIP); cacheado por host+modelo. Un modelo
    text-only rechaza mensajes con "images" (HTTP 400), así que el
    thumbnail inline SOLO se activa cuando esto devuelve True — si no,
    el análisis de la imagen cae al caption de MODELO_VISION.
    """
    import requests
    from core.config.settings import OLLAMA_HOST
    clave = f"{OLLAMA_HOST}|{modelo}"
    if clave in _CACHE_MODELO_VISION:
        return _CACHE_MODELO_VISION[clave]
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/show",
                          json={"model": modelo}, timeout=10)
        data = r.json() if r.status_code == 200 else {}
    except Exception:
        return False  # sin info confiable: no arriesgar un 400
    caps = data.get("capabilities") or []
    projector = data.get("projector_info") or {}
    admite = ("vision" in caps
              or bool(projector.get("clip.has_vision_encoder")))
    _CACHE_MODELO_VISION[clave] = admite
    return admite


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
            metas.append({
                "name": nombre, "mime": mime,
                "kind": _kind(nombre, mime),
                "path": str(p), "size": size,
            })
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
                   describir: bool = True, inline: bool = False) -> str:
    """Mensaje + bloque [ADJUNTOS] con lo que el agente necesita:
    contenido de texto embebido, descripciones de imágenes y rutas en disco.

    describir=False saltea la descripción por visión (llamada al modelo,
    potencialmente decenas de segundos). El upload endpoint la usa porque su
    `orden` es preliminar: la visión corre UNA sola vez, al mandar el chat.

    inline=True marca las imágenes cuyo THUMBNAIL va incrustado en el
    mensaje del usuario del grafo (ver _preparar_orden): el modelo con
    visión nativa las analiza directamente y el caption de MODELO_VISION
    es redundante (el caller lo saltea con describir=False).
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
    for m in videos:
        txt = m.get("texto_extraido")
        if txt:
            if len(txt) > _MAX_TEXTO_INLINE:
                txt = txt[:_MAX_TEXTO_INLINE] + "\n…[truncado]"
            lineas.append(f"-- Video '{m['name']}' (guardado en {m['path']}). Transcripción:\n" f"```\n{txt}\n```")
        else:
            lineas.append(f"-- Video '{m['name']}' guardado en {m['path']} (no se pudo transcribir).")
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
