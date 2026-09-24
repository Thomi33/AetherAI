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

Todo best-effort: un adjunto roto nunca tira el request completo.
"""
from __future__ import annotations

import base64
import binascii
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
    """
    try:
        import requests
        from core.config.config_manager import get_config_manager
        from core.config.settings import OLLAMA_HOST
        cfg = get_config_manager()
        modelo = cfg.get("MODELO_VISION") or cfg.get("MODELO", "ornith:9b")
        imagen_b64 = base64.b64encode(Path(ruta).read_bytes()).decode("utf-8")

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
                   describir: bool = True) -> str:
    """Mensaje + bloque [ADJUNTOS] con lo que el agente necesita:
    contenido de texto embebido, descripciones de imágenes y rutas en disco.

    describir=False saltea la descripción por visión (llamada al modelo,
    potencialmente decenas de segundos). El upload endpoint la usa porque su
    `orden` es preliminar: la visión corre UNA sola vez, al mandar el chat.
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
        if describir and i < _MAX_IMAGENES_DESCRITAS:
            desc = describir_imagen(m["path"], pregunta=mensaje)
            if desc:
                lineas.append(f"-- Imagen '{m['name']}' (guardada en {m['path']}). " f"Descripción por visión:\n{desc}")
                continue
            lineas.append(f"-- Imagen '{m['name']}' guardada en {m['path']} "
                          "(la descripción por visión no está disponible).")
        else:
            lineas.append(f"-- Imagen '{m['name']}' guardada en {m['path']}.")
    for m in otros:
        lineas.append(f"-- Archivo binario '{m['name']}' ({m['mime']}, {m['size']} bytes) " f"guardado en {m['path']}.")
    for e in errores:
        lineas.append(f"-- ⚠️ Adjunto descartado: {e}")
    base = mensaje.strip() or "Analizá los archivos adjuntos."
    return base + "\n" + "\n".join(lineas)
