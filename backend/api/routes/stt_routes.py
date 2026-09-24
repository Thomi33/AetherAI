"""
stt_routes.py — Dictado por voz para la Web UI (transcripción con whisper).

La Web UI graba audio en el navegador (MediaRecorder, webm/opus) y lo sube
acá; el backend lo transcribe con faster-whisper. Motores, en orden de
preferencia:

  1. Worker STT persistente (core/services/stt_service.py): faster-whisper
     en su venv aislado (~/whisper_aether_test/venv-stt o el path de config
     STT_VENV_PYTHON), con el modelo ya cargado en RAM. Es el mismo motor
     que el push-to-talk de la TUI.
  2. faster-whisper in-proceso (backend/core/attachments._transcribir_audio)
     para instalaciones donde faster-whisper vive en el venv principal.

Endpoints:
- GET  /api/stt/status     → disponibilidad del motor (sin levantar worker)
- POST /api/stt/transcribe → multipart 'file' (+ 'language' opcional)
"""

import asyncio
import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

router = APIRouter()

_MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB por grabación


def _stt_venv_disponible() -> bool:
    """True si existe el intérprete del venv de STT (sin levantar el worker)."""
    try:
        from core.services.stt_service import _stt_venv_python, STT_WORKER_SCRIPT
        return _stt_venv_python().exists() and Path(STT_WORKER_SCRIPT).exists()
    except Exception:  # noqa: BLE001
        return False


def _faster_whisper_local() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def _convertir_a_wav(ruta: Path) -> Path:
    """webm/opus/ogg → wav 16k mono con ffmpeg. faster-whisper también
    decodifica vía PyAV, pero las grabaciones del navegador (MediaRecorder)
    suelen traer timestamps rotos; la conversión previa es más robusta.
    Sin ffmpeg, se devuelve el original y el worker intenta con PyAV."""
    if ruta.suffix.lower() in (".wav", ".wave"):
        return ruta
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            salida = Path(tmp.name)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(ruta),
             "-ac", "1", "-ar", "16000", "-vn", str(salida)],
            check=True, capture_output=True, timeout=60,
        )
        return salida
    except Exception:  # noqa: BLE001 — el worker prueba con el original
        return ruta


def _idioma_y_hint() -> tuple[str | None, str]:
    """STT_LANGUAGE / STT_VOCAB_HINT de config (como hace stt_service)."""
    lang, hint = None, ""
    try:
        from core.config.config_manager import get_config_manager
        cfg = get_config_manager()
        lang = (cfg.get("STT_LANGUAGE", "") or "").strip() or None
        hint = (cfg.get("STT_VOCAB_HINT", "") or "").strip()
    except Exception:  # noqa: BLE001
        pass
    return lang, hint


def _run_transcripcion(ruta: Path, language: str) -> dict:
    """Transcribe con el mejor motor disponible. Siempre devuelve dict
    JSON-safe ({ok, text, language, engine} o {ok: False, error})."""
    extra: list[Path] = []
    try:
        ruta_wav = _convertir_a_wav(ruta)
        if ruta_wav != ruta:
            extra.append(ruta_wav)
        lang_cfg, vocab_hint = _idioma_y_hint()
        lang = language.strip() or lang_cfg

        # Motor 1: worker STT persistente (venv aislado, modelo warm)
        err_worker = "sin venv de STT"
        if _stt_venv_disponible():
            try:
                from core.services.stt_service import get_stt_worker
                res = get_stt_worker().transcribir(
                    str(ruta_wav), idioma=lang, timeout=180.0,
                    initial_prompt=vocab_hint or None)
                return {"ok": True, "text": res.get("text", ""),
                        "language": res.get("language", ""), "engine": "worker"}
            except Exception as e:  # noqa: BLE001
                err_worker = str(e)

        # Motor 2 (fallback): faster-whisper en este mismo proceso
        if _faster_whisper_local():
            from backend.core.attachments import _transcribir_audio
            texto = _transcribir_audio(ruta_wav)
            if texto is not None:
                return {"ok": True, "text": texto,
                        "language": lang or "", "engine": "local"}
            err_worker = f"{err_worker}; fallback local devolvió vacío"

        return {"ok": False,
                "error": "STT no disponible — requiere whisper o "
                         f"faster-whisper ({err_worker})"}
    finally:
        for p in extra:
            p.unlink(missing_ok=True)


@router.get("/stt/status")
async def stt_status():
    """Disponibilidad del dictado por voz (no levanta el worker ni el modelo)."""

    def _check() -> dict:
        worker = _stt_venv_disponible()
        local = _faster_whisper_local()
        return {
            "ok": True,
            "available": worker or local,
            "engine": "worker" if worker else ("local" if local else None),
            "detail": (
                "worker faster-whisper listo" if worker
                else ("faster-whisper local" if local
                      else "requiere whisper o faster-whisper "
                           "(venv STT o 'pip install faster-whisper')")
            ),
        }

    return await asyncio.get_running_loop().run_in_executor(None, _check)


@router.post("/stt/transcribe")
async def stt_transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
):
    """Transcribe una grabación de la Web UI (MediaRecorder webm/opus, wav…)."""
    data = await file.read()
    if not data:
        return JSONResponse({"ok": False, "error": "Audio vacío"}, status_code=400)
    if len(data) > _MAX_AUDIO_BYTES:
        return JSONResponse(
            {"ok": False,
             "error": f"Grabación muy larga (máx {_MAX_AUDIO_BYTES // (1024 * 1024)} MB)"},
            status_code=400)

    sufijo = Path(file.filename or "audio.webm").suffix or ".webm"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        return await asyncio.get_running_loop().run_in_executor(
            None, _run_transcripcion, tmp_path, language)
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)
