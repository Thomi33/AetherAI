"""
stt_service.py — Puente hacia el worker de STT (faster-whisper) que corre
aislado en ~/whisper_aether_test/venv-stt (ver stt_worker.py).

Responsabilidades:
  1. SttWorker: mantiene vivo el subproceso del worker (stdin/stdout JSON-lines)
     entre turnos, para no pagar el costo de recargar 'medium' en cada orden.
  2. GrabadorAudio: graba push-to-talk con `arecord` (alsa-utils, sin
     dependencias Python nuevas en el venv principal — ni PyAudio ni
     sounddevice).
  3. transcribir_wav(): función de conveniencia que junta ambas cosas para
     el caller (engine_bridge / TUI).

Config relevante (ver core/config/config_manager.py):
  STT_ENABLED           — activa/desactiva el dictado por voz (default True)
  STT_LANGUAGE          — idioma fijo para faster-whisper ("es", "en", ...);
                           "" o None → autodetección por turno
  STT_VENV_PYTHON       — override del path al intérprete del venv de STT
  AUDIO_INPUT_MATCH     — texto que identifica la placa ALSA a usar; no se
                           usa el micrófono predeterminado como fallback
  STT_VOCAB_HINT         — lista de palabras/nombres propios frecuentes que se
                           pasan como initial_prompt a whisper para mejorar el
                           reconocimiento de vocabulario específico del proyecto
                           (nombres como "Aether", "GitHub", que sin esto se
                           confunden con la palabra en español más parecida
                           fonéticamente, arrastrando errores al resto de la
                           frase). "" = sin condicionar (comportamiento viejo).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from core.services.audio_input import AudioInputNoDisponible, resolver_fuente_pulse

# ─────────────────────────────────────────────────────────────────────
# Configuración por defecto (puede pisarse vía ConfigManager, ver abajo)
# ─────────────────────────────────────────────────────────────────────
_STT_VENV_PYTHON_DEFAULT = Path.home() / "whisper_aether_test" / "venv-stt" / "bin" / "python"
STT_WORKER_SCRIPT = Path(__file__).parent / "stt_worker.py"
STT_MODEL_SIZE = "medium"
STT_DEVICE = "cpu"
STT_COMPUTE_TYPE = "int8"

AUDIO_TMP_DIR = Path(tempfile.gettempdir()) / "aether_stt"

# Header WAV de arecord = 44 bytes; si el archivo no supera eso, no hubo
# audio real (usuario soltó la tecla casi instantáneo, o arecord no arrancó).
_WAV_HEADER_BYTES = 44


def _stt_venv_python() -> Path:
    """
    Resuelve el intérprete del venv de STT, con posibilidad de override
    desde config.json (útil si el usuario mueve el venv de lugar).
    """
    try:
        from core.config.config_manager import get_config_manager
        override = get_config_manager().get("STT_VENV_PYTHON", "")
        if override:
            return Path(override).expanduser()
    except Exception:
        pass
    return _STT_VENV_PYTHON_DEFAULT


class SttNoDisponible(RuntimeError):
    """El venv de STT, el worker, o faster-whisper no están disponibles."""


class SttWorker:
    """
    Envoltorio del subproceso persistente de stt_worker.py. Se arranca lazy
    en la primera transcripción y se reusa entre turnos (igual criterio que
    el motor LangGraph en engine_bridge: un proceso por ejecución de la TUI).
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._siguiente_id = 0
        self._listo = False

    def _asegurar_proceso(self) -> None:
        if self._proc is not None and self._proc.poll() is None and self._listo:
            return

        python_bin = _stt_venv_python()
        if not python_bin.exists():
            raise SttNoDisponible(
                f"No se encontró el intérprete de STT en {python_bin}. "
                f"¿Existe el venv ~/whisper_aether_test/venv-stt? "
                f"(podés apuntar a otro con /set STT_VENV_PYTHON <ruta>)"
            )

        self._proc = subprocess.Popen(
            [str(python_bin), str(STT_WORKER_SCRIPT), STT_MODEL_SIZE, STT_DEVICE, STT_COMPUTE_TYPE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._listo = False

        # Esperar la línea {"ready": true/false} con timeout — cargar
        # 'medium' en CPU/int8 puede tardar varios segundos la primera vez.
        primera_linea = None
        deadline = time.time() + 45
        while time.time() < deadline:
            if self._proc.stdout is None:
                break
            linea = self._proc.stdout.readline()
            if linea:
                primera_linea = linea
                break
            if self._proc.poll() is not None:
                break

        if not primera_linea:
            stderr = self._proc.stderr.read() if self._proc.stderr else ""
            raise SttNoDisponible(f"El worker de STT no arrancó a tiempo. stderr: {stderr[:500]}")

        try:
            data = json.loads(primera_linea)
        except json.JSONDecodeError:
            raise SttNoDisponible(f"Respuesta inesperada del worker de STT: {primera_linea[:200]!r}")

        if not data.get("ready"):
            raise SttNoDisponible(f"El worker de STT falló al cargar el modelo: {data.get('error')}")

        self._listo = True

    def transcribir(self, wav_path: str, idioma: str | None = None, timeout: float = 60.0, initial_prompt: str | None = None) -> dict:
        """
        Envía el path de un WAV al worker y espera la transcripción.
        Devuelve {"text", "language", "language_probability"}.
        Lanza SttNoDisponible si el worker no pudo arrancar/cargar el modelo
        o si la transcripción falla.
        """
        with self._lock:
            self._asegurar_proceso()
            self._siguiente_id += 1
            req_id = self._siguiente_id

            payload: dict = {"cmd": "transcribe", "path": wav_path, "id": req_id}
            if idioma:
                payload["language"] = idioma
            if initial_prompt:
                payload["initial_prompt"] = initial_prompt

            assert self._proc and self._proc.stdin and self._proc.stdout
            try:
                self._proc.stdin.write(json.dumps(payload) + "\n")
                self._proc.stdin.flush()
            except BrokenPipeError:
                self._listo = False
                raise SttNoDisponible("El worker de STT se cerró inesperadamente (pipe roto).")

            deadline = time.time() + timeout
            while time.time() < deadline:
                linea = self._proc.stdout.readline()
                if not linea:
                    if self._proc.poll() is not None:
                        stderr = self._proc.stderr.read() if self._proc.stderr else ""
                        self._listo = False
                        raise SttNoDisponible(f"El worker de STT murió durante la transcripción: {stderr[:500]}")
                    continue
                try:
                    data = json.loads(linea)
                except json.JSONDecodeError:
                    continue
                if data.get("id") == req_id:
                    if not data.get("ok"):
                        raise SttNoDisponible(f"Error transcribiendo: {data.get('error')}")
                    return data

            raise SttNoDisponible("Timeout esperando la transcripción del worker de STT.")

    def cerrar(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                    self._proc.stdin.flush()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
        self._proc = None
        self._listo = False


_worker: Optional[SttWorker] = None
_worker_lock = threading.Lock()


def get_stt_worker() -> SttWorker:
    global _worker
    if _worker is None:
        with _worker_lock:
            if _worker is None:
                _worker = SttWorker()
    return _worker


class GrabadorAudio:
    """
    Grabación push-to-talk vía `arecord` (alsa-utils). No depende de
    PyAudio/sounddevice — arecord ya viene con cualquier instalación
    estándar de Arch y evita meter otra dependencia binaria al venv
    principal de Aether.
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._wav_path: Optional[Path] = None
        self._device: Optional[str] = None
        self._audio_env: Optional[dict[str, str]] = None

    @property
    def grabando(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def iniciar(self) -> Path:
        if self.grabando:
            raise RuntimeError("Ya hay una grabación de voz en curso.")

        AUDIO_TMP_DIR.mkdir(parents=True, exist_ok=True)
        self._wav_path = AUDIO_TMP_DIR / f"orden_{int(time.time() * 1000)}.wav"

        try:
            source = resolver_fuente_pulse()
            self._device = "pulse"
            self._audio_env = {**os.environ, "PULSE_SOURCE": source}
        except AudioInputNoDisponible as exc:
            self._wav_path = None
            raise SttNoDisponible(str(exc)) from exc

        # 16kHz mono 16-bit: formato nativo que espera faster-whisper/Whisper,
        # evita que el worker tenga que resamplear en cada transcripción.
        try:
            self._proc = subprocess.Popen(
                [
                    "arecord", "-q", "-D", self._device,
                    "-f", "S16_LE", "-r", "16000", "-c", "1", str(self._wav_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                env=self._audio_env,
            )
            print(f"[STT] Grabando desde la fuente PipeWire seleccionada ({source}).")
        except FileNotFoundError:
            self._proc = None
            self._wav_path = None
            self._device = None
            self._audio_env = None
            raise SttNoDisponible("No se encontró 'arecord' (paquete alsa-utils). Instalalo para usar dictado por voz.")

        return self._wav_path

    def detener(self) -> Optional[Path]:
        """Detiene la grabación y devuelve el path del WAV si quedó audio real, o None."""
        if not self._proc:
            return None
        self._proc.terminate()
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._proc = None

        wav = self._wav_path
        self._wav_path = None
        self._device = None
        self._audio_env = None
        if wav and wav.exists() and wav.stat().st_size > _WAV_HEADER_BYTES:
            return wav
        return None

    def descartar(self) -> None:
        """Corta la grabación en curso sin devolver nada (cancelar dictado)."""
        self.detener()


def transcribir_wav(wav_path: Path, idioma: str | None = "es") -> str:
    """
    Función de conveniencia de alto nivel: entrega el texto transcripto o
    levanta SttNoDisponible.

    idioma="es" por default: fijarlo evita que faster-whisper pierda tiempo
    en detección automática y reduce errores cuando el usuario alterna es/en a
    mitad de frase (code-switching, ya manejado igual del lado de
    faster-whisper con vad_filter). Pasar idioma=None para autodetección.

    Arma el initial_prompt desde STT_VOCAB_HINT (config) automáticamente —
    el caller no necesita saber nada de esto, solo se beneficia del mejor
    reconocimiento de nombres propios del proyecto (Aether, GitHub, etc.).
    """
    worker = get_stt_worker()

    vocab_hint = ""
    try:
        from core.config.config_manager import get_config_manager
        vocab_hint = get_config_manager().get("STT_VOCAB_HINT", "") or ""
    except Exception:
        pass

    resultado = worker.transcribir(str(wav_path), idioma=idioma, initial_prompt=vocab_hint or None)
    return resultado.get("text", "")
