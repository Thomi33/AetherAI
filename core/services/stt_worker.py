#!/usr/bin/env python3
"""
stt_worker.py — Proceso persistente de transcripción (faster-whisper).

Corre DENTRO del venv aislado (~/Whisper_TTS/venv), NO en el
venv principal de Aether (crewai-env). Se comunica con el proceso padre
(core/services/stt_service.py) por stdin/stdout con un protocolo JSON-lines:

  → stdin:  {"cmd": "transcribe", "path": "/tmp/xxx.wav", "id": 1, "initial_prompt": "..."}
  ← stdout: {"id": 1, "ok": true, "text": "...", "language": "es", "language_probability": 0.98}
  ← stdout: {"id": 1, "ok": false, "error": "..."}

  → stdin:  {"cmd": "shutdown"}
  ← stdout: {"ok": true, "bye": true}

Carga el modelo UNA sola vez al arrancar y emite {"ready": true} (o
{"ready": false, "error": ...} si algo falla) antes de aceptar pedidos.
Mantener el modelo cargado en RAM entre transcripciones es el motivo real
de correr esto como servicio persistente en vez de un subprocess por
llamada: recargar "medium" en CPU/int8 toma varios segundos, inaceptable
en el loop de un push-to-talk.

Aislado en su propio venv porque ctranslate2 (dependencia de faster-whisper)
trae su propio set de wheels binarios que puede chocar con lo que ya usa
el venv principal (langgraph/langchain/ollama) — separar procesos evita ese
lío de dependencias a cambio de hablarles por stdin/stdout.
"""

from __future__ import annotations

import sys
import json


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> int:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        emit({"ready": False, "error": f"faster_whisper no instalado en este venv: {e}"})
        return 1

    modelo_size = sys.argv[1] if len(sys.argv) > 1 else "medium"
    device = sys.argv[2] if len(sys.argv) > 2 else "cpu"
    compute_type = sys.argv[3] if len(sys.argv) > 3 else "int8"

    try:
        model = WhisperModel(modelo_size, device=device, compute_type=compute_type)
    except Exception as e:  # noqa: BLE001 — cualquier fallo de carga debe llegar al padre
        emit({"ready": False, "error": f"Error cargando modelo '{modelo_size}': {type(e).__name__}: {e}"})
        return 1

    emit({"ready": True, "model": modelo_size, "device": device, "compute_type": compute_type})

    for linea in sys.stdin:
        linea = linea.strip()
        if not linea:
            continue
        try:
            req = json.loads(linea)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        cmd = req.get("cmd")

        if cmd == "shutdown":
            emit({"id": req_id, "ok": True, "bye": True})
            break

        if cmd != "transcribe":
            emit({"id": req_id, "ok": False, "error": f"cmd desconocido: {cmd}"})
            continue

        wav_path = req.get("path")
        idioma = req.get("language")  # None → autodetección
        # Texto de contexto que condiciona la decodificación hacia vocabulario
        # esperado (nombres propios como "Aether", "GitHub", "Ollama" que sin
        # esto whisper decodifica como la palabra en español más parecida
        # fonéticamente, arrastrando errores al resto de la frase). Lo arma
        # stt_service.py desde STT_VOCAB_HINT; None = sin condicionar.
        prompt_inicial = req.get("initial_prompt") or None

        if not wav_path:
            emit({"id": req_id, "ok": False, "error": "falta 'path' en el pedido"})
            continue

        try:
            segments, info = model.transcribe(
                wav_path,
                language=idioma,
                vad_filter=True,  # descarta silencios; conviene en grabaciones push-to-talk
                initial_prompt=prompt_inicial,
            )
            texto = " ".join(seg.text.strip() for seg in segments).strip()
            emit({
                "id": req_id,
                "ok": True,
                "text": texto,
                "language": info.language,
                "language_probability": round(float(info.language_probability), 3),
            })
        except Exception as e:  # noqa: BLE001 — un error de transcripción no debe matar el worker
            emit({"id": req_id, "ok": False, "error": f"{type(e).__name__}: {e}"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
