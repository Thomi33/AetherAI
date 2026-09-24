"""
Captura de pantalla y análisis de visión con Ollama.

CORRECCIÓN vs versión anterior:
- Usa el modelo principal configurado; Ornith 1.5 soporta visión nativa
- Mejor reporte de errores: muestra stderr de grim si falla
"""
import subprocess
import os
import base64
import json
import time
import requests

from core.config.settings import OLLAMA_HOST
from core.agent.model_policy import ModelDecisionContext, choose_model, record_model_latency


def capturar_pantalla() -> bytes:
    """Captura la pantalla completa y devuelve los bytes PNG para un consumidor."""
    resultado = subprocess.run(
        ["grim"],
        capture_output=True,
        timeout=10,
    )
    if resultado.returncode != 0:
        stderr = resultado.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"No pude capturar la pantalla: {stderr or 'grim falló'}")
    return resultado.stdout


def ver_pantalla(
    pregunta: str = "¿Qué ves en esta pantalla?",
    crop: tuple[int, int, int, int] | None = None,
) -> str:
    """
    Toma screenshot del monitor activo en Hyprland.
    Lo analiza con el modelo de visión local en Ollama.

    Requiere:
    - grim instalado (pacman -S grim)
    - Un modelo multimodal en Ollama (el modelo principal configurado)
    """
    screenshot = "/tmp/aether_vision.png"

    try:
        # Detectar monitor activo en Hyprland
        _activo = None
        try:
            _mon = subprocess.run(
                ["hyprctl", "monitors", "-j"],
                capture_output=True, text=True, timeout=5
            )
            if _mon.returncode == 0 and _mon.stdout.strip():
                _mons   = json.loads(_mon.stdout)
                _activo = next((m["name"] for m in _mons if m.get("focused")), None)
        except Exception:
            pass  # Sin Hyprland o hyprctl → captura pantalla completa

        # Capturar screenshot con grim
        if crop is not None:
            x, y, width, height = (int(value) for value in crop)
            if width <= 0 or height <= 0:
                return "El recorte de pantalla debe tener ancho y alto positivos."
            _cmd = ["grim", "-g", f"{x},{y} {width}x{height}", screenshot]
        else:
            _cmd = ["grim", "-o", _activo, screenshot] if _activo else ["grim", screenshot]
        resultado = subprocess.run(_cmd, capture_output=True, timeout=10)

        if resultado.returncode != 0:
            stderr = resultado.stderr.decode("utf-8", errors="replace").strip()
            detalle = f" Detalle: {stderr}" if stderr else " (sin detalles, ¿grim instalado?)"
            return (
                f"No pude capturar la pantalla, compa.{detalle}\n"
                f"Revisá: pacman -S grim"
            )

        # Leer imagen y convertir a base64
        with open(screenshot, "rb") as f:
            imagen_b64 = base64.b64encode(f.read()).decode("utf-8")

        # Enviar al modelo de visión con inyección de contexto para mitigar la resistencia/alineamiento de LLaVA
        prompt_ajustado = (
            "Eres un asistente técnico experto en análisis de interfaces de usuario. "
            "La siguiente imagen es una captura de pantalla del escritorio del sistema operativo del usuario. "
            "El sistema operativo es Linux (Arch Linux), usando Hyprland como compositor de Wayland. "
            "NO es macOS, NO es Windows. No asumas estilos de barra superior, dock o iconos típicos de macOS: "
            "describí únicamente lo que realmente ves en la imagen, identificando aplicaciones, ventanas y barras "
            "según su apariencia real, sin presuponer un sistema operativo distinto al indicado. "
            "No contiene personas reales ni datos personales de nadie. Describe de forma objetiva, técnica y detallada "
            "las ventanas, el texto, código, barras de estado o aplicaciones que se encuentran visibles.\n\n"
            f"Pregunta del usuario: {pregunta}"
        )

        from core.config.config_manager import get_config_manager
        modelo_principal = get_config_manager().get("MODELO", "ornith:9b")
        decision_modelo = choose_model(ModelDecisionContext(
            task_kind="vision",
            requested_model=modelo_principal,
            prompt_chars=len(prompt_ajustado),
            context_size=8192,
        ))

        payload = {
            "model":  decision_modelo.model,
            "prompt": prompt_ajustado,
            "images": [imagen_b64],
            "stream": False,   # ← antes era True
            "options": {"temperature": 0.1, "num_ctx": 8192},
                }
    
        policy_start = time.time()
        try:
            r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
        finally:
            record_model_latency(decision_modelo, int((time.time() - policy_start) * 1000))

        if r.status_code == 200:
            data = r.json()
            # Ollama a veces reporta error dentro del 200
            if "error" in data:
                return (
                    f"El modelo de visión reportó un error: {data['error']}\n"
                    f"Revisá que '{modelo_principal}' soporte imágenes (ollama list)."
                )
            return data.get("response", "Sin respuesta de mi modelo de visión.")

        # Error HTTP
        try:
            detalle_http = r.json().get("error", r.text[:200])
        except Exception:
            detalle_http = r.text[:200]
        return (
            f"Error HTTP {r.status_code} al consultar visión Ollama.\n"
            f"Detalle: {detalle_http}\n"
            f"Modelo configurado: '{modelo_principal}' — ¿está instalado? (ollama list)"
        )

    except Exception as e:
        return f"Error en mi sistema de visión, compa: {e}"

    finally:
        if os.path.exists(screenshot):
            try:
                os.unlink(screenshot)
            except Exception:
                pass
