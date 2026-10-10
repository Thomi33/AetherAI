"""
capabilities.py — Matriz de capacidades de Aether (fuente ÚNICA de verdad).

Centraliza TODA decisión de "¿qué puede hacer Aether con este modelo?".
Ningún otro módulo debe duplicar esta lógica; los wrappers existentes
delegan acá:

  - backend/core/attachments.modelo_admite_imagenes   → delega acá
  - backend/core/aether_service._modelo_con_reasoning → delega acá
  - core/agent/graph_nodes.node_vision                → consulta acá

Dos capas, no confundir (ajuste #1 del pedido):

  1. Capacidades DECLARADAS por Ollama — capacidades_modelo():
     POST /api/show del modelo. Ollama >= 0.12 reporta `capabilities`
     como LISTA de strings; versiones intermedias usaban un dict con
     `multimodal`. Para visión hay fallback a `projector_info` (encoder
     CLIP del mmproj adjunto).

  2. Capacidades EFECTIVAS de Aether — funciones_disponibles():
     cruza lo declarado con el entorno real (ffmpeg / faster-whisper /
     grim instalados) y la configuración (MODELO, MODELO_VISION).

Sobre `thinking` (verificado contra la Ollama instalada, v0.40.0,
09/10/2026 — ajuste #3):
  - deepseek-r1:14b  → ['tools', 'thinking', 'completion']
  - qwen3:4b         → ['completion', 'tools', 'thinking']
  - gpt-oss:20b      → ['completion', 'tools', 'thinking']
  - llama3.1:8b      → ['completion', 'tools']   (sin thinking)
  - Gemma-4-E4B (hf.co, MODELO actual) → ['tools', 'completion',
    'vision', 'audio'] — NO declara thinking.
Ollama representa la capacidad como el string "thinking" en la lista
`capabilities`. Es una capacidad DECLARADA por el manifest (el modelo
emite razonamiento nativo); NO es razonamiento "efectivamente observado"
en runtime — eso solo se sabría ejecutando una inferencia. Es lo mejor
disponible sin gastar un turno y es lo que la Web UI necesita para
decidir si mostrar el panel de razonamiento. Cuando Ollama no reporta
capabilities (caída, modelo no instalado, versión vieja), se recurre al
heurístico por nombre — el mismo fallback que existía en aether_service
(conservado).
"""
from __future__ import annotations

import importlib.util
import shutil

# Cache de capacidades declaradas. key: f"{host}|{modelo}", value: dict
# normalizado de capacidades. Los FALLOS no se cachean: el próximo
# llamado reintenta (mismo contrato que tenía modelo_admite_imagenes en
# attachments: ver el test "retry-qa").
# NOTA: backend/core/attachments._CACHE_MODELO_VISION es ALIAS de este
# dict, para que los fixtures de tests existentes (que hacen .clear())
# sigan limpiando el cache real tras la delegación.
_CACHE_CAPACIDADES: dict[str, dict] = {}

# Capacidades que Ollama puede declarar (formato normalizado).
_CAPS_NORMALES = ("vision", "tools", "thinking", "embedding", "audio",
                  "completion")

# Fallback heurístico de razonamiento POR NOMBRE: solo se usa cuando
# Ollama no reporta capabilities. Conservado tal cual estaba en
# aether_service._modelo_con_reasoning.
_HEURISTICA_REASONING = ("gemma", "qwen", "deepseek", "r1", "reasoning")


def _caps_desconocidas() -> dict:
    """Dict de capacidades cuando no se pudo consultar a Ollama.
    Con desconocido=True el resto de las claves NO es fiable."""
    return {**{k: False for k in _CAPS_NORMALES}, "desconocido": True}


def _host_ollama() -> str:
    """OLLAMA_HOST actual: config_manager primero (hot reload), settings
    como fallback; nunca revienta."""
    try:
        from core.config.config_manager import get_config_manager
        host = get_config_manager().get("OLLAMA_HOST")
        if host:
            return str(host)
    except Exception:  # noqa: BLE001
        pass
    try:
        from core.config.settings import OLLAMA_HOST
        return OLLAMA_HOST or "http://localhost:11434"
    except Exception:  # noqa: BLE001
        return "http://localhost:11434"


def _modelo_principal() -> str:
    """MODELO actual: config_manager primero (dinámico), settings como
    fallback; nunca revienta."""
    try:
        from core.config.config_manager import get_config_manager
        m = get_config_manager().get("MODELO")
        if m:
            return str(m)
    except Exception:  # noqa: BLE001
        pass
    try:
        from core.config.settings import MODELO
        return MODELO or "ornith:9b"
    except Exception:  # noqa: BLE001
        return "ornith:9b"


def _modelo_vision_cfg() -> str:
    """MODELO_VISION configurado, con el mismo fallback que usa
    attachments.describir_imagen (MODELO_VISION or MODELO)."""
    try:
        from core.config.config_manager import get_config_manager
        m = get_config_manager().get("MODELO_VISION")
        if m:
            return str(m)
    except Exception:  # noqa: BLE001
        pass
    return _modelo_principal()


def _normalizar_capabilities(data: dict) -> dict | None:
    """Normaliza la sección `capabilities` de /api/show a un dict de
    claves fijas. Acepta ambos formatos que usó Ollama:
      - lista (>= 0.12): ["completion", "tools", "vision", ...]
      - dict intermedio: {"completion": true, "multimodal": ["vision"]}
    Devuelve None si la respuesta no declara nada usable (sin
    capabilities ni projector_info con encoder de visión): eso se
    trata como "desconocido" en capacidades_modelo.
    """
    caps = data.get("capabilities")
    resultado = {k: False for k in _CAPS_NORMALES}
    if isinstance(caps, list):
        for k in _CAPS_NORMALES:
            resultado[k] = k in caps
    elif isinstance(caps, dict):
        for k in _CAPS_NORMALES:
            resultado[k] = bool(caps.get(k))
        multimodal = caps.get("multimodal") or []
        if isinstance(multimodal, list):
            for k in ("vision", "audio", "embedding"):
                if k in multimodal:
                    resultado[k] = True
    else:
        # Sin `capabilities` en la respuesta: solo projector_info puede
        # aportar algo (visión). Si tampoco hay, es desconocido.
        projector = data.get("projector_info")
        if (isinstance(projector, dict)
                and projector.get("clip.has_vision_encoder")):
            resultado["vision"] = True
            return resultado
        return None
    # Fallback CLIP: modelos con mmproj (encoder de visión) aunque la
    # lista capabilities no lo declare.
    projector = data.get("projector_info") or {}
    if (isinstance(projector, dict)
            and projector.get("clip.has_vision_encoder")):
        resultado["vision"] = True
    return resultado


def capacidades_modelo(modelo: str) -> dict:
    """Capacidades DECLARADAS por Ollama para un modelo (cacheado por
    host+modelo; los fallos NO se cachean: se reintenta).

    Devuelve {vision, tools, thinking, embedding, audio, completion,
    desconocido}. desconocido=True → no se pudo consultar (Ollama
    caída, HTTP != 200, JSON inválido, modelo ausente o versión sin
    metadata) y el resto de las claves NO es fiable. No mutar el dict
    devuelto: el cacheado es el mismo objeto.
    """
    modelo = str(modelo or "").strip()
    if not modelo:
        return _caps_desconocidas()
    host = _host_ollama()
    clave = f"{host}|{modelo}"
    cacheado = _CACHE_CAPACIDADES.get(clave)
    if cacheado is not None:
        return cacheado
    try:
        import requests
        r = requests.post(f"{host}/api/show",
                          json={"model": modelo}, timeout=10)
        if r.status_code != 200:
            return _caps_desconocidas()   # fallo: sin cachear
        data = r.json()
        if not isinstance(data, dict):
            return _caps_desconocidas()   # fallo: sin cachear
    except Exception:  # noqa: BLE001 — sin info confiable no decidir nada
        return _caps_desconocidas()       # fallo: sin cachear
    caps = _normalizar_capabilities(data)
    if caps is None:
        return _caps_desconocidas()       # sin metadata: sin cachear
    resultado = {**caps, "desconocido": False}
    _CACHE_CAPACIDADES[clave] = resultado
    return resultado


def modelo_admite_imagenes(modelo: str) -> bool:
    """True si el modelo acepta imágenes (visión nativa declarada).
    Sin info confiable → False: un modelo text-only rechaza mensajes
    con "images" (HTTP 400); no arriesgar.
    """
    return bool(capacidades_modelo(modelo).get("vision"))


def modelo_con_reasoning(modelo: str | None = None) -> bool:
    """True si el modelo declara razonamiento nativo (thinking).

    FUENTE: capacidades REALES de Ollama (/api/show → capabilities).
    El heurístico por nombre SOLO se usa cuando no se pudo consultar
    (caída / versión vieja / modelo inexistente) — fallback conservado
    de aether_service._modelo_con_reasoning.

    OJO: `thinking` es una capacidad DECLARADA por el manifest, no
    razonamiento efectivamente observado en runtime (ver docstring del
    módulo).
    """
    modelo = str(modelo or _modelo_principal())
    caps = capacidades_modelo(modelo)
    if not caps.get("desconocido"):
        return bool(caps.get("thinking"))
    nombre = modelo.lower()
    return any(x in nombre for x in _HEURISTICA_REASONING)


def _ffmpeg_disponible() -> bool:
    return shutil.which("ffmpeg") is not None


def _whisper_disponible() -> bool:
    """faster-whisper instalado. Detectar el spec NO importa el paquete
    pesado (en attachments el import es perezoso dentro de la función)."""
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except Exception:  # noqa: BLE001
        return False


def _grim_disponible() -> bool:
    return shutil.which("grim") is not None


def funciones_disponibles(modelo: str | None = None,
                          modelo_vision: str | None = None) -> dict[str, bool]:
    """Matriz de capacidades EFECTIVAS de Aether con la config y el
    entorno actuales. Fuente única de verdad para decidir qué features
    se activan en los pipelines:

      imagen_inline:       el MODELO principal ve imágenes nativamente →
                           los thumbnails viajan incrustados en su
                           propio mensaje del grafo.
      caption_fallback:    MODELO_VISION ve imágenes → describir_imagen()
                           puede generar captions (para modelos
                           text-only).
      video_transcripcion: ffmpeg extrae el audio + faster-whisper
                           transcribe (pipeline actual de attachments).
      video_frames:        ffmpeg extrae frames y ALGÚN modelo con
                           visión los describe/ve (pipeline de frames).
      vision_pantalla:     grim captura + el MODELO principal ve la
                           pantalla (ver_pantalla usa el MODELO
                           principal, NO MODELO_VISION — ver
                           core/tools/vision.py).
    """
    modelo = str(modelo or _modelo_principal())
    modelo_vision = str(modelo_vision or _modelo_vision_cfg())
    caps_principal = capacidades_modelo(modelo)
    caps_vision = (caps_principal if modelo_vision == modelo
                   else capacidades_modelo(modelo_vision))
    ffmpeg = _ffmpeg_disponible()
    vision_alguna = (bool(caps_principal.get("vision"))
                     or bool(caps_vision.get("vision")))
    return {
        "imagen_inline":       bool(caps_principal.get("vision")),
        "caption_fallback":    bool(caps_vision.get("vision")),
        "video_transcripcion": ffmpeg and _whisper_disponible(),
        "video_frames":        ffmpeg and vision_alguna,
        "vision_pantalla":     (_grim_disponible()
                               and bool(caps_principal.get("vision"))),
    }
