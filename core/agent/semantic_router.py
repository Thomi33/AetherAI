"""
semantic_router.py — Routing semántico OPT-IN por embeddings para Aether.

Alternativa de baja latencia al agent loop para el routing de las 10 tools.
Activación explícita (nada de esto corre con la configuración por defecto):

    PLANNER_ROUTER = "semantic"        (config.json / config.local.json)
    PLANNER_ROUTER_SEMANTIC_MODEL = "embeddinggemma:latest"
    PLANNER_ROUTER_SEMANTIC_MARGIN = 0.03        # umbral de confianza

    # o por sesión, sin persistir (lo usa `python run.py --planner semantic`):
    AETHER_PLANNER_ROUTER=semantic

Arquitectura (dentro de node_planner):

    planner → checks deterministas (sin cambios)
            → [OPT-IN] ruta_semantica()
                 ├─ margin >= umbral + tool enrutable directo → plan 1 paso → tool
                 └─ margin < umbral | tool fs_* | error Ollama → None
            → agent loop (comportamiento por defecto, SIN CAMBIOS)

Benchmark de referencia (/mnt/nvme/data/semantic-router-bench, test v2 holdout):
    embeddinggemma + prototype: accuracy 87.21%, macro 87.64%, top-3 97.7%
    (baseline Laya v2 fine-tuning: 75.58% / 76.47%). Estrategia prototype:
    coseno contra el centroide (media re-normalizada) de los embeddings de
    los 394 ejemplos de train de v2 (ver semantic_data/README.md).

Decisiones de diseño:
- SOLO decide la intención (tool). La ejecución pasa por el nodo real de la
  tool (plan_executor → node_*), que conserva TODAS sus validaciones y
  confirmaciones: el runtime sigue siendo la autoridad.
- Fallback silencioso y total: margin bajo, tools que necesitan args
  estructurados (fs_write/fs_read/fs_mkdir/fs_list requieren 'path' que el
  router no puede extraer), o cualquier error de Ollama/modelo → el planner
  sigue exactamente como antes (agent loop). Nunca rompe un turno.
- Python puro (numpy no es dependencia del core): coseno de 768 dims en
  Python es trivial para 10 prototipos.
- Métricas opcionales vía AETHER_BENCH_INSTRUMENT=1 (mismo mecanismo que
  el resto del core, ver core/config/settings.py): JSONL en
  ~/.aether/bench_metrics.jsonl, sin ruido en el modo normal.
- Prototipos cacheados en ~/.aether/semantic_router/ (por modelo); se
  recalculan si cambia el modelo o el archivo de ejemplos.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import ollama

from core.config import get_config_manager
from core.config.settings import BENCH_INSTRUMENT, BENCH_LOG_PATH

# ══════════════════════════════════════════════════════════════════════
# Constantes
# ══════════════════════════════════════════════════════════════════════

# Las 10 tools del benchmark v2 (todas existen en tool_registry.TOOL_REGISTRY).
TOOL_LABELS: List[str] = [
    "text", "web", "shell", "launch", "vision",
    "fs_write", "fs_read", "fs_mkdir", "fs_list", "computer_use",
]

# Tools cuyo nodo puede ejecutarse a partir de la orden textual (plan de 1
# paso con args={}). Las fs_* necesitan 'path' estructurado que el router
# NO puede extraer de texto crudo: routearlas directo produciría un error
# ("fs_read requiere 'path'") que caería al error handler con búsqueda web,
# que es PEOR que el agent loop (que las invoca con args correctos vía tool
# calling nativo). Predicciones fs_* → fallback al agent loop.
HERRAMIENTAS_ENRUTABLES_DIRECTO = frozenset({
    "text", "web", "shell", "launch", "vision", "computer_use",
})

DEFAULT_ROUTER = "default"
DEFAULT_MODEL = "embeddinggemma:latest"
DEFAULT_MARGIN = 0.03
EMBED_BATCH = 32

# Fuente de ejemplos: viaja con el repo (ver semantic_data/README.md).
EJEMPLOS_PATH = Path(__file__).resolve().parent / "semantic_data" / "router_examples_v2.jsonl"

# Cache de prototipos por modelo (mismo directorio que bench_metrics.jsonl).
CACHE_DIR = Path.home() / ".aether" / "semantic_router"

# Override de sesión (lo setea `python run.py --planner semantic`).
ENV_OVERRIDE = "AETHER_PLANNER_ROUTER"


# ══════════════════════════════════════════════════════════════════════
# Vectores — Python puro
# ══════════════════════════════════════════════════════════════════════

def _normalizar_vec(vec: Sequence[float]) -> List[float]:
    norma = math.sqrt(sum(x * x for x in vec))
    if norma <= 0.0:
        return list(vec)
    return [x / norma for x in vec]


def _coseno(a: Sequence[float], b: Sequence[float]) -> float:
    """Coseno entre vectores ya normalizados (ambos deben tener igual dim)."""
    if len(a) != len(b):
        raise ValueError(f"dimensión inconsistente: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b))


# ══════════════════════════════════════════════════════════════════════
# Capa Ollama (única puerta al exterior; monkeypatcheable en tests)
# ══════════════════════════════════════════════════════════════════════

def _ollama_embed(model: str, textos: Sequence[str]) -> List[List[float]]:
    """Embeddings vía ollama.embed(). Valida forma y dimensión.

    Respeta la política global de keep_alive de Ollama del app
    (OLLAMA_KEEP_ALIVE, default -1 = siempre cargado): sin esto, el modelo
    de embeddings se descargaría tras el idle por defecto y la siguiente
    consulta pagaría ~8s de recarga.

    Lanza excepción ante: Ollama caído, modelo inexistente, respuesta sin
    embeddings, cantidad/dimensión inconsistentes. Quien llama decide el
    fallback (ruta_semantica NUNCA propaga errores al planner).
    """
    textos = list(textos)
    if not textos:
        return []
    keep_alive = get_config_manager().get("OLLAMA_KEEP_ALIVE", -1)
    respuesta = ollama.embed(model=model, input=textos, keep_alive=keep_alive)
    embs = getattr(respuesta, "embeddings", None)
    if embs is None and isinstance(respuesta, dict):
        embs = respuesta.get("embeddings")
    if not isinstance(embs, list) or len(embs) != len(textos):
        raise RuntimeError(f"respuesta de embeddings inválida (se esperaban {len(textos)})")
    dim = None
    for v in embs:
        if not isinstance(v, list) or not v:
            raise RuntimeError("embedding inválido en la respuesta de Ollama")
        if dim is None:
            dim = len(v)
        elif len(v) != dim:
            raise RuntimeError("embeddings con dimensión inconsistente")
    return embs


# ══════════════════════════════════════════════════════════════════════
# Resultado
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SemanticRouteResult:
    """Predicción del router semántico para una orden."""
    tool: str                      # tool top-1
    second_tool: str               # tool top-2 (la que le sigue en score)
    top_score: float              # similitud coseno top-1
    second_score: float           # similitud coseno top-2
    margin: float                  # top_score - second_score (confianza)
    margin_threshold: float        # umbral vigente al decidir
    ranked_tools: List[str]        # ranking completo (mayor a menor score)
    scores: Dict[str, float]       # score por tool
    embedding_latency_ms: float    # latencia del embedding de la consulta
    routing_latency_ms: float      # latencia total del routing
    model: str                     # modelo de embeddings usado
    direct: bool                   # True si es enrutable directo (plan 1 paso)
    fallback_reason: Optional[str] = None  # motivo por el que NO se usó (si direct=False)


# ══════════════════════════════════════════════════════════════════════
# SemanticRouter — prototipos + coseno + margin
# ══════════════════════════════════════════════════════════════════════

class SemanticRouter:
    """Router por prototipos (benchmark: embeddinggemma + prototype)."""

    def __init__(self, model: str, prototypes: Dict[str, List[float]]):
        faltan = [t for t in TOOL_LABELS if t not in prototypes]
        if faltan:
            raise ValueError(f"prototipos faltantes: {faltan}")
        self.model = model
        self.dim = len(next(iter(prototypes.values())))
        self.prototypes: Dict[str, List[float]] = {
            t: _normalizar_vec(prototypes[t]) for t in TOOL_LABELS
        }

    def route(self, texto: str, margin_threshold: float = DEFAULT_MARGIN) -> SemanticRouteResult:
        """Enruta `texto`. Lanza ante errores de Ollama (ver _ollama_embed).

        La decisión de fallback la toma ruta_semantica, no este método.
        """
        t0 = time.perf_counter()
        vec = _normalizar_vec(_ollama_embed(self.model, [texto])[0])
        t_embed = time.perf_counter()

        scores = {t: _coseno(vec, self.prototypes[t]) for t in TOOL_LABELS}
        # Ranking determinista: score desc, desempate alfabético.
        ranked = sorted(TOOL_LABELS, key=lambda t: (-scores[t], t))
        top, second = ranked[0], ranked[1]
        margin = scores[top] - scores[second]
        t_fin = time.perf_counter()

        return SemanticRouteResult(
            tool=top,
            second_tool=second,
            top_score=scores[top],
            second_score=scores[second],
            margin=margin,
            margin_threshold=float(margin_threshold),
            ranked_tools=ranked,
            scores=scores,
            embedding_latency_ms=(t_embed - t0) * 1000.0,
            routing_latency_ms=(t_fin - t0) * 1000.0,
            model=self.model,
            direct=top in HERRAMIENTAS_ENRUTABLES_DIRECTO,
        )


# ══════════════════════════════════════════════════════════════════════
# Prototipos: construcción + cache por modelo
# ══════════════════════════════════════════════════════════════════════

def _cargar_ejemplos(path=None) -> List[Dict[str, str]]:
    """Carga los ejemplos de train (orden + tool) del JSONL empaquetado."""
    if path is None:
        path = EJEMPLOS_PATH
    ejemplos: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for linea in fh:
            linea = linea.strip()
            if not linea:
                continue
            reg = json.loads(linea)
            ejemplos.append({"text": str(reg["orden"]).strip(), "tool": reg["tool"]})
    if not ejemplos:
        raise RuntimeError(f"sin ejemplos en {path}")
    return ejemplos


def construir_prototipos(
    model: str,
    ejemplos: List[Dict[str, str]],
    embed_fn: Optional[Callable[[str, List[str]], List[List[float]]]] = None,
    batch: int = EMBED_BATCH,
) -> Dict[str, List[float]]:
    """Centroide por tool: media de los embeddings (L2-normalizados) de sus
    ejemplos, re-normalizada. Estrategia 'prototype' del benchmark."""
    if embed_fn is None:
        embed_fn = _ollama_embed  # se resuelve en call time (monkeypatcheable)
    acumulado: Dict[str, List[float]] = {t: None for t in TOOL_LABELS}
    conteo: Dict[str, int] = {t: 0 for t in TOOL_LABELS}
    textos = [e["text"] for e in ejemplos]

    for ini in range(0, len(textos), batch):
        chunk = textos[ini:ini + batch]
        vectores = embed_fn(model, chunk)
        for e, vec in zip(ejemplos[ini:ini + batch], vectores):
            v = _normalizar_vec(vec)
            tool = e["tool"]
            if tool not in acumulado:
                raise ValueError(f"tool desconocida en ejemplos: {tool}")
            if acumulado[tool] is None:
                acumulado[tool] = list(v)
            else:
                for i, x in enumerate(v):
                    acumulado[tool][i] += x
            conteo[tool] += 1

    faltan = [t for t in TOOL_LABELS if conteo[t] == 0]
    if faltan:
        raise RuntimeError(f"sin ejemplos de entrenamiento para: {faltan}")
    return {t: _normalizar_vec(acumulado[t]) for t in TOOL_LABELS}


def _cache_path(model: str, cache_dir=None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    seguro = model.replace("/", "_").replace(":", "_")
    return base / f"prototypes__{seguro}.json"


def _cargar_o_construir_prototipos(
    model: str,
    cache_dir=None,
    ejemplos_path=None,
) -> Dict[str, List[float]]:
    """Prototipos desde cache; recalcula si cambió modelo/ejemplos/cache."""
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    ejemplos_path = Path(ejemplos_path) if ejemplos_path is not None else EJEMPLOS_PATH
    ejemplos_sha = hashlib.sha256(ejemplos_path.read_bytes()).hexdigest()
    path = _cache_path(model, cache_dir)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                cache = json.load(fh)
            if (
                cache.get("model") == model
                and cache.get("examples_sha256") == ejemplos_sha
                and isinstance(cache.get("prototypes"), dict)
                and all(t in cache["prototypes"] for t in TOOL_LABELS)
            ):
                protos = {
                    t: [float(x) for x in cache["prototypes"][t]]
                    for t in TOOL_LABELS
                }
                dims = {len(v) for v in protos.values()}
                if len(dims) == 1 and dims != {0}:
                    return protos  # cache válida
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass  # cache corrupta → recompute

    ejemplos = _cargar_ejemplos(ejemplos_path)
    protos = construir_prototipos(model, ejemplos)

    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "model": model,
                "dim": len(next(iter(protos.values()))),
                "examples_file": str(ejemplos_path),
                "examples_sha256": ejemplos_sha,
                "tool_count": len(protos),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "prototypes": protos,
            }, fh)
    except OSError:
        pass  # cache best-effort: si no se puede escribir, no es fatal
    return protos


# ══════════════════════════════════════════════════════════════════════
# Configuración (env de sesión > config manager > defaults)
# ══════════════════════════════════════════════════════════════════════

def resolver_modo_router() -> str:
    """"default" | "semantic". Env AETHER_PLANNER_ROUTER gana (sesión, CLI)."""
    env = (os.environ.get(ENV_OVERRIDE) or "").strip().lower()
    if env in ("default", "semantic"):
        return env
    if env:
        return DEFAULT_ROUTER  # valor inválido en env → ignorar
    valor = get_config_manager().get("PLANNER_ROUTER", DEFAULT_ROUTER)
    return valor if valor in ("default", "semantic") else DEFAULT_ROUTER


def _modelo_configurado() -> str:
    modelo = get_config_manager().get("PLANNER_ROUTER_SEMANTIC_MODEL", DEFAULT_MODEL)
    return str(modelo) if modelo else DEFAULT_MODEL


def _margin_configurado() -> float:
    valor = get_config_manager().get("PLANNER_ROUTER_SEMANTIC_MARGIN", DEFAULT_MARGIN)
    try:
        margen = float(valor)
    except (TypeError, ValueError):
        return DEFAULT_MARGIN
    return margen if 0.0 <= margen <= 1.0 else DEFAULT_MARGIN


# ══════════════════════════════════════════════════════════════════════
# Métricas opcionales (AETHER_BENCH_INSTRUMENT=1, mismo mecanismo que _llm_chat)
# ══════════════════════════════════════════════════════════════════════

def _registrar_metrica(resultado: Optional["SemanticRouteResult"],
                       outcome: str, fallback_reason: Optional[str],
                       margin_threshold: float, error: Optional[str] = None) -> None:
    """JSONL en ~/.aether/bench_metrics.jsonl. Nunca rompe la operación."""
    if not BENCH_INSTRUMENT:
        return
    try:
        registro: Dict[str, Any] = {
            "ts": time.time(),
            "mode": "semantic-router",
            "outcome": outcome,                      # "direct" | "fallback"
            "fallback_reason": fallback_reason,      # low_margin | tool_requires_structured_args | error | None
            "selected_tool": resultado.tool if resultado else None,
            "top_score": resultado.top_score if resultado else None,
            "second_score": resultado.second_score if resultado else None,
            "margin": resultado.margin if resultado else None,
            "margin_threshold": margin_threshold,
            "embedding_latency_ms": resultado.embedding_latency_ms if resultado else None,
            "routing_latency_ms": resultado.routing_latency_ms if resultado else None,
            "error": error,
        }
        with open(BENCH_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(registro) + "\n")
    except Exception:
        pass  # instrumentación best-effort


# ══════════════════════════════════════════════════════════════════════
# Singleton del router (se reconstruye si cambia el modelo configurado)
# ══════════════════════════════════════════════════════════════════════

_router_lock = threading.Lock()
_router: Optional[SemanticRouter] = None
_router_model: Optional[str] = None


def _get_router() -> SemanticRouter:
    """Router lazy: carga prototipos de cache o los construye la 1a vez."""
    global _router, _router_model
    modelo = _modelo_configurado()
    with _router_lock:
        if _router is not None and _router_model == modelo:
            return _router
        prototipos = _cargar_o_construir_prototipos(modelo)
        _router = SemanticRouter(modelo, prototipos)
        _router_model = modelo
        return _router


def reset_semantic_router() -> None:
    """Invalida el singleton (tests / cambio de configuración)."""
    global _router, _router_model
    with _router_lock:
        _router = None
        _router_model = None


# ══════════════════════════════════════════════════════════════════════
# API pública — punto de entrada único del planner
# ══════════════════════════════════════════════════════════════════════

def ruta_semantica(orden: str) -> Optional[SemanticRouteResult]:
    """Intenta enrutar `orden` por embeddings. NUNCA lanza.

    Devuelve SemanticRouteResult cuando:
      - PLANNER_ROUTER está en "semantic" (o env de sesión),
      - el margin top1−top2 >= PLANNER_ROUTER_SEMANTIC_MARGIN, y
      - la tool top-1 es enrutable directo (text/web/shell/launch/vision/
        computer_use).

    Devuelve None (→ el planner sigue su camino actual) cuando:
      - el modo es "default" (costo cero: ni toca Ollama),
      - margin < umbral (predicción dudosa → agent loop),
      - la tool es fs_* (requiere args estructurados → agent loop), o
      - cualquier error de Ollama/modelo/embeddings (fallback limpio).
    """
    if resolver_modo_router() != "semantic":
        return None

    margin_threshold = _margin_configurado()
    try:
        router = _get_router()
        resultado = router.route(orden, margin_threshold=margin_threshold)
    except Exception as exc:  # noqa: BLE001 — fallback total ante cualquier fallo
        _registrar_metrica(None, "fallback", "error", margin_threshold,
                           error=f"{type(exc).__name__}: {exc}")
        return None

    if resultado.margin < margin_threshold:
        resultado.fallback_reason = "low_margin"
        _registrar_metrica(resultado, "fallback", "low_margin", margin_threshold)
        return None

    if not resultado.direct:
        resultado.fallback_reason = "tool_requires_structured_args"
        _registrar_metrica(resultado, "fallback", "tool_requires_structured_args",
                           margin_threshold)
        return None

    _registrar_metrica(resultado, "direct", None, margin_threshold)
    return resultado



