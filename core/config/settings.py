"""
Configuración global y constantes del sistema Aether.
"""
import os
import warnings
import ast
from pathlib import Path

# =====================================================================
# ⚙️ CONFIGURACIÓN BASE
# =====================================================================
from core.config.config_manager import get_config_manager

_CONFIG = get_config_manager()


def _config_value(nombre: str, default):
    """Lee valores configurables desde config.json mediante ConfigManager."""
    return _CONFIG.get(nombre, default)


MODO_AUTONOMO = _config_value("MODO_AUTONOMO", True)
OLLAMA_HOST = _config_value("OLLAMA_HOST", "http://localhost:11434")
SEARXNG_URL = _config_value("SEARXNG_URL", "http://localhost:8081")
MODELO = _config_value("MODELO", "ornith:9b")

# === Minimal optional instrumentation for AetherBench ===
# Set AETHER_BENCH_INSTRUMENT=1 before importing to enable metrics logging
# to ~/.aether/bench_metrics.jsonl (does not affect normal operation).
# Ornith-native behavior (reasoning-based intent detection, <think> parsing,
# recommended sampling) is now always enabled. No legacy mode.
BENCH_INSTRUMENT = os.environ.get("AETHER_BENCH_INSTRUMENT", "0") == "1"
BENCH_LOG_PATH = Path.home() / ".aether" / "bench_metrics.jsonl"
BENCH_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
TIMEOUT_CMD = _config_value("TIMEOUT_CMD", 60)
# Carpeta base de datos del agente (DB, notas, logs, screenshots…).
# El instalador puede definir AETHER_DATA_DIR en .env para usar una carpeta
# separada o el directorio del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _env_local(nombre: str) -> str:
    valor = os.environ.get(nombre)
    if valor:
        return valor
    env_path = PROJECT_ROOT / ".env"
    try:
        for linea in env_path.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if linea.startswith(f"{nombre}="):
                return linea.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return ""


BASE_AETHER = Path(
    _env_local("AETHER_DATA_DIR")
    or _config_value("BASE_AETHER", "~/Aether")
).expanduser().resolve()

MODELO_VISION = _config_value("MODELO_VISION", MODELO)

# ───────────────────────────────────────────────────────
# 🖱️  COMPUTER USE (loop percepción-acción: click/escribir vía ydotool)
# ───────────────────────────────────────────────────────
# Límite duro de pasos por invocación. Sin esto, un objetivo mal
# especificado (o un elemento que el VLM no encuentra) puede dejar el
# loop dando vueltas indefinidamente gastando inferencia y arriesgando
# clicks en lugares no deseados. 8 es conservador; subílo si ves que
# tareas legítimas se cortan antes de tiempo.
MAX_STEPS_COMPUTER_USE = _config_value("MAX_STEPS_COMPUTER_USE", 8)

TOOL_CALLING_NATIVO = _config_value("TOOL_CALLING_NATIVO", True)

RUTA_LOGS        = BASE_AETHER / "logs"
RUTA_SCREENSHOTS = BASE_AETHER / "screenshots"
RUTA_EMBEDDINGS  = BASE_AETHER / "embeddings"
RUTA_BACKUPS     = BASE_AETHER / "backups"
RUTA_NOTAS       = BASE_AETHER / "notes"

# ─────────────────────────────────────────────────────────────────────
# 🧠 MEMORIA — store JSON único (core/memory/memoria_store.py)
# ─────────────────────────────────────────────────────────────────────
# NO hay SQLite: toda la memoria persistente vive en
# <BASE_AETHER>/db/memoria.json (+ .bak rotativo). current.db (sqlite) solo
# se lee UNA VEZ si existe, para la migración legacy automática del store.
MEMORIA_DIR   = BASE_AETHER / "db"

BASE_AETHER.mkdir(parents=True, exist_ok=True)
RUTA_NOTAS.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────
# 🧩 SKILLS (core/skills/registry.py)
# ─────────────────────────────────────────────────────────────────────
# A diferencia de BASE_AETHER (datos de runtime, fuera del repo: ~/Aether),
# las skills son comportamiento del agente -- viven DENTRO del proyecto para
# versionarse junto al código.
RUTA_SKILLS  = PROJECT_ROOT / "skills"

# ─────────────────────────────────────────────────────────────────────
# 📁 CARPETA DE TRABAJO (feature: Aether trabaja sobre la ruta desde la
# que se lo abrió, con autorización explícita -- ver core/config/dir_authorization.py)
# ─────────────────────────────────────────────────────────────────────
# bin/aether hace `cd "$AETHER_HOME"` antes de ejecutar Python (para que
# imports y venv resuelvan siempre igual) -- por eso os.getcwd() en runtime
# YA NO es la carpeta desde la que el usuario lo invocó. bin/aether exporta
# AETHER_CWD con el $PWD original ANTES de ese cd (o lo que pidió --workdir).
# Si Aether se corre directo (python run.py, sin pasar por bin/aether) no
# hay AETHER_CWD seteada y usamos os.getcwd() tal cual.
#
# OJO: se resuelve LAZY (PEP 562, __getattr__ abajo) porque AETHER_CWD /
# --workdir pueden llegar DESPUÉS de los imports (argparse corre tras ellos).
# Cada acceso a settings.RUTA_TRABAJO reevalúa la env var. NO hacer
# `from ... import RUTA_TRABAJO` una sola vez y cachearlo: accedé vía
# `settings.RUTA_TRABAJO` o reimportá. Los módulos ya migrados son:
# filesystem_tool, file_writer, shell_executor, prompts.
def _resolver_ruta_trabajo() -> Path:
    return Path(os.environ.get("AETHER_CWD") or os.getcwd()).resolve()


def __getattr__(name: str):
    if name == "RUTA_TRABAJO":
        return _resolver_ruta_trabajo()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

MAX_HISTORIAL = _config_value("MAX_HISTORIAL", 100000)


# ─────────────────────────────────────────────────────────────────────
# 🧠 CONTEXTO CONVERSACIONAL (memoria inyectada en el prompt)
# ─────────────────────────────────────────────────────────────────────
# Ventana de contexto del modelo en Ollama. 
# Ornith-1.0: contexto nativo 262144 (256K). 
# Usamos valor conservador para no agotar VRAM/RAM en inyección de memoria;
# el template y el modelo manejan bien ventanas grandes si el hardware lo permite.
# Subir (ej 32768 o más) mejora continuidad de historial.
NUM_CTX = _config_value("NUM_CTX", 8192)

# Máximo de turnos de conversación a inyectar (RAM → prompt). Subido de 50
# a 200. Se acota además por presupuesto de caracteres (abajo) para que un
# pico de turnos largos nunca desborde NUM_CTX.
MAX_TURNOS_CONTEXTO = _config_value("MAX_TURNOS_CONTEXTO", 200)

# Presupuesto de caracteres del bloque de conversación. Se incluyen los
# turnos MÁS RECIENTES hacia atrás hasta llegar a este tope (los más viejos
# se descartan). ~4 chars/token → 16000 ≈ 4000 tokens, holgado en NUM_CTX.
CONTEXTO_CONV_MAX_CHARS = _config_value("CONTEXTO_CONV_MAX_CHARS", 16000)

# Turnos de conversación a inyectar DURANTE pasos de un plan multi-tool.
# El chat normal (planes de 1 paso) usa el historial completo (arriba), pero
# en un pipeline multi-tool el historial largo CONTAMINA cada paso (un paso
# "se acuerda" de tareas viejas y hace algo distinto). Por eso aquí se recorta
# fuerte: 0 = sin historial conversacional (cada paso se ejecuta con su propia
# instrucción + el contexto de pasos previos, que es lo único relevante).
MAX_TURNOS_CONTEXTO_PLAN = _config_value("MAX_TURNOS_CONTEXTO_PLAN", 0)

# Turnos de conversación a inyectar en el camino de CHARLA (tool=text /
# modo_chat). El chat necesita continuidad, pero NO los 200 turnos completos:
# el historial largo hace que el modelo "siga" una tarea vieja (p.ej. un "Hola"
# que derivaba en un diagnóstico de hardware de una sesión anterior). Una
# ventana chica da contexto reciente sin arrastrar tareas viejas (anti-
# contaminación del chat). 0 = sin historial (no recomendado para chat).
MAX_TURNOS_CONTEXTO_CHAT = _config_value("MAX_TURNOS_CONTEXTO_CHAT", 10)


# ─────────────────────────────────────────────────────────────────────
# ⚡ RENDIMIENTO DE OLLAMA (latencia ↓ a cambio de recursos)
# ─────────────────────────────────────────────────────────────────────
# keep_alive: cuánto mantener el modelo CARGADO en VRAM tras cada request.
# Es la mayor palanca de latencia: evita recargar el modelo (varios segundos)
# entre cada llamada (intent gate, node_text, synthesizer, resúmenes de tools).
# Valores: "30m", "1h" o "-1" (cargado para siempre). Con recursos de sobra,
# conviene mantenerlo caliente. Lo consume _llm_chat vía ollama.chat(keep_alive=).
OLLAMA_KEEP_ALIVE = _config_value("OLLAMA_KEEP_ALIVE", -1)

# Opciones de generación que se mergean en CADA llamada a _llm_chat (además de
# num_ctx). 
#
# Ornith-1.0 oficiales (deep-reinforce.com + HF model cards):
#   Recommended: temperature=0.6, top_p=0.95, top_k=20
#   Para benchmarks reproducibles: temperature=1.0, top_p=1.0 o 0.95
# Ajustes para throughput:
#   num_batch, num_gpu, num_thread como antes.
OLLAMA_GEN_OPTIONS = _config_value("OLLAMA_GEN_OPTIONS", {})

# Concurrencia del SERVIDOR ollama (no del cliente). El grafo es SECUENCIAL
# (una request a la vez), así que un solo modelo cargado alcanza. Estas vars
# de entorno solo importan si en el futuro se paralelizan pasos:
#   OLLAMA_NUM_PARALLEL      : requests concurrentes por modelo.
#   OLLAMA_MAX_LOADED_MODELS : modelos distintos cargados a la vez (texto+visión).
# Se setean en el entorno del servidor (no acá); ver explicación en el chat.
OLLAMA_NUM_PARALLEL = _config_value("OLLAMA_NUM_PARALLEL", 4)
OLLAMA_MAX_LOADED_MODELS = _config_value("OLLAMA_MAX_LOADED_MODELS", 2)


# =====================================================================
# 🛠️ PARCHES DE COMPATIBILIDAD AST
# =====================================================================
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

for _nodo in ("NameConstant", "Str", "Num", "Bytes", "Ellipsis"):
    if not hasattr(ast, _nodo):
        setattr(ast, _nodo, type(_nodo, (ast.AST,), {}))


# =====================================================================
# 🔇 ENTORNO / TELEMETRÍA
# =====================================================================
# Compat OpenAI-sobre-Ollama para los clientes langchain (que además pasan la
# key explícitamente; estas vars son un fallback inofensivo). El motor
# LangGraph usa el cliente `ollama` nativo (apunta a localhost:11434).
os.environ.update({
    "OPENAI_API_KEY":  "ollama",
    "OLLAMA_API_BASE": f"{OLLAMA_HOST}",
})

import logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)


# =====================================================================
# 🗣️ PALABRAS CLAVE DE INTENCIÓN
# =====================================================================

PALABRAS_CLAVE_ESCRITURA: set[str] = {
    "escribe", "crea", "genera", "redacta", "escribeme", "escríbeme",
    "archivo", "fichero", "guarda", "nota", "crea un archivo",
}

PALABRAS_CLAVE_VISION: list[str] = [
    "qué ves", "que ves", "describe", "mira", "captura", "pantalla",
    "screenshot", "imagen", "foto", "observa", "analiza la pantalla",
]

PALABRAS_CLAVE_LANZAR: set[str] = {
    "abre", "lanza", "inicia", "ejecuta", "arranca", "corre", "run",
    "start", "open", "abrir", "lanzar", "iniciar",
}

PALABRAS_CLAVE_WEB: list[str] = [
    "busca", "buscar", "search", "qué es", "que es", "quién es",
    "quien es", "cuánto", "cuanto", "precio", "noticia", "noticias",
    "clima", "tiempo en", "cómo se", "como se", "últimas", "ultimas",
    "wikipedia", "define", "explica qué", "explica que",
]