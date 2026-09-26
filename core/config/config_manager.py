"""
config_manager.py — Configuración centralizada con hot reload para Aether.

La configuración está dividida en dos capas:
  1. config.json       → defaults versionados del proyecto (solo lectura lógica).
  2. config.local.json → overrides locales/hardware/usuario (ignorado por Git).

Esto permite que el instalador adapte Aether a cada máquina sin modificar
archivos versionados y que una actualización del repositorio conserve los
ajustes locales.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ConfigChangeEvent:
    """Evento de cambio de configuración."""
    key: str
    value: Any
    old_value: Any
    timestamp: float


class ConfigValidator:
    """Validador de configuraciones."""

    VALID_MODELS = [
        "ornith:9b", "ornith:7b", "ornith-1.5:9b", "qwen3.5:9b",
        "qwen2.5vl:7b", "qwen2.5:7b", "qwen2.5:3b", "qwen2.5:1.5b",
        "moondream", "minicpm-v", "minicpm-v4.6:latest",
        "hf.co/deepreinforce-ai/Ornith-1.0-35B-GGUF:q4_K_M",
        "hf.co/deepreinforce-ai/Ornith-1.0-35B-GGUF:Q4_K_M",
    ]
    VALID_PROVIDERS = ["Ollama", "OpenAI", "Groq"]
    VALID_TEMP_RANGE = (0.0, 2.0)
    VALID_MAX_TOKENS_RANGE = (1, 8192)
    VALID_NUM_PREDICT_RANGE = (1, 65536)
    VALID_TIMEOUT_RANGE = (10, 300)
    VALID_CTX_RANGE = (4096, 262144)
    VALID_MAX_TURNOS_RANGE = (0, 1000)

    @staticmethod
    def validate_mode(value: Any) -> bool:
        return value in (True, False)

    @staticmethod
    def validate_timeout(value: Any) -> bool:
        try:
            val = int(value)
            return ConfigValidator.VALID_TIMEOUT_RANGE[0] <= val <= ConfigValidator.VALID_TIMEOUT_RANGE[1]
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_num_ctx(value: Any) -> bool:
        try:
            val = int(value)
            return ConfigValidator.VALID_CTX_RANGE[0] <= val <= ConfigValidator.VALID_CTX_RANGE[1]
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_max_turnos(value: Any) -> bool:
        try:
            val = int(value)
            return ConfigValidator.VALID_MAX_TURNOS_RANGE[0] <= val <= ConfigValidator.VALID_MAX_TURNOS_RANGE[1]
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_temp(value: Any) -> bool:
        try:
            val = float(value)
            return ConfigValidator.VALID_TEMP_RANGE[0] <= val <= ConfigValidator.VALID_TEMP_RANGE[1]
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_max_tokens(value: Any) -> bool:
        try:
            val = int(value)
            return ConfigValidator.VALID_MAX_TOKENS_RANGE[0] <= val <= ConfigValidator.VALID_MAX_TOKENS_RANGE[1]
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_num_predict(value: Any) -> bool:
        try:
            val = int(value)
            return ConfigValidator.VALID_NUM_PREDICT_RANGE[0] <= val <= ConfigValidator.VALID_NUM_PREDICT_RANGE[1]
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_model(value: Any) -> bool:
        if value in ConfigValidator.VALID_MODELS:
            return True
        if isinstance(value, str) and re.match(r"^[a-zA-Z0-9][\w\-./]{1,80}(:[\w\-+.]{1,40})?$", value):
            return True
        return False

    @staticmethod
    def validate_provider(value: Any) -> bool:
        return value in ConfigValidator.VALID_PROVIDERS

    @staticmethod
    def validate_bool(value: Any) -> bool:
        return isinstance(value, bool)

    @staticmethod
    def validate_string(value: Any) -> bool:
        return isinstance(value, str) and len(value) > 0

    @staticmethod
    def validate_keep_alive(value: Any) -> bool:
        return isinstance(value, (int, float)) or (isinstance(value, str) and len(value) > 0)


class ConfigWatcher:
    """Observador de uno o varios archivos de configuración."""

    def __init__(self, config_paths: Union[Path, List[Path]], callback: Callable[[str], None]):
        if isinstance(config_paths, Path):
            config_paths = [config_paths]
        self._config_paths = list(config_paths)
        self._callback = callback
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_mtimes: Dict[Path, float] = {}
        self._poll_interval = 2

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self) -> None:
        for path in self._config_paths:
            try:
                self._last_mtimes[path] = path.stat().st_mtime
            except FileNotFoundError:
                self._last_mtimes[path] = 0

        while self._running:
            for path in self._config_paths:
                try:
                    current_mtime = path.stat().st_mtime
                    if current_mtime != self._last_mtimes.get(path, 0):
                        self._last_mtimes[path] = current_mtime
                        self._callback(str(path))
                except FileNotFoundError:
                    pass
            time.sleep(self._poll_interval)


class ConfigManager:
    """Gestor centralizado con configuración base + overrides locales."""

    VALIDATORS: Dict[str, Callable[[Any], bool]] = {
        "MODELO": ConfigValidator.validate_model,
        "MODELO_VISION": ConfigValidator.validate_model,
        "OLLAMA_HOST": ConfigValidator.validate_string,
        "SEARXNG_URL": ConfigValidator.validate_string,
        "MODO_AUTONOMO": ConfigValidator.validate_bool,
        "TOOL_CALLING_NATIVO": ConfigValidator.validate_bool,
        "TIMEOUT_CMD": ConfigValidator.validate_timeout,
        "NUM_CTX": ConfigValidator.validate_num_ctx,
        "MAX_TURNOS_CONTEXTO": ConfigValidator.validate_max_turnos,
        "MAX_TURNOS_CONTEXTO_PLAN": ConfigValidator.validate_max_turnos,
        "MAX_TURNOS_CONTEXTO_CHAT": ConfigValidator.validate_max_turnos,
        "OLLAMA_KEEP_ALIVE": ConfigValidator.validate_keep_alive,
        "OLLAMA_GEN_OPTIONS": lambda x: isinstance(x, dict),
        "OLLAMA_NUM_PARALLEL": lambda x: isinstance(x, int) and x > 0,
        "OLLAMA_MAX_LOADED_MODELS": lambda x: isinstance(x, int) and x > 0,
        "MAX_STEPS_COMPUTER_USE": lambda x: isinstance(x, int) and 1 <= x <= 64,
        "MAX_HISTORIAL": lambda x: isinstance(x, int) and x > 0,
        "CONTEXTO_CONV_MAX_CHARS": lambda x: isinstance(x, int) and x > 0,
        "BASE_AETHER": ConfigValidator.validate_string,
        "CARPETA_AETHER": ConfigValidator.validate_string,
        "TEMPERATURE": ConfigValidator.validate_temp,
        "MAX_TOKENS": ConfigValidator.validate_max_tokens,
        "NUM_PREDICT": ConfigValidator.validate_num_predict,
        "NUM_PREDICT_PLANNER": ConfigValidator.validate_num_predict,
        "VERBOSE": ConfigValidator.validate_bool,
        "DEBUG": ConfigValidator.validate_bool,
        "THEME": ConfigValidator.validate_string,
        "REFRESH_RATE": lambda x: isinstance(x, (int, float)) and x > 0,
        "STT_ENABLED": ConfigValidator.validate_bool,
        "STT_LANGUAGE": lambda x: x is None or isinstance(x, str),
        "STT_VENV_PYTHON": lambda x: isinstance(x, str),
        "AUDIO_INPUT_MATCH": ConfigValidator.validate_string,
        "STT_VOCAB_HINT": lambda x: isinstance(x, str),
        # ── SYSTEM PROMPT POR CAPAS (ver el bloque de core/agent/prompts.py) ──
        # BEHAVIOR es la casilla principal ("solo comportamiento conversacional").
        # EXTRA / SINTESIS_EXTRA / OVERRIDE son claves legadas que ahora se
        # suman como capa de comportamiento: ninguna REEMPLAZA las
        # instrucciones internas ni el comportamiento predeterminado.
        "SYSTEM_PROMPT_BEHAVIOR": lambda x: isinstance(x, str),
        "SYSTEM_PROMPT_OVERRIDE": lambda x: isinstance(x, str),
        "SYSTEM_PROMPT_EXTRA": lambda x: isinstance(x, str),
        "SYSTEM_PROMPT_SINTESIS_EXTRA": lambda x: isinstance(x, str),
        # ── RUNTIME EDITABLE (TUI /effort y /agents; Web UI /api/effort y
        # /api/agent). EFFORT persiste el último nivel elegido; el mapeo a
        # TEMPERATURE/NUM_CTX/NUM_PREDICT lo aplica quien lo setea (mismo mapa
        # en tui/app.py y backend/api/routes/runtime_routes.py).
        "EFFORT": lambda x: x in ("low", "medium", "high", "max"),
        # Nativos (build/plan) + custom (kebab-case lowercase, ver
        # runtime_routes._RE de agentes). Antes solo aceptaba build/plan y los
        # agentes custom no se podían activar ("no hacía nada").
        "AGENTE": lambda x: (
            isinstance(x, str)
            and re.fullmatch(r"[a-z][a-z0-9-]{0,63}", x) is not None
        ),
        # ── SEMANTIC ROUTER (OPT-IN; ver core/agent/semantic_router.py) ──
        # default = comportamiento actual (agent loop). semantic = atajo por
        # embeddings (embeddinggemma + prototipos) con fallback automático.
        "PLANNER_ROUTER": lambda x: x in ("default", "semantic"),
        "PLANNER_ROUTER_SEMANTIC_MODEL": ConfigValidator.validate_model,
        "PLANNER_ROUTER_SEMANTIC_MARGIN": lambda x: (
            isinstance(x, (int, float)) and 0.0 <= float(x) <= 1.0
        ),
    }

    DEFAULTS: Dict[str, Any] = {
        "MODELO": "ornith:9b",
        "MODELO_VISION": "ornith-1.5:9b",
        "OLLAMA_HOST": "http://localhost:11434",
        "SEARXNG_URL": "http://localhost:8081",
        "MODO_AUTONOMO": True,
        "TOOL_CALLING_NATIVO": True,
        "TIMEOUT_CMD": 60,
        "NUM_CTX": 8192,
        "MAX_TURNOS_CONTEXTO": 200,
        "MAX_TURNOS_CONTEXTO_PLAN": 0,
        "MAX_TURNOS_CONTEXTO_CHAT": 10,
        "OLLAMA_KEEP_ALIVE": -1,
        "OLLAMA_GEN_OPTIONS": {"num_batch": 512, "num_gpu": 8, "num_thread": 8},
        "OLLAMA_NUM_PARALLEL": 4,
        "OLLAMA_MAX_LOADED_MODELS": 2,
        "MAX_STEPS_COMPUTER_USE": 8,
        "MAX_HISTORIAL": 100000,
        "CONTEXTO_CONV_MAX_CHARS": 16000,
        "BASE_AETHER": "~/Aether",
        "CARPETA_AETHER": "~/Aether",
        "TEMPERATURE": 0.6,
        "MAX_TOKENS": 2048,
        "NUM_PREDICT": 2048,
        "NUM_PREDICT_PLANNER": 3072,
        "PLANNER_ROUTER": "default",
        "PLANNER_ROUTER_SEMANTIC_MODEL": "embeddinggemma:latest",
        "PLANNER_ROUTER_SEMANTIC_MARGIN": 0.03,
        "VERBOSE": False,
        "DEBUG": False,
        "THEME": "default",
        "REFRESH_RATE": 0.1,
        "STT_ENABLED": True,
        "STT_LANGUAGE": "es",
        "STT_VENV_PYTHON": "",
        "AUDIO_INPUT_MATCH": "AudioBox USB 96",
        "STT_VOCAB_HINT": (
            "Aether, GitHub, Ollama, LangGraph, Ornith, MCP, Textual, "
            "Hyprland, Wayland, NVMe, Notion, Steam, Roblox, Minecraft, "
            "VLSM, subnetting, ydotool, faster-whisper, SQLite, "
            "consolidator, AudioBox USB 96, CrewAI."
        ),
        # ── SYSTEM PROMPT POR CAPAS (ver core/agent/prompts.py) ──
        # SYSTEM_PROMPT_BEHAVIOR: la casilla principal — comportamiento
        #   conversacional del usuario (personalidad/tono/estilo), capa que
        #   NO puede pisar ni las instrucciones internas ni la personalidad
        #   predeterminada de Aether.
        # OVERRIDE/EXTRA/SINTESIS_EXTRA: legado — se suman como capa de
        #   comportamiento; OVERRIDE ya NO reemplaza el prompt completo.
        "SYSTEM_PROMPT_BEHAVIOR": "",
        "SYSTEM_PROMPT_OVERRIDE": "",
        "SYSTEM_PROMPT_EXTRA": "",
        "SYSTEM_PROMPT_SINTESIS_EXTRA": "",
        # ── RUNTIME EDITABLE (TUI /effort y /agents; Web UI /api/effort y
        # /api/agent). Persisten la última elección para compartir estado.
        "EFFORT": "medium",
        "AGENTE": "build",
    }

    def __init__(self, config_path: Optional[Path] = None):
        self._config_path = config_path or Path(__file__).parent / "config.json"
        self._local_config_path = self._config_path.with_name("config.local.json")
        self._config: Dict[str, Any] = {}
        self._local_config: Dict[str, Any] = {}
        self._subscriptions: Dict[str, List[Callable[[ConfigChangeEvent], None]]] = {}
        self._watcher: Optional[ConfigWatcher] = None
        self._lock = threading.RLock()
        self._load_config()

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"La configuración debe ser un objeto JSON: {path}")
        return data

    def _load_config(self) -> None:
        """Cargar defaults, base versionada y overrides locales."""
        with self._lock:
            try:
                base = self._read_json(self._config_path)
                local = self._read_json(self._local_config_path)
                merged = dict(self.DEFAULTS)
                merged.update(base)
                merged.update(local)
                self._local_config = dict(local)
                self._config = merged
            except (json.JSONDecodeError, IOError, ValueError) as e:
                print(f"⚠️  [CONFIG]: Error cargando configuración: {e}")
                self._local_config = {}
                self._config = dict(self.DEFAULTS)

    def _save_config(self) -> None:
        """Persistir únicamente overrides locales, nunca config.json."""
        with self._lock:
            self._local_config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._local_config_path.with_suffix(".json.tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._local_config, f, indent=2, ensure_ascii=False)
                f.write("\n")
            tmp_path.replace(self._local_config_path)

    def _notify_subscribers(self, key: str, new_value: Any, old_value: Any) -> None:
        if key in self._subscriptions:
            event = ConfigChangeEvent(key, new_value, old_value, time.time())
            for handler in self._subscriptions[key]:
                try:
                    handler(event)
                except Exception as e:
                    print(f"⚠️  [CONFIG]: Error en subscriber de '{key}': {e}")

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._config.get(key, default)

    def get_all(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._config)

    def set(self, key: str, value: Any, validate: bool = True) -> bool:
        with self._lock:
            if validate and key in self.VALIDATORS and not self.VALIDATORS[key](value):
                print(f"❌ [CONFIG]: Validación falló para '{key}': {value}")
                return False
            old_value = self._config.get(key)
            old_local = self._local_config.get(key, None)
            had_local = key in self._local_config
            self._config[key] = value
            self._local_config[key] = value
            try:
                self._save_config()
                self._notify_subscribers(key, value, old_value)
                print(f"✅ [CONFIG]: {key} = {value}")
                return True
            except Exception as e:
                self._config[key] = old_value
                if had_local:
                    self._local_config[key] = old_local
                else:
                    self._local_config.pop(key, None)
                print(f"❌ [CONFIG]: Error aplicando cambio '{key}': {e}")
                return False

    def set_multiple(self, config: Dict[str, Any], validate: bool = True) -> Dict[str, bool]:
        return {key: self.set(key, value, validate) for key, value in config.items()}

    def reload(self) -> bool:
        with self._lock:
            try:
                old_config = dict(self._config)
                self._load_config()
                for key in set(old_config) | set(self._config):
                    old = old_config.get(key)
                    new = self._config.get(key)
                    if old != new:
                        self._notify_subscribers(key, new, old)
                print(f"🔄 [CONFIG]: Recargado desde {self._config_path} + {self._local_config_path}")
                return True
            except Exception as e:
                print(f"❌ [CONFIG]: Error recargando: {e}")
                return False

    def reset(self, key: Optional[str] = None) -> bool:
        """Eliminar overrides locales y volver al valor de config.json/defaults."""
        with self._lock:
            if key:
                if key not in self._local_config:
                    return key in self._config or key in self.DEFAULTS
                old_value = self._config.get(key)
                self._local_config.pop(key, None)
                self._save_config()
                self._load_config()
                self._notify_subscribers(key, self._config.get(key), old_value)
                print(f"🔄 [CONFIG]: {key} volvió al valor base: {self._config.get(key)}")
                return True

            old_config = dict(self._config)
            self._local_config = {}
            self._save_config()
            self._load_config()
            for key in set(old_config) | set(self._config):
                if old_config.get(key) != self._config.get(key):
                    self._notify_subscribers(key, self._config.get(key), old_config.get(key))
            print("🔄 [CONFIG]: Overrides locales eliminados; configuración base restaurada")
            return True

    def subscribe(self, key: str, handler: Callable[[ConfigChangeEvent], None]) -> None:
        with self._lock:
            self._subscriptions.setdefault(key, []).append(handler)

    def unsubscribe(self, key: str, handler: Callable[[ConfigChangeEvent], None]) -> None:
        with self._lock:
            if key in self._subscriptions:
                try:
                    self._subscriptions[key].remove(handler)
                except ValueError:
                    pass

    def start_watching(self) -> None:
        if self._watcher is None:
            self._watcher = ConfigWatcher(
                [self._config_path, self._local_config_path],
                lambda path: self.reload(),
            )
            self._watcher.start()

    def stop_watching(self) -> None:
        if self._watcher:
            self._watcher.stop()
            self._watcher = None

    def save(self) -> bool:
        with self._lock:
            try:
                self._save_config()
                return True
            except Exception as e:
                print(f"❌ [CONFIG]: Error guardando: {e}")
                return False

    @property
    def modelo(self) -> str:
        return self.get("MODELO", "ornith:9b")

    @modelo.setter
    def modelo(self, value: str) -> None:
        self.set("MODELO", value)

    @property
    def modo_autonomo(self) -> bool:
        return self.get("MODO_AUTONOMO", True)

    @modo_autonomo.setter
    def modo_autonomo(self, value: bool) -> None:
        self.set("MODO_AUTONOMO", value)

    @property
    def verbose(self) -> bool:
        return self.get("VERBOSE", False)

    @verbose.setter
    def verbose(self, value: bool) -> None:
        self.set("VERBOSE", value)

    @property
    def debug(self) -> bool:
        return self.get("DEBUG", False)

    @debug.setter
    def debug(self, value: bool) -> None:
        self.set("DEBUG", value)

    @property
    def theme(self) -> str:
        return self.get("THEME", "default")

    @theme.setter
    def theme(self, value: str) -> None:
        self.set("THEME", value)

    @property
    def refresh_rate(self) -> float:
        return self.get("REFRESH_RATE", 0.1)

    @refresh_rate.setter
    def refresh_rate(self, value: float) -> None:
        self.set("REFRESH_RATE", value)


_config_manager: Optional[ConfigManager] = None
_config_manager_lock = threading.Lock()


def get_config_manager() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        with _config_manager_lock:
            if _config_manager is None:
                _config_manager = ConfigManager()
    return _config_manager


def reset_config_manager() -> None:
    global _config_manager
    with _config_manager_lock:
        _config_manager = None
