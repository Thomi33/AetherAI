"""
Configuración simple para Aether — reemplazo de config_manager.py (526 líneas → ~80 líneas).

Dos capas:
  1. config.json       → defaults versionados (solo lectura)
  2. config.local.json → overrides locales (ignorado por Git)

No hay: validadores custom, watchers, subscriptions, events, thread-locks, hot-reload.
Solo: carga JSON, merge shallow, propiedades @property para claves comunes.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class Settings:
    """Configuración simple con merge de dos capas JSON."""
    
    def __init__(self):
        self._config_path = Path(__file__).parent / "config.json"
        self._local_path = Path(__file__).parent / "config.local.json"
        self._lock = threading.Lock()
        self._config: dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Carga config.json + merge config.local.json (shallow merge por clave)."""
        # Base config
        if self._config_path.exists():
            try:
                with open(self._config_path, encoding="utf-8") as f:
                    self._config = json.load(f)
            except Exception:
                self._config = {}
        
        # Local overrides (shallow merge: local keys overwrite base keys)
        if self._local_path.exists():
            try:
                with open(self._local_path, encoding="utf-8") as f:
                    local = json.load(f)
                if isinstance(local, dict):
                    self._config.update(local)
            except Exception:
                pass
    
    def get(self, key: str, default: Any = None) -> Any:
        """Obtiene un valor de configuración."""
        with self._lock:
            return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """Establece un valor en config.local.json (persistente)."""
        with self._lock:
            self._config[key] = value
            self._save_local()
        return True
    
    def _save_local(self) -> None:
        """Guarda solo la capa local."""
        try:
            local = {}
            if self._local_path.exists():
                with open(self._local_path, encoding="utf-8") as f:
                    local = json.load(f)
            if not isinstance(local, dict):
                local = {}
            # Update only keys that exist in merged config (avoid saving base defaults)
            for key, value in self._config.items():
                local[key] = value
            with open(self._local_path, "w", encoding="utf-8") as f:
                json.dump(local, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"❌ [CONFIG]: Error guardando local: {e}")
    
    def reload(self) -> None:
        """Recarga la configuración desde disco."""
        with self._lock:
            self._load()
    
    # Propiedades de conveniencia para claves comunes
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


# Singleton
_settings: Settings | None = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """Obtiene la instancia singleton de Settings."""
    global _settings
    if _settings is None:
        with _settings_lock:
            if _settings is None:
                _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Resetea el singleton (para tests)."""
    global _settings
    with _settings_lock:
        _settings = None


# Compatibilidad: alias para código que use get_config_manager
get_config_manager = get_settings
reset_config_manager = reset_settings