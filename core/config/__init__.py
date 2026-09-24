from .settings import (
    MODO_AUTONOMO,
    OLLAMA_HOST,
    SEARXNG_URL,
    MODELO,
    TIMEOUT_CMD,
    BASE_AETHER,
    RUTA_LOGS,
    RUTA_SCREENSHOTS,
    RUTA_EMBEDDINGS,
    RUTA_BACKUPS,
    MAX_HISTORIAL,
)

# Config Manager export
from .config_manager import ConfigManager, get_config_manager, reset_config_manager

__all__ = [
    "MODO_AUTONOMO",
    "OLLAMA_HOST",
    "SEARXNG_URL",
    "MODELO",
    "TIMEOUT_CMD",
    "BASE_AETHER",
    "RUTA_LOGS",
    "RUTA_SCREENSHOTS",
    "RUTA_EMBEDDINGS",
    "RUTA_BACKUPS",
    "MAX_HISTORIAL",
    "ConfigManager",
    "get_config_manager",
    "reset_config_manager",
]
