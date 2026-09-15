"""Configuración del backend API (fastapi/uvicorn/web UI)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False

    # CORS (backend CLI/local: permisivo)
    CORS_ORIGINS: str = "*"

    # Carpeta de la Web UI (vacío = autodetección, ver resolver_web_ui_dir)
    WEB_UI_DIR: str = ""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()


def resolver_web_ui_dir() -> Path | None:
    """
    Resuelve la carpeta estática de la Web UI, en orden:
      1. WEB_UI_DIR del .env / entorno (backend/.env)
      2. ~/aether_web_ui/web  (repo dedicado de la Web UI)
      3. <proyecto>/web_ui    (copia de desarrollo)
    """
    if settings.WEB_UI_DIR:
        p = Path(settings.WEB_UI_DIR).expanduser().resolve()
        return p if p.is_dir() else None
    candidatos = (
        Path.home() / "aether_web_ui" / "web",
        PROJECT_ROOT / "web_ui",
    )
    for c in candidatos:
        if c.is_dir():
            return c
    return None

