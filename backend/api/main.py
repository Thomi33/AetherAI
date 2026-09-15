"""
Aether API — Backend revivido.

- Motor único: el grafo LangGraph de core/ (mismo motor que la TUI).
- /api/chat/stream: SSE con tokens en vivo + delta del grafo (plan/actividad).
- /api/config: lectura/escritura de config.json (configuración de la TUI).
- /api/system-prompt: system prompt editable sin tocar código.
- /api/sessions: historial de sesiones del runtime (/sesiones y /historial de la TUI).
- /ws/chat: WebSocket (socket único, multi-mensaje; mismo contrato de eventos).
- Sirve la Web UI (carpeta estática autodetectada, ver backend/core/config.py).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging

from backend.core.config import settings, resolver_web_ui_dir
from backend.core.aether_service import AetherService
from backend.api.routes import chat
from backend.api.routes import config_routes
from backend.api.routes import prompt_routes
from backend.api.routes import models_routes
from backend.api.routes import session_routes
from backend.api.routes import ws_routes

# =====================================================================
# LOGGING
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)

logger = logging.getLogger(__name__)

# =====================================================================
# LIFESPAN
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 Starting Aether API on {settings.API_HOST}:{settings.API_PORT}")

    try:
        AetherService.initialize()
        logger.info("✅ Aether Agent inicializado exitosamente")
    except Exception as e:
        logger.warning(f"⚠️ Aether se inicializará on-demand: {e}")

    yield

    logger.info("👋 Shutting down Aether API")


# =====================================================================
# APP
# =====================================================================
app = FastAPI(
    title="Aether API",
    description="API para el agente local Aether",
    version="2.1.0",
    lifespan=lifespan
)

# =====================================================================
# CORS
# =====================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# ROUTERS
# =====================================================================
app.include_router(chat.router, prefix="/api")
app.include_router(config_routes.router, prefix="/api")
app.include_router(prompt_routes.router, prefix="/api")
app.include_router(models_routes.router, prefix="/api")
app.include_router(session_routes.router, prefix="/api")
app.include_router(ws_routes.router)  # /ws/chat (websocket, sin prefijo /api)


# =====================================================================
# HEALTH / STATUS
# =====================================================================
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "agent": "Aether",
        "ollama": (AetherService.get_status().get("ollama") or "?"),
    }


@app.get("/api/status")
async def api_status():
    return AetherService.get_status()


# =====================================================================
# WEB UI (estáticos — se monta al final para no pisar las rutas de la API)
# =====================================================================
_web_ui_dir = resolver_web_ui_dir()
if _web_ui_dir is not None:
    app.mount("/", StaticFiles(directory=str(_web_ui_dir), html=True), name="web_ui")
    logger.info(f"🖥️  Web UI servida desde {_web_ui_dir}")
else:
    logger.warning("⚠️ No se encontró la Web UI (WEB_UI_DIR vacío y sin carpeta por defecto)")

    @app.get("/")
    async def root():
        return {
            "name": "Aether",
            "version": "2.1.0",
            "status": "online",
            "docs": "/docs",
            "hint": "Web UI no encontrada; configurá WEB_UI_DIR en backend/.env",
        }


# =====================================================================
# RUN
# =====================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
