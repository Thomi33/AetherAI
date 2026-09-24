"""
Aether API — Backend revivido.

- Motor único: el grafo LangGraph de core/ (mismo motor que la TUI).
- /api/chat/stream: SSE con tokens en vivo + delta del grafo (plan/actividad).
- /api/config: lectura/escritura de config.json (configuración de la TUI).
- /api/system-prompt: system prompt editable sin tocar código.
- /api/sessions: historial de sesiones del runtime (/sesiones y /historial de la TUI).
- /api/memory: resumen acumulativo y recuerdos (/memory de la TUI).
- /api/skills: catálogo de skills (core/skills/registry.py).
- /api/mcps: servers MCP (/mcps de la TUI) + alta de MCPs custom.
- /api/effort y /api/agent: nivel de esfuerzo y agente activo (/effort, /agents).
- /api/roblox: runtime autónomo de Roblox (/play-roblox).
- /api/proyectos y /api/tareas: registros de workspace (~<BASE_AETHER>/*.json).
- /ws/chat: WebSocket (socket único, multi-mensaje; mismo contrato de eventos).
- /api/chat y /ws/chat aceptan "attachments" (imágenes/archivos en base64).
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
from backend.api.routes import memory_routes
from backend.api.routes import skills_routes
from backend.api.routes import mcp_routes
from backend.api.routes import runtime_routes
from backend.api.routes import roblox_routes
from backend.api.routes import workspace_routes
from backend.api.routes import account_routes
from backend.api.routes import stt_routes

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
app.include_router(memory_routes.router, prefix="/api")
app.include_router(skills_routes.router, prefix="/api")
app.include_router(mcp_routes.router, prefix="/api")
app.include_router(runtime_routes.router, prefix="/api")
app.include_router(roblox_routes.router, prefix="/api")
app.include_router(workspace_routes.router, prefix="/api")
app.include_router(account_routes.router, prefix="/api")
app.include_router(stt_routes.router, prefix="/api")
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
# AVATARES (Settings → Account) — servidos desde <BASE_AETHER>/avatars
# =====================================================================
try:
    from backend.api.routes.account_routes import _AVATAR_DIR
    _AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/avatars", StaticFiles(directory=str(_AVATAR_DIR)),
              name="avatars")
except Exception as _avatars_exc:  # nunca bloquear el arranque por avatares
    logger.warning(f"⚠️ No se pudo montar /avatars: {_avatars_exc}")


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
