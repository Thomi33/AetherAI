"""
session_routes.py — Sesiones del historial persistente de Aether.

Es la MISMA fuente de datos que los comandos de la TUI:
- GET /api/sessions              → lista de sesiones (equivalente a /sesiones)
- GET /api/sessions/{sesion_id}  → turnos de una sesión (equivalente a
                                   /historial <n>, pero por sesion_id)

Las "conversaciones" de la Web UI viven en localStorage del navegador;
estos endpoints la conectan con el historial real del runtime (memoria.json),
así la web puede abrir sesiones hechas desde la TUI (y viceversa: todo lo
que la web procesa también queda registrado por el grafo).
"""

from fastapi import APIRouter

router = APIRouter()

_MAX_SESIONES = 50


@router.get("/sessions")
async def listar(limit: int = 15):
    """Resumen de sesiones recientes: sesion_id, inicio, turnos y preview."""
    from core.memory.memory_manager import listar_sesiones

    try:
        sesiones = listar_sesiones(limit=max(1, min(limit, _MAX_SESIONES)))
        return {"ok": True, "sessions": sesiones}
    except Exception as e:  # noqa: BLE001 — la UI muestra el error
        return {"ok": False, "sessions": [], "error": str(e)}


@router.get("/sessions/{sesion_id}")
async def detalle(sesion_id: str):
    """Turnos completos de una sesión, mapeados al formato de la Web UI."""
    from core.memory.memory_manager import obtener_turnos_por_sesion

    try:
        turnos = obtener_turnos_por_sesion(sesion_id)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "sesion_id": sesion_id, "messages": [], "error": str(e)}

    mensajes = [
        {
            # Misma convención que la TUI (chat_panel.mostrar_sesion):
            # rol "usuario" → user; cualquier otro rol → assistant.
            "role": "user" if t.get("rol") == "usuario" else "assistant",
            "text": t.get("texto") or "",
            "fecha": t.get("fecha") or "",
        }
        for t in turnos
    ]
    return {"ok": True, "sesion_id": sesion_id, "messages": mensajes}
