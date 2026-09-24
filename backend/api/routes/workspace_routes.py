"""
workspace_routes.py — Proyectos y Tareas de la Web UI.

No existían en el core: se implementan como registros JSON livianos dentro de
BASE_AETHER (~/Aether), junto a la DB de memoria — datos de runtime, fuera del
repo. El agente trabaja sobre las rutas con sus tools de fs/shell; estos
registros son el índice que la UI (y en el futuro el propio Aether) consulta.

- Proyectos: {name, path, descripcion, creado}
- Tareas:    {id, titulo, estado: pendiente|en_curso|hecha, notas, creada}
"""

import json
import re
import time
import uuid
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

_RE_NOMBRE = re.compile(r"^[^/\\\0]{1,80}$")  # cualquier texto sin separadores
_ESTADOS_TAREA = ("pendiente", "en_curso", "hecha")


def _ruta(nombre_archivo: str) -> Path:
    from core.config.settings import BASE_AETHER
    base = Path(BASE_AETHER)
    base.mkdir(parents=True, exist_ok=True)
    return base / nombre_archivo


def _leer(nombre_archivo: str) -> list:
    ruta = _ruta(nombre_archivo)
    if not ruta.exists():
        return []
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001 — JSON corrupto no rompe la UI
        return []


def _guardar(nombre_archivo: str, items: list) -> None:
    _ruta(nombre_archivo).write_text(
        json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ══════════════════════════ PROYECTOS ══════════════════════════

@router.get("/proyectos")
async def listar_proyectos():
    items = _leer("proyectos.json")
    for p in items:
        # Marca si la ruta sigue existiendo (info útil, sin borrar nada).
        p["existe"] = Path(p.get("path", "")).expanduser().is_dir()
    return {"ok": True, "proyectos": items}


class ProyectoBody(BaseModel):
    name: str
    path: str
    descripcion: str = ""


@router.post("/proyectos")
async def crear_proyecto(body: ProyectoBody):
    nombre = body.name.strip()
    ruta = body.path.strip()
    if not nombre or not _RE_NOMBRE.match(nombre):
        return JSONResponse({"ok": False, "error": "Nombre inválido"},
                            status_code=400)
    expandida = Path(ruta).expanduser()
    if not expandida.is_dir():
        return JSONResponse(
            {"ok": False,
             "error": f"La ruta no existe o no es una carpeta: {ruta}"},
            status_code=400)
    items = _leer("proyectos.json")
    if any(p["name"] == nombre for p in items):
        return JSONResponse(
            {"ok": False, "error": f"Ya existe el proyecto '{nombre}'"},
            status_code=409)
    items.append({
        "name": nombre,
        "path": str(expandida),
        "descripcion": body.descripcion.strip(),
        "creado": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _guardar("proyectos.json", items)
    return {"ok": True}


@router.delete("/proyectos/{nombre}")
async def borrar_proyecto(nombre: str):
    items = _leer("proyectos.json")
    nuevos = [p for p in items if p.get("name") != nombre]
    if len(nuevos) == len(items):
        return JSONResponse(
            {"ok": False, "error": f"No existe el proyecto '{nombre}'"},
            status_code=404)
    _guardar("proyectos.json", nuevos)
    return {"ok": True}


# ══════════════════════════ TAREAS ══════════════════════════

@router.get("/tareas")
async def listar_tareas():
    return {"ok": True, "tareas": _leer("tareas.json")}


class TareaBody(BaseModel):
    titulo: str
    notas: str = ""
    estado: str = "pendiente"


@router.post("/tareas")
async def crear_tarea(body: TareaBody):
    titulo = body.titulo.strip()
    if not titulo:
        return JSONResponse(
            {"ok": False, "error": "El título es obligatorio"}, status_code=400)
    if body.estado not in _ESTADOS_TAREA:
        return JSONResponse(
            {"ok": False,
             "error": f"Estado inválido. Opciones: {', '.join(_ESTADOS_TAREA)}"},
            status_code=400)
    items = _leer("tareas.json")
    tarea = {
        "id": uuid.uuid4().hex[:8],
        "titulo": titulo,
        "notas": body.notas.strip(),
        "estado": body.estado,
        "creada": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    items.insert(0, tarea)
    _guardar("tareas.json", items)
    return {"ok": True, "tarea": tarea}


class TareaPatch(BaseModel):
    titulo: str | None = None
    notas: str | None = None
    estado: str | None = None


@router.patch("/tareas/{tarea_id}")
async def editar_tarea(tarea_id: str, body: TareaPatch):
    items = _leer("tareas.json")
    tarea = next((t for t in items if t.get("id") == tarea_id), None)
    if tarea is None:
        return JSONResponse(
            {"ok": False, "error": f"No existe la tarea '{tarea_id}'"},
            status_code=404)
    if body.estado is not None and body.estado not in _ESTADOS_TAREA:
        return JSONResponse(
            {"ok": False,
             "error": f"Estado inválido. Opciones: {', '.join(_ESTADOS_TAREA)}"},
            status_code=400)
    if body.titulo is not None:
        tarea["titulo"] = body.titulo.strip() or tarea["titulo"]
    if body.notas is not None:
        tarea["notas"] = body.notas.strip()
    if body.estado is not None:
        tarea["estado"] = body.estado
    _guardar("tareas.json", items)
    return {"ok": True, "tarea": tarea}


@router.delete("/tareas/{tarea_id}")
async def borrar_tarea(tarea_id: str):
    items = _leer("tareas.json")
    nuevos = [t for t in items if t.get("id") != tarea_id]
    if len(nuevos) == len(items):
        return JSONResponse(
            {"ok": False, "error": f"No existe la tarea '{tarea_id}'"},
            status_code=404)
    _guardar("tareas.json", nuevos)
    return {"ok": True}
