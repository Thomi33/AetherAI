"""
skills_routes.py — Catálogo de skills de Aether (core/skills/registry.py).

Una skill es una carpeta <RUTA_SKILLS>/<nombre>/SKILL.md con frontmatter
'name:' y 'description:'. Son CONOCIMIENTO (no tools): el modelo las lee con
fs_read antes de encarar una tarea recurrente.

Endpoints:
- GET    /api/skills          → lista (name, description, path)
- GET    /api/skills/{name}   → + contenido completo del SKILL.md
- POST   /api/skills          → crea <RUTA_SKILLS>/<name>/SKILL.md
- DELETE /api/skills/{name}   → borra la carpeta de la skill
"""

import os
import re
import shutil

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

# Nombre de skill válido: minúsculas, números, guiones (como las carpetas
# existentes: 'crear-proyecto-python'). Se usa también como nombre de carpeta,
# así que va saneado estricto.
_RE_NOMBRE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")

_MAX_SKILL_CHARS = 60000  # lectura acotada del SKILL.md


def _ruta_skill(nombre: str) -> str | None:
    """Ruta ABS de la carpeta de la skill, o None si el nombre es inválido."""
    if not _RE_NOMBRE.match(nombre):
        return None
    from core.config.settings import RUTA_SKILLS
    return os.path.join(str(RUTA_SKILLS), nombre)


@router.get("/skills")
async def listar():
    from core.skills.registry import listar_skills

    try:
        return {"ok": True, "skills": listar_skills()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "skills": [], "error": str(e)}


@router.get("/skills/{nombre}")
async def detalle(nombre: str):
    ruta = _ruta_skill(nombre)
    if ruta is None:
        return JSONResponse({"ok": False, "error": "Nombre inválido"},
                            status_code=400)
    skill_md = os.path.join(ruta, "SKILL.md")
    if not os.path.isfile(skill_md):
        return JSONResponse({"ok": False, "error": f"No existe '{nombre}'"},
                            status_code=404)
    try:
        with open(skill_md, "r", encoding="utf-8") as f:
            contenido = f.read(_MAX_SKILL_CHARS)
        return {"ok": True, "name": nombre, "content": contenido,
                "path": os.path.abspath(skill_md)}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


class SkillBody(BaseModel):
    name: str
    description: str
    content: str = ""  # cuerpo libre (debajo del frontmatter)


@router.post("/skills")
async def crear(body: SkillBody):
    ruta = _ruta_skill(body.name.strip().lower())
    if ruta is None:
        return JSONResponse(
            {"ok": False,
             "error": "Nombre inválido: minúsculas, números y guiones "
                      "(ej. 'crear-api-rest')"},
            status_code=400,
        )
    descripcion = body.description.strip()
    if not descripcion:
        return JSONResponse(
            {"ok": False, "error": "La descripción es obligatoria "
                                   "(sin ella el registry ignora la skill)"},
            status_code=400,
        )
    if os.path.exists(ruta):
        return JSONResponse(
            {"ok": False, "error": f"Ya existe la skill '{body.name}'"},
            status_code=409,
        )

    texto = (
        f"---\nname: {body.name.strip().lower()}\n"
        f"description: {descripcion.splitlines()[0]}\n---\n\n"
        f"{body.content.strip()}\n"
    )
    try:
        os.makedirs(ruta, exist_ok=False)
        with open(os.path.join(ruta, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(texto)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.delete("/skills/{nombre}")
async def borrar(nombre: str):
    ruta = _ruta_skill(nombre)
    if ruta is None:
        return JSONResponse({"ok": False, "error": "Nombre inválido"},
                            status_code=400)
    if not os.path.isdir(ruta):
        return JSONResponse({"ok": False, "error": f"No existe '{nombre}'"},
                            status_code=404)
    try:
        shutil.rmtree(ruta)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
