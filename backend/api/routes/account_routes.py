"""
account_routes.py — Perfil de usuario, avatar y preferencias de la Web UI.

Persiste en config.local.json bajo la clave ACCOUNT_PROFILE.
"""
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import base64
import hashlib
from pathlib import Path

from core.config.config_manager import get_config_manager
from core.config.settings import BASE_AETHER

router = APIRouter()

_AVATAR_DIR = Path(BASE_AETHER) / "avatars"
_MAX_AVATAR_SIZE = 2 * 1024 * 1024  # 2 MB
_ALLOWED_AVATAR_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


_GENEROS_VALIDOS = {"hombre", "mujer", "no_binario", "otro", ""}


def _pronombres_de(genero: str, pronombre_custom: str) -> str:
    """Mapeo inclusivo: el agente usa estos pronombres para dirigirse al
    usuario. 'otro' permite pronombre a elección del usuario."""
    return {
        "hombre": "él/lo/suyo",
        "mujer": "ella/la/suya",
        "no_binario": "elle/le/sue",
        "otro": (pronombre_custom or "").strip(),
    }.get(genero, "")


class ProfileBody(BaseModel):
    nombre: Optional[str] = None
    bio: Optional[str] = None
    timezone: Optional[str] = None
    idioma: Optional[str] = None
    genero: Optional[str] = None              # hombre | mujer | no_binario | otro
    pronombre_custom: Optional[str] = None    # obligatorio si genero=otro
    fecha_nacimiento: Optional[str] = None    # ISO AAAA-MM-DD


class PreferencesBody(BaseModel):
    tema: Optional[str] = None  # "dark", "light", "auto"
    modo_compacto: Optional[bool] = None
    notificaciones_push: Optional[bool] = None
    idioma_ui: Optional[str] = None


@router.get("/account/profile")
async def get_profile():
    """Obtiene el perfil del usuario (incluye género → pronombres y fecha de
    nacimiento; la UI no muestra el avatar al agente — es solo para la UI)."""
    cfg = get_config_manager()
    profile = cfg.get("ACCOUNT_PROFILE", {}) or {}
    genero = profile.get("genero", "")
    return {"ok": True, "profile": {
        "nombre": profile.get("nombre", ""),
        "bio": profile.get("bio", ""),
        "timezone": profile.get("timezone", "America/Argentina/Buenos_Aires"),
        "idioma": profile.get("idioma", "es"),
        "genero": genero,
        "pronombres": _pronombres_de(genero, profile.get("pronombre_custom", "")),
        "pronombre_custom": profile.get("pronombre_custom", ""),
        "fecha_nacimiento": profile.get("fecha_nacimiento", ""),
        "avatar_url": profile.get("avatar", ""),
    }}


@router.post("/account/profile")
async def update_profile(body: ProfileBody):
    """Actualiza el perfil del usuario. Valida género y fecha de nacimiento."""
    cfg = get_config_manager()
    profile = cfg.get("ACCOUNT_PROFILE", {}) or {}

    if body.nombre is not None:
        profile["nombre"] = body.nombre.strip()
    if body.bio is not None:
        profile["bio"] = body.bio.strip()
    if body.timezone is not None:
        profile["timezone"] = body.timezone.strip()
    if body.idioma is not None:
        profile["idioma"] = body.idioma.strip()
    if body.genero is not None:
        genero = body.genero.strip().lower()
        if genero not in _GENEROS_VALIDOS:
            return JSONResponse(
                {"ok": False, "error": f"Género inválido: {genero!r} "
                 f"(válidos: hombre, mujer, no_binario, otro)"},
                status_code=400)
        if genero == "otro" and not (body.pronombre_custom
                                     or profile.get("pronombre_custom", "")).strip():
            return JSONResponse(
                {"ok": False, "error": "Con género 'otro' elegí un pronombre."},
                status_code=400)
        profile["genero"] = genero
    if body.pronombre_custom is not None:
        profile["pronombre_custom"] = body.pronombre_custom.strip()
    if body.fecha_nacimiento is not None:
        fecha = body.fecha_nacimiento.strip()
        if fecha:
            try:
                from datetime import date as _date
                # Valida formato ISO real (rechaza 2025-02-30, etc.)
                _date.fromisoformat(fecha)
            except ValueError:
                return JSONResponse(
                    {"ok": False, "error": "fecha_nacimiento inválida "
                     "(formato AAAA-MM-DD)"},
                    status_code=400)
        profile["fecha_nacimiento"] = fecha

    ok = cfg.set("ACCOUNT_PROFILE", profile, validate=True)
    return {"ok": ok, "profile": profile}


@router.post("/account/avatar")
async def upload_avatar(file: UploadFile = File(...)):
    """Sube avatar (JPG/PNG/WEBP/GIF, máx 2MB)."""
    if file.content_type not in _ALLOWED_AVATAR_TYPES:
        return JSONResponse({"ok": False, "error": "Tipo no permitido (jpg, png, webp, gif)"}, status_code=400)
    
    data = await file.read()
    if len(data) > _MAX_AVATAR_SIZE:
        return JSONResponse({"ok": False, "error": f"Máximo {_MAX_AVATAR_SIZE // (1024*1024)} MB"}, status_code=400)
    
    # Generar nombre único basado en hash
    ext = Path(file.filename).suffix.lower() or ".png"
    hash_name = hashlib.sha256(data).hexdigest()[:16]
    fname = f"avatar_{hash_name}{ext}"
    
    _AVATAR_DIR.mkdir(parents=True, exist_ok=True)
    fpath = _AVATAR_DIR / fname
    fpath.write_bytes(data)
    
    # Actualizar perfil
    cfg = get_config_manager()
    profile = cfg.get("ACCOUNT_PROFILE", {}) or {}
    profile["avatar"] = f"/avatars/{fname}"
    cfg.set("ACCOUNT_PROFILE", profile, validate=True)
    
    return {"ok": True, "avatar_url": profile["avatar"]}


@router.get("/account/preferences")
async def get_preferences():
    """Obtiene preferencias de UI."""
    cfg = get_config_manager()
    prefs = cfg.get("ACCOUNT_PREFERENCES", {}) or {}
    return {"ok": True, "preferences": {
        "tema": prefs.get("tema", "dark"),
        "modo_compacto": prefs.get("modo_compacto", False),
        "notificaciones_push": prefs.get("notificaciones_push", True),
        "idioma_ui": prefs.get("idioma_ui", "es"),
    }}


@router.post("/account/preferences")
async def update_preferences(body: PreferencesBody):
    """Actualiza preferencias de UI."""
    cfg = get_config_manager()
    prefs = cfg.get("ACCOUNT_PREFERENCES", {}) or {}
    
    if body.tema is not None:
        prefs["tema"] = body.tema
    if body.modo_compacto is not None:
        prefs["modo_compacto"] = body.modo_compacto
    if body.notificaciones_push is not None:
        prefs["notificaciones_push"] = body.notificaciones_push
    if body.idioma_ui is not None:
        prefs["idioma_ui"] = body.idioma_ui
    
    ok = cfg.set("ACCOUNT_PREFERENCES", prefs, validate=True)
    return {"ok": ok, "preferences": prefs}
