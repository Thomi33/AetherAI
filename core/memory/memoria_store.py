"""memoria_store.py — Store ÚNICO de memoria persistente de Aether, en JSON.

Reemplaza SQLite por completo. Un solo archivo:

    <BASE_AETHER>/db/memoria.json

Estructura:

    {
      "version": 2,
      "core":          {"clave": "valor", ...},
      "resumen":       {"texto": str, "ultimo_turno_id": int, "actualizado": str},
      "conversaciones": [{"id": int, "fecha": str, "rol": str, "texto": str,
                          "sesion_id": str, "tema": str, "proyecto": str}, ...],
      "comandos":       [{"id": int, "fecha": str, "orden": str, "cmd": str,
                          "sesion_id": str, "exitoso": bool}, ...],
      "recuerdos":      [{"id": int, "fecha": str, "categoria": str,
                          "contenido": str, "importancia": int}, ...]
    }

Robustez (mismo contrato que exige el backlog):
- Escritura ATÓMICA (tmp + os.replace): un corte a mitad nunca deja el
  archivo roto.
- Backup rotativo: antes de cada escritura, el archivo vigente sano se copia
  a memoria.json.bak.
- Recuperación: JSON corrupto → restaura desde .bak; si ambos fallan, el
  archivo roto se aparta (memoria.corrupt-<ts>.json) y se arranca vacío.
  NUNCA se lanza excepción hacia arriba.
- Migración única: si memoria.json no existe y existe la DB legacy
  (current.db, sqlite), se importa TODO (core/resumen/conversaciones/
  comandos/recuerdos) una sola vez. Es la única parte del sistema que sabe
  leer el formato viejo; después de migrar, sqlite no se toca nunca más.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from core.config.settings import BASE_AETHER

RUTA_JSON = Path(BASE_AETHER) / "db" / "memoria.json"
RUTA_BAK = RUTA_JSON.with_suffix(".json.bak")

VERSION = 2
# Techos para que el archivo nunca crezca sin control (el historial largo
# lo resume el rolling summary; esto es detalle reciente para sesiones/UI).
MAX_TURNOS_JSON = 2000      # conversaciones conservadas
MAX_COMANDOS_JSON = 500

_SECCIONES_LISTA = ("conversaciones", "comandos", "recuerdos")


def _vacio() -> dict:
    return {
        "version": VERSION,
        "core": {},
        "resumen": {"texto": "", "ultimo_turno_id": 0, "actualizado": ""},
        "conversaciones": [],
        "comandos": [],
        "recuerdos": [],
    }


def _normalizar(data: object) -> dict | None:
    """Valida/normaliza un payload crudo leído de disco; None si no sirve."""
    if not isinstance(data, dict):
        return None
    out = _vacio()
    if isinstance(data.get("core"), dict):
        out["core"] = {str(k): str(v) for k, v in data["core"].items()}
    res = data.get("resumen")
    if isinstance(res, dict):
        out["resumen"] = {
            "texto": str(res.get("texto") or ""),
            "ultimo_turno_id": int(res.get("ultimo_turno_id") or 0),
            "actualizado": str(res.get("actualizado") or ""),
        }
    for seccion in _SECCIONES_LISTA:
        items = data.get(seccion)
        if isinstance(items, list):
            out[seccion] = [it for it in items if isinstance(it, dict)]
    return out


def _migrar_legacy() -> dict | None:
    """Importa la DB legacy (current.db, sqlite) UNA sola vez.

    Es el ÚNICO punto del sistema que conoce el formato viejo. sqlite3 se
    importa lazy acá adentro: tras la migración, este código no vuelve a
    ejecutarse jamás (memoria.json ya existe).
    """
    legacy = Path(BASE_AETHER) / "db" / "current.db"
    if not legacy.exists():
        return None
    try:
        import sqlite3
        con = sqlite3.connect(legacy)
        con.row_factory = sqlite3.Row
        try:
            def _tabla(sql, params=()):
                try:
                    return [dict(f) for f in con.execute(sql, params).fetchall()]
                except sqlite3.Error:
                    return []

            data = _vacio()
            data["core"] = {f["clave"]: f["valor"] for f in
                            _tabla("SELECT clave, valor FROM core_memory")}
            fila = _tabla("SELECT texto, ultimo_turno_id, actualizado "
                          "FROM resumen_memoria WHERE id = 1")
            if fila:
                data["resumen"] = {"texto": str(fila[0].get("texto") or ""),
                                   "ultimo_turno_id": int(fila[0].get("ultimo_turno_id") or 0),
                                   "actualizado": str(fila[0].get("actualizado") or "")}
            data["conversaciones"] = _tabla(
                "SELECT id, fecha, rol, texto, sesion_id, tema, proyecto "
                "FROM conversaciones ORDER BY id ASC")[-MAX_TURNOS_JSON:]
            data["comandos"] = _tabla(
                "SELECT id, fecha, orden, cmd, sesion_id, exitoso "
                "FROM comandos ORDER BY id ASC")[-MAX_COMANDOS_JSON:]
            data["recuerdos"] = _tabla(
                "SELECT id, fecha, categoria, contenido, importancia "
                "FROM recuerdos ORDER BY id ASC")
        finally:
            con.close()
    except Exception as e:
        print(f"[MEMORIA] ⚠️ no se pudo migrar la DB legacy: {e}")
        return None
    n = sum(len(data[s]) for s in _SECCIONES_LISTA) + len(data["core"])
    print(f"[MEMORIA] Migración sqlite → JSON completa ({n} registros).")
    return data


def cargar() -> dict:
    """Carga el store aplicando recuperación/migración. Nunca lanza."""
    if RUTA_JSON.exists():
        try:
            data = json.loads(RUTA_JSON.read_text(encoding="utf-8"))
            ok = _normalizar(data)
            if ok is not None:
                return ok
        except Exception:
            pass
        # Corrupto → backup; sin backup → cuarentena + vacío.
        bak = None
        if RUTA_BAK.exists():
            try:
                bak = _normalizar(json.loads(RUTA_BAK.read_text(encoding="utf-8")))
            except Exception:
                bak = None
        if bak is not None:
            try:
                shutil.copy2(RUTA_BAK, RUTA_JSON)
            except Exception:
                pass
            print("[MEMORIA] memoria.json corrupto → restaurado desde .bak")
            return bak
        try:
            RUTA_JSON.rename(RUTA_JSON.with_name(
                f"memoria.corrupt-{datetime.now():%Y%m%d-%H%M%S}.json"))
        except Exception:
            pass
        print("[MEMORIA] ⚠️ memoria.json y .bak corruptos; arranque vacío")
        return _vacio()

    migrado = _migrar_legacy()
    if migrado is not None:
        guardar(migrado)
        return migrado
    return _vacio()


def guardar(data: dict) -> None:
    """Escritura atómica + backup del estado previo sano."""
    RUTA_JSON.parent.mkdir(parents=True, exist_ok=True)
    sano = _normalizar(data) or _vacio()
    if RUTA_JSON.exists():
        try:
            if _normalizar(json.loads(RUTA_JSON.read_text(encoding="utf-8"))) is not None:
                shutil.copy2(RUTA_JSON, RUTA_BAK)
        except Exception:
            pass
    tmp = RUTA_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sano, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, RUTA_JSON)


def proximo_id(data: dict, seccion: str) -> int:
    """Siguiente id entero de una sección (monótono; nunca reusa ids)."""
    ids = []
    for it in data[seccion]:
        try:
            ids.append(int(it.get("id") or 0))
        except (TypeError, ValueError):
            continue
    return max(ids, default=0) + 1
