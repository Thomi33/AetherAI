"""
memory_routes.py — Acceso a la memoria persistente de Aether ("lo que Aether
recuerda"): el resumen acumulativo (rolling summary de resumen_memoria) y los
recuerdos permanentes (tabla recuerdos de la DB de memoria).

Equivale a los comandos de la TUI:
- /memory              → GET/POST /api/memory/summary (ver y editar el resumen)
- /memory consolidar   → POST /api/memory/consolidate (consolidación manual)

Todo lee/escribe la MISMA DB que usa la TUI y el grafo, así que los cambios
hechos desde la Web UI los ve Aether en el próximo turno (el resumen también
se espeja en la memoria RAM del runtime vía AetherService).
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()


# ══════════════════════════ RESUMEN (rolling summary) ═══════════════════════

@router.get("/memory/summary")
async def get_summary():
    """Resumen acumulativo de memoria: texto, ultimo_turno_id y actualizado."""
    from core.memory.memory_manager import obtener_resumen

    try:
        res = obtener_resumen()
        return {"ok": True, "summary": res}
    except Exception as e:  # noqa: BLE001 — la UI muestra el error
        return {"ok": False, "summary": {"texto": "", "ultimo_turno_id": 0,
                                          "actualizado": ""}, "error": str(e)}


class SummaryBody(BaseModel):
    texto: str


@router.post("/memory/summary")
async def set_summary(body: SummaryBody):
    """
    Edición manual del resumen (como guardar desde el MemoryEditorScreen).
    No toca ultimo_turno_id: la próxima consolidación automática sigue
    integrando desde donde iba, sin re-procesar turnos ya vistos.
    """
    from core.memory.memory_manager import guardar_resumen
    from backend.core.aether_service import AetherService

    try:
        guardar_resumen(body.texto)
        # Espejo en la memoria RAM del motor (igual que _motor_actualizar_resumen
        # de la TUI): el próximo turno ya ve el resumen nuevo sin recargar.
        AetherService.actualizar_resumen_ram(body.texto)
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.post("/memory/consolidate")
async def consolidate():
    """Fuerza la consolidación del resumen con los turnos pendientes."""
    from core.memory.consolidator import consolidar_resumen
    from backend.core.aether_service import AetherService

    try:
        nuevo = consolidar_resumen(forzar=True)
        if nuevo is None:
            return {
                "ok": False,
                "error": "No había turnos nuevos para consolidar (o la "
                         "consolidación falló; ver logs del backend).",
            }
        AetherService.actualizar_resumen_ram(nuevo)
        return {"ok": True, "texto": nuevo}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ══════════════════════════ RECUERDOS (hechos permanentes) ══════════════════

@router.get("/memory/recuerdos")
async def listar_recuerdos(categoria: str = "", importancia_min: int = 1,
                           limit: int = 50):
    """Recuerdos guardados por Aether (los que inyecta el context builder)."""
    from core.memory.memory_manager import obtener_recuerdos

    try:
        recs = obtener_recuerdos(
            categoria=categoria or None,
            importancia_min=max(1, min(int(importancia_min), 10)),
            limit=max(1, min(int(limit), 200)),
        )
        return {"ok": True, "recuerdos": recs}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "recuerdos": [], "error": str(e)}


class RecuerdoBody(BaseModel):
    contenido: str
    categoria: str = ""
    importancia: int = 1


@router.post("/memory/recuerdos")
async def agregar_recuerdo(body: RecuerdoBody):
    """Guarda un hecho permanente (misma write-path que guardar_recuerdo)."""
    if not body.contenido.strip():
        return JSONResponse(
            {"ok": False, "error": "El contenido no puede estar vacío"},
            status_code=400,
        )
    from core.memory.memory_manager import guardar_recuerdo

    try:
        guardar_recuerdo(
            body.contenido.strip(),
            categoria=body.categoria.strip(),
            importancia=max(1, min(int(body.importancia), 10)),
        )
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.delete("/memory/recuerdos/{recuerdo_id}")
async def borrar_recuerdo(recuerdo_id: int):
    """Borra un recuerdo por id (los ids vienen en GET /api/memory/recuerdos)."""
    from core.memory.memory_manager import borrar_recuerdo as _borrar

    try:
        ok = _borrar(recuerdo_id)
        if not ok:
            return JSONResponse(
                {"ok": False, "error": f"No existe el recuerdo {recuerdo_id}"},
                status_code=404,
            )
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ══════════════════════════ MEMORY IMPORT ═══════════════════════════════════

import csv
import hashlib
import io
import json
import re

# ⚠️ IMPORTANTE: NO redeclarar `router` aquí. Una versión anterior hacía
# `router = APIRouter()` en este punto del archivo, lo que descartaba en
# silencio TODAS las rutas registradas arriba (/memory/summary,
# /memory/recuerdos, /memory/consolidate, …): FastAPI solo recibe el router
# final del módulo, así que la Web UI recibía 404 Not Found en toda la
# pestaña Memoria y 405 Method Not Allowed en "Guardar resumen".


class MemoryImportBody(BaseModel):
    source: str  # "chatgpt", "claude", "gemini", "custom"
    format: str  # "json", "markdown", "csv"
    data: dict   # raw export data
    options: dict = {}  # merge_strategy: "append" | "replace" | "smart"


def _get_existing_hashes() -> set:
    """Obtiene hashes de recuerdos existentes para deduplicación."""
    from core.memory.memory_manager import obtener_recuerdos
    try:
        existing = obtener_recuerdos(limit=2000)
        hashes = set()
        for m in existing:
            key = hashlib.sha256(m.get("contenido", "")[:200].encode()).hexdigest()[:16]
            hashes.add(key)
        return hashes
    except Exception:
        return set()


def _parse_chatgpt_export(data: dict) -> tuple[str, list[dict]]:
    """Parse ChatGPT conversations.json export."""
    memories = []
    try:
        conversations = data.get("conversations", [])
        for conv in conversations:
            mapping = conv.get("mapping", {})
            for msg_id, node in mapping.items():
                msg = node.get("message")
                if not msg:
                    continue
                role = msg.get("author", {}).get("role", "")
                content = msg.get("content", {})
                parts = content.get("parts", [])
                if not parts:
                    continue
                text = "\n".join(str(p) for p in parts if p)
                if not text.strip():
                    continue
                if role in ("user", "assistant"):
                    cat = "chatgpt-user" if role == "user" else "chatgpt-assistant"
                    memories.append({
                        "contenido": text.strip()[:4000],
                        "categoria": cat,
                        "importancia": 3,
                    })
    except Exception:
        pass
    total = len(memories)
    summary = f"Importados {total} mensajes de ChatGPT"
    return summary, memories


def _parse_claude_export(data: dict) -> tuple[str, list[dict]]:
    """Parse Claude export (JSON or markdown)."""
    memories = []
    try:
        if "conversations" in data:
            for conv in data["conversations"]:
                for msg in conv.get("messages", []):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(c.get("text", "") for c in content if c.get("type") == "text")
                    if not content.strip():
                        continue
                    if role in ("user", "assistant"):
                        cat = "claude-user" if role == "user" else "claude-assistant"
                        memories.append({
                            "contenido": content.strip()[:4000],
                            "categoria": cat,
                            "importancia": 3,
                        })
        elif "text" in data:
            s, m = _parse_markdown(data["text"])
            memories.extend(m)
    except Exception:
        pass
    total = len(memories)
    return f"Importados {total} mensajes de Claude", memories


def _parse_gemini_export(data: dict) -> tuple[str, list[dict]]:
    """Parse Google Takeout Gemini export."""
    memories = []
    try:
        if "conversations" in data:
            for conv in data["conversations"]:
                for turn in conv.get("turns", []):
                    role = turn.get("author", "")
                    text = turn.get("text", "")
                    if not text.strip():
                        continue
                    if role in ("user", "model"):
                        cat = "gemini-user" if role == "user" else "gemini-model"
                        memories.append({
                            "contenido": text.strip()[:4000],
                            "categoria": cat,
                            "importancia": 3,
                        })
    except Exception:
        pass
    total = len(memories)
    return f"Importados {total} mensajes de Gemini", memories


def _parse_custom_json(data: dict) -> tuple[str, list[dict]]:
    """Parse custom JSON with schema: {messages: [{role, content, timestamp}], summary: ""}."""
    memories = []
    summary = data.get("summary", "")
    try:
        for msg in data.get("messages", []):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not content.strip():
                continue
            cat = f"import-{role}"
            memories.append({
                "contenido": content.strip()[:4000],
                "categoria": cat,
                "importancia": 3,
            })
    except Exception:
        pass
    total = len(memories)
    return summary or f"Importados {total} mensajes (custom)", memories


def _parse_markdown(text: str) -> tuple[str, list[dict]]:
    """Simple markdown parser: busca bloques de diálogo."""
    memories = []
    lines = text.split("\n")
    current_role = None
    current_content = []
    for line in lines:
        line = line.strip()
        if line.startswith("**User**:") or line.startswith("**Usuario**:"):
            if current_role and current_content:
                memories.append({
                    "contenido": "\n".join(current_content).strip()[:4000],
                    "categoria": f"import-{current_role.lower()}",
                    "importancia": 3,
                })
            current_role = "User" if "User" in line or "Usuario" in line else "Assistant"
            current_content = [line.split(":", 1)[1].strip()]
        elif line.startswith("**Assistant**:") or line.startswith("**Asistente**:"):
            if current_role and current_content:
                memories.append({
                    "contenido": "\n".join(current_content).strip()[:4000],
                    "categoria": f"import-{current_role.lower()}",
                    "importancia": 3,
                })
            current_role = "Assistant" if "Assistant" in line else "Asistente"
            current_content = [line.split(":", 1)[1].strip()]
        elif line and current_role:
            current_content.append(line)
    if current_role and current_content:
        memories.append({
            "contenido": "\n".join(current_content).strip()[:4000],
            "categoria": f"import-{current_role.lower()}",
            "importancia": 3,
        })
    return f"Importados {len(memories)} mensajes (markdown)", memories


def _parse_csv(text: str) -> tuple[str, list[dict]]:
    """Parse CSV with columns: role, content, timestamp, category, importance."""
    memories = []
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            content = row.get("content", "").strip()
            if not content:
                continue
            role = row.get("role", "custom")
            cat = row.get("category", f"import-{role}")
            imp = int(row.get("importance", "3"))
            memories.append({
                "contenido": content[:4000],
                "categoria": cat,
                "importancia": max(1, min(imp, 10)),
            })
    except Exception:
        pass
    return f"Importados {len(memories)} filas (CSV)", memories


def _get_existing_hashes() -> set:
    """Obtiene hashes de recuerdos existentes para deduplicación."""
    from core.memory.memory_manager import obtener_recuerdos
    try:
        existing = obtener_recuerdos(limit=2000)
        hashes = set()
        for m in existing:
            key = hashlib.sha256(m.get("contenido", "")[:200].encode()).hexdigest()[:16]
            hashes.add(key)
        return hashes
    except Exception:
        return set()


def _deduplicate_memories(new_memories: list[dict], existing_hashes: set) -> list[dict]:
    """Dedupe by content hash (SHA256 of first 200 chars)."""
    unique = []
    for m in new_memories:
        key = hashlib.sha256(m["contenido"][:200].encode()).hexdigest()[:16]
        if key not in existing_hashes:
            existing_hashes.add(key)
            unique.append(m)
    return unique


@router.post("/memory/import")
async def import_memory(body: MemoryImportBody):
    """Importa memoria de otras IAs (ChatGPT, Claude, Gemini, custom JSON/MD/CSV).
    
    Body:
    {
      "source": "chatgpt|claude|gemini|custom",
      "format": "json|markdown|csv",
      "data": {...},  # raw export data
      "options": {"merge_strategy": "append|replace|smart"}  # default: smart
    }
    """
    source = body.source.lower()
    fmt = body.format.lower()
    data = body.data
    merge = body.options.get("merge_strategy", "smart")

    # Parse according to source/format
    if body.source == "chatgpt" and body.format == "json":
        summary, memories = _parse_chatgpt_export(body.data)
    elif body.source == "claude" and body.format in ("json", "markdown"):
        summary, memories = _parse_claude_export(body.data)
    elif body.source == "gemini" and body.format == "json":
        summary, memories = _parse_gemini_export(body.data)
    elif body.source == "custom" and body.format == "json":
        summary, memories = _parse_custom_json(body.data)
    elif body.format == "markdown":
        summary, memories = _parse_markdown(json.dumps(body.data) if isinstance(body.data, dict) else str(body.data))
    elif body.format == "csv":
        summary, memories = _parse_csv(json.dumps(body.data) if isinstance(body.data, dict) else str(body.data))
    else:
        return JSONResponse(
            {"ok": False, "error": f"Combinación no soportada: source={body.source}, format={body.format}"},
            status_code=400,
        )

    # Dedupe
    existing_hashes = _get_existing_hashes()
    memories = _deduplicate_memories(memories, existing_hashes)

    # Persist
    from core.memory.memory_manager import guardar_recuerdo
    saved = 0
    for m in memories:
        try:
            guardar_recuerdo(
                contenido=m["contenido"],
                categoria=m["categoria"],
                importancia=m["importancia"],
            )
            saved += 1
        except Exception:
            pass

    return {
        "ok": True,
        "summary": summary,
        "total_parsed": len(memories),
        "saved": saved,
        "duplicates_skipped": len(memories) - saved,
        "merge_strategy": merge,
    }


# ══════════════════ IMPORTAR MEMORIA VÍA PROMPT (roundtrip) ══════════════════
# Flujo: Aether genera un "prompt de extracción" → el usuario se lo pega a la
# otra IA (ChatGPT, Claude, Gemini, …) → esa IA contesta con un JSON con lo
# que sabe del usuario → el usuario pega esa respuesta acá y Aether la importa
# a sus recuerdos permanentes (deduplicados) y, si viene, suma el resumen.

PROMPT_EXTRACCION_MEMORIA = """\
Actuá como un extractor de memoria. Analizá TODO el historial de conversaciones que tengas conmigo y devolvé UN SOLO OBJETO JSON (sin texto extra, sin ``` ni comentarios) con este formato exacto:

{
  "resumen": "Párrafo corto con quién soy, cómo trabajo y en qué ando (si lo sabés; si no, string vacío).",
  "perfil": {
    "nombre": "solo si te lo dije",
    "otros": "cualquier dato personal relevante que yo haya compartido"
  },
  "recuerdos": [
    {
      "contenido": "Un hecho concreto que aprendiste de mí (preferencia, proyecto, dato, decisión...), en una o dos frases.",
      "categoria": "preferencias | proyectos | personal | trabajo | otro",
      "importancia": 5
    }
  ]
}

Reglas:
- Solo hechos que realmente te conté o demostré. Nada inventado ni supuesto.
- "importancia" va de 1 (trivia) a 10 (crítico para mi trabajo o vida).
- Entre 5 y 40 recuerdos: lo más útil para que otro asistente te reemplace.
- Respondé SOLO con el JSON. Sin introducción ni cierre."""


def _extraer_json_tolerante(texto: str) -> dict | None:
    """Saca el primer objeto JSON de la respuesta pegada: tolera ```json
    fences, texto antes/después y comas colgantes (los modelos se equivocan
    seguido). Devuelve None si no hay JSON parseable."""
    if not texto or not texto.strip():
        return None
    t = texto.strip()
    # Quitar fences ```json ... ``` / ``` ... ```
    t = re.sub(r"^```(?:json|JSON)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    ini, fin = t.find("{"), t.rfind("}")
    if ini < 0 or fin <= ini:
        return None
    candidato = t[ini:fin + 1]
    try:
        data = json.loads(candidato)
    except json.JSONDecodeError:
        # Intento de rescate: comas antes de } o ]
        try:
            rescatado = re.sub(r",(\s*[}\]])", r"\1", candidato)
            data = json.loads(rescatado)
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


class ImportTextoBody(BaseModel):
    """La respuesta cruda que devolvió la otra IA al prompt de extracción."""
    texto: str
    categoria_default: str = "importado"
    importancia_default: int = 5
    actualizar_resumen: bool = True


@router.get("/memory/import-prompt")
async def import_prompt():
    """Prompt de extracción para pegarle a otra IA (ChatGPT, Claude, …)."""
    return {"ok": True, "prompt": PROMPT_EXTRACCION_MEMORIA}


@router.post("/memory/import-text")
async def import_text(body: ImportTextoBody):
    """Importa la respuesta (JSON) de la otra IA a los recuerdos de Aether."""
    data = _extraer_json_tolerante(body.texto)
    if data is None:
        return JSONResponse(
            {"ok": False,
             "error": "No se encontró un JSON válido en el texto pegado. "
                      "Pedile a la otra IA que responda SOLO con el JSON "
                      "(o usá 'Obtener prompt' de nuevo)."},
            status_code=400,
        )

    # ── Recuerdos: normalizar strings/dicts, dedupe, persistir ──
    crudos = data.get("recuerdos") or []
    if not isinstance(crudos, list):
        crudos = []
    cat_def = (body.categoria_default or "importado").strip() or "importado"
    imp_def = max(1, min(int(body.importancia_default or 5), 10))

    memorias: list[dict] = []
    for r in crudos:
        if isinstance(r, str) and r.strip():
            memorias.append({"contenido": r.strip()[:4000],
                             "categoria": cat_def, "importancia": imp_def})
        elif isinstance(r, dict):
            contenido = str(r.get("contenido") or r.get("texto") or "").strip()
            if not contenido:
                continue
            try:
                imp = max(1, min(int(r.get("importancia", imp_def)), 10))
            except (TypeError, ValueError):
                imp = imp_def
            memorias.append({
                "contenido": contenido[:4000],
                "categoria": str(r.get("categoria") or cat_def).strip(),
                "importancia": imp,
            })

    # Perfil (opcional): se guarda como recuerdo (nunca pisa ACCOUNT_PROFILE)
    perfil = data.get("perfil")
    if isinstance(perfil, dict) and perfil:
        pares = [f"{k}: {v}" for k, v in perfil.items() if str(v).strip()]
        if pares:
            memorias.append({"contenido": "Perfil importado de otra IA — "
                                          + "; ".join(pares)[:4000],
                             "categoria": "perfil-importado",
                             "importancia": 6})

    total_crudos = len(crudos) + (1 if isinstance(perfil, dict) and perfil else 0)
    memorias = _deduplicate_memories(memorias, _get_existing_hashes())
    from core.memory.memory_manager import guardar_recuerdo
    guardados, errores = 0, []
    for m in memorias:
        try:
            guardar_recuerdo(m["contenido"], categoria=m["categoria"],
                             importancia=m["importancia"])
            guardados += 1
        except Exception:  # noqa: BLE001 — uno malo no frena al resto
            errores.append(m["contenido"][:80])

    # ── Resumen (opcional): se ANEXA al rolling summary, nunca lo pisa ──
    resumen_agregado = False
    resumen = str(data.get("resumen") or "").strip()
    if resumen and body.actualizar_resumen:
        try:
            from core.memory.memory_manager import obtener_resumen, guardar_resumen
            from backend.core.aether_service import AetherService
            actual = (obtener_resumen().get("texto") or "").strip()
            nuevo = ((actual + "\n\n[Memoria importada de otra IA]:\n" + resumen)
                     if actual else resumen).strip()
            guardar_resumen(nuevo)
            AetherService.actualizar_resumen_ram(nuevo)
            resumen_agregado = True
        except Exception as e:  # noqa: BLE001
            errores.append(f"resumen: {e}")

    return {
        "ok": True,
        "recuerdos_importados": guardados,
        "duplicados_salteados": max(0, total_crudos - guardados - len(errores)),
        "resumen_agregado": resumen_agregado,
        "errores": errores,
    }
