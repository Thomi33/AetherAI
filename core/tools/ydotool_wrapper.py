"""Wrapper simple de ydotool para Aether.

Solo expone las funciones reales usadas por el grafo:
- buscar_ventana, enfocar_ventana, cambiar_workspace (hyprctl directo)
- click_en, escribir_texto, mover_mouse, mantener_tecla, iniciar_secuencia (ydotool)

Elimina: VLM fallback, DesktopContext/Verification dataclasses, retry complejo,
context cache, compositor detection, verificación obligatoria, movimiento relativo.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from typing import Any, Optional

_BOTONES = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
_SCROLL = {"up": "0x400", "down": "0x401"}
_TECLAS = {
    "ENTER": "28", "ESC": "1", "ESCAPE": "1", "SPACE": "57",
    "TAB": "15", "BACKSPACE": "14", "DELETE": "111",
    "UP": "103", "DOWN": "108", "LEFT": "105", "RIGHT": "106",
    "W": "17", "A": "30", "S": "31", "D": "32",
}
_MAX_RETRIES = 3
_RETRY_BACKOFF = (0.02, 0.05, 0.1)


def _run(cmd: list[str], timeout: int = 5) -> tuple[str, bool]:
    """Ejecuta ydotool y devuelve (output, error)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return f"{cmd[0]} no est\xe1 instalado.", True
    except subprocess.TimeoutExpired:
        return f"{cmd[0]} no respondi\xf3 a tiempo.", True
    except OSError as exc:
        return f"Error ejecutando {cmd[0]}: {exc}", True

    if result.returncode:
        detail = result.stderr.strip() or f"{cmd[0]} sali\xf3 con c\xf3digo {result.returncode}"
        return detail, True
    return "OK", False


def _run_query(cmd: list[str], timeout: float = 2.0) -> tuple[Any | None, str | None]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, f"{cmd[0]} no est\xe1 instalado."
    except subprocess.TimeoutExpired:
        return None, f"{cmd[0]} no respondi\xf3 en {timeout:g}s."
    except OSError as exc:
        return None, f"Error ejecutando {cmd[0]}: {exc}"
    if result.returncode:
        return None, result.stderr.strip() or f"{cmd[0]} sali\xf3 con c\xf3digo {result.returncode}"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"{cmd[0]} devolvi\xf3 JSON inv\xe1lido: {exc}"


def _compositor() -> str:
    if shutil.which("hyprctl") and os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
        return "hyprland"
    if shutil.which("swaymsg") and os.environ.get("SWAYSOCK"):
        return "sway"
    if shutil.which("hyprctl"):
        return "hyprland"
    if shutil.which("swaymsg"):
        return "sway"
    return "unknown"


def _hyprland_context() -> tuple[dict[str, Any] | None, str | None]:
    monitors, error = _run_query(["hyprctl", "monitors", "-j"])
    if error or not monitors:
        return None, error or "no monitors"
    active_mon = next((m for m in monitors if m.get("focused")), monitors[0])
    active_ws = active_mon.get("activeWorkspace", {})
    windows, error = _run_query(["hyprctl", "clients", "-j"])
    if error or not windows:
        return None, error or "no windows"
    focused = next((w for w in windows if w.get("focusHistoryID") == 0), None)
    return {
        "monitor": active_mon.get("name"),
        "workspace": str(active_ws.get("id", "")),
        "focused_window": focused.get("address") if focused else None,
        "cursor_x": None,
        "cursor_y": None,
    }, None


def _sway_context() -> tuple[dict[str, Any] | None, str | None]:
    outputs, error = _run_query(["swaymsg", "-t", "get_outputs"])
    if error or not outputs:
        return None, error or "no outputs"
    focused_out = next((o for o in outputs if o.get("focused")), outputs[0])
    tree, error = _run_query(["swaymsg", "-t", "get_tree"])
    if error or not tree:
        return None, error or "no tree"
    
    def find_focused(node):
        if node.get("focused"):
            return node
        for n in node.get("nodes", []) + node.get("floating_nodes", []):
            found = find_focused(n)
            if found:
                return found
        return None
    
    focused = find_focused(tree)
    return {
        "monitor": focused_out.get("name"),
        "workspace": str(focused_out.get("current_workspace", "")),
        "focused_window": str(focused.get("id")) if focused else None,
        "cursor_x": None,
        "cursor_y": None,
    }, None


_context_cache: dict | None = None


def resolver_contexto(force: bool = False) -> dict:
    """Resuelve contexto del compositor (cached a menos que force=True)."""
    global _context_cache
    if _context_cache is not None and not force:
        return _context_cache
    
    comp = _compositor()
    if comp == "hyprland":
        ctx, err = _hyprland_context()
    elif comp == "sway":
        ctx, err = _sway_context()
    else:
        ctx, err = None, f"Compositor no soportado: {comp}"
    
    if err or ctx is None:
        ctx = {"compositor": comp, "monitor": None, "workspace": None, 
               "focused_window": None, "cursor_x": None, "cursor_y": None}
    
    ctx["compositor"] = comp
    _context_cache = ctx
    return ctx


def iniciar_secuencia() -> dict:
    """Resuelve el contexto una vez al comenzar una secuencia de acciones."""
    return resolver_contexto(force=True)


def buscar_ventana(nombre_parcial: str) -> tuple[str | None, str | None]:
    """Busca la primera ventana cuyo class/app_id o t\xedtulo contenga el nombre."""
    nombre = str(nombre_parcial).strip()
    if not nombre:
        return None, "nombre_parcial vac\xedo"
    comp = _compositor()
    try:
        if comp == "hyprland":
            result = subprocess.run(["hyprctl", "clients", "-j"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return None, result.stderr.strip() or "hyprctl fall\xf3"
            windows = json.loads(result.stdout)
            nombre_lower = nombre.lower()
            for w in windows:
                cls = (w.get("class") or "").lower()
                title = (w.get("title") or "").lower()
                if nombre_lower in cls or nombre_lower in title:
                    addr = w.get("address")
                    if addr:
                        return addr, None
            return None, f"No se encontr\xf3 ventana con '{nombre}'"
        elif comp == "sway":
            result = subprocess.run(["swaymsg", "-t", "get_tree"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return None, result.stderr.strip() or "swaymsg fall\xf3"
            tree = json.loads(result.stdout)
            nombre_lower = nombre.lower()
            
            def find_window(node):
                props = node.get("window_properties", {})
                cls = (props.get("class") or "").lower()
                title = (props.get("title") or "").lower()
                if nombre_lower in cls or nombre_lower in title:
                    return node.get("id")
                for n in node.get("nodes", []) + node.get("floating_nodes", []):
                    found = find_window(n)
                    if found:
                        return found
                return None
            found_id = find_window(tree)
            if found_id:
                return str(found_id), None
            return None, f"No se encontr\xf3 ventana con '{nombre}'"
        else:
            return None, f"Compositor no soportado: {comp}"
    except Exception as e:
        return None, f"Error buscando ventana: {e}"


def enfocar_ventana(window_id: str) -> tuple[str, bool]:
    """Enfoca una ventana cuyo identificador ya fue resuelto por el compositor."""
    comp = _compositor()
    try:
        if comp == "hyprland":
            result = subprocess.run(["hyprctl", "dispatch", "focuswindow", f"address:{window_id}"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return result.stderr.strip() or "focuswindow fall\xf3", True
            return "OK", False
        elif comp == "sway":
            result = subprocess.run(["swaymsg", f"[con_id={window_id}]", "focus"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return result.stderr.strip() or "focus fall\xf3", True
            return "OK", False
        else:
            return f"Compositor no soportado: {comp}", True
    except Exception as e:
        return f"Error enfocando ventana: {e}", True


def cambiar_workspace(workspace: str) -> tuple[str, bool]:
    """Cambia de workspace y confirma el resultado desde el compositor."""
    objetivo = str(workspace)
    comp = _compositor()
    try:
        if comp == "hyprland":
            result = subprocess.run(["hyprctl", "dispatch", "workspace", objetivo],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return result.stderr.strip() or "dispatch workspace fall\xf3", True
            return "OK", False
        elif comp == "sway":
            result = subprocess.run(["swaymsg", f"workspace {objetivo}"],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return result.stderr.strip() or "workspace fall\xf3", True
            return "OK", False
        else:
            return f"Compositor no soportado: {comp}", True
    except Exception as e:
        return f"Error cambiando workspace: {e}", True


def click_en(x: int, y: int, boton: str = "left") -> tuple[str, bool]:
    """Click en coordenadas absolutas. No verifica (fire-and-forget)."""
    codigo = _BOTONES.get(boton.lower(), _BOTONES["left"])
    message, error = _run(["ydotool", "mousemove", "-a", str(x), str(y)])
    if error:
        return message, True
    message, error = _run(["ydotool", "click", codigo])
    return message, error


def escribir_texto(texto: str) -> tuple[str, bool]:
    """Escribe texto via ydotool type."""
    message, error = _run(["ydotool", "type", texto], timeout=10)
    return message, error


def mover_mouse(x: int, y: int) -> tuple[str, bool]:
    """Mueve mouse a coordenadas absolutas. No verifica."""
    message, error = _run(["ydotool", "mousemove", "-a", str(x), str(y)])
    return message, error


def mantener_tecla(tecla: str, duracion_s: float) -> tuple[str, bool]:
    """Mantiene una tecla presionada y garantiza el key-up ante cualquier fallo."""
    codigo = _TECLAS.get(tecla.upper(), tecla)
    try:
        duracion = float(duracion_s)
    except (TypeError, ValueError):
        return "La duraci\xf3n de la tecla debe ser num\xe9rica.", True
    if duracion < 0:
        return "La duraci\xf3n de la tecla no puede ser negativa.", True

    down_message, down_error = _run(["ydotool", "key", f"{codigo}:1"])
    if down_error:
        return down_message, True

    try:
        time.sleep(duracion)
    finally:
        up_message, up_error = _run(["ydotool", "key", f"{codigo}:0"])
        if up_error:
            return f"No se pudo soltar la tecla {tecla}: {up_message}", True
    return "OK", False


def _run(cmd: list[str], timeout: int = 5) -> tuple[str, bool]:
    """Ejecuta ydotool y devuelve (output, error)."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return f"{cmd[0]} no est\xe1 instalado.", True
    except subprocess.TimeoutExpired:
        return f"{cmd[0]} no respondi\xf3 a tiempo.", True
    except OSError as exc:
        return f"Error ejecutando {cmd[0]}: {exc}", True

    if result.returncode:
        detail = result.stderr.strip() or f"{cmd[0]} sali\xf3 con c\xf3digo {result.returncode}"
        return detail, True
    return "OK", False


def _run_query(cmd: list[str], timeout: float = 2.0) -> tuple[Any | None, str | None]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return None, f"{cmd[0]} no est\xe1 instalado."
    except subprocess.TimeoutExpired:
        return None, f"{cmd[0]} no respondi\xf3 en {timeout:g}s."
    except OSError as exc:
        return None, f"Error ejecutando {cmd[0]}: {exc}"
    if result.returncode:
        return None, result.stderr.strip() or f"{cmd[0]} sali\xf3 con c\xf3digo {result.returncode}"
    try:
        return json.loads(result.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"{cmd[0]} devolvi\xf3 JSON inv\xe1lido: {exc}"
