"""Memoria central compartida de Aether (JSON, fuera de los runtimes).

Uso típico desde cualquier runtime:
    from core.memory.central import get_central_memory
    mem = get_central_memory()
    mem.learn("El usuario prefiere respuestas breves", importance=7,
              tags=["preferencias"], runtime="terminal")
    resultados = mem.search("preferencias del usuario", limit=5)
"""
from core.memory.central.store import CentralMemory, default_root, get_central_memory

__all__ = ["CentralMemory", "default_root", "get_central_memory"]
