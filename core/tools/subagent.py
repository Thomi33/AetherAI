"""
subagent.py — Herramienta para lanzar sub-agentes de Aether.

Permite al agente principal delegar tareas específicas a instancias aisladas
del mismo modelo, cada una con su propio contexto y memoria temporal.
Esto acelera tareas complejas paralelas (ej. analizar varios archivos,
investigar múltiples temas, etc.).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class SubagentResult:
    """Resultado de la ejecución de un sub-agente."""
    task_id: str
    task: str
    success: bool
    response: str
    error: str | None = None


def _run_subagent(task: str, parent_mem: dict) -> SubagentResult:
    """
    Ejecuta un sub-agente usando AetherService (misma ruta que el agente principal).
    
    El sub-agente:
    - Usa AetherService que ya maneja correctamente la ejecución del grafo
    - Hereda solo lo esencial: resumen truncado y recuerdos importantes
    - Tiene su propia conversación vacía
    - No afecta la memoria principal hasta que se consolide
    """
    task_id = uuid.uuid4().hex[:8]
    try:
        # Memoria mínima: solo lo esencial para no saturar el contexto
        resumen = parent_mem.get("resumen", "") or ""
        if len(resumen) > 3000:
            resumen = resumen[:3000] + "... [truncado]"
        
        recuerdos = parent_mem.get("recuerdos", []) or []
        if not isinstance(recuerdos, list):
            recuerdos = []
        recuerdos_top = recuerdos[:5]
        
        # Construir el mensaje para el sub-agente incluyendo el contexto heredado
        contexto_heredado = ""
        if resumen:
            contexto_heredado += f"CONTEXTO HEREDADO (resumen):\n{resumen}\n\n"
        if recuerdos_top:
            contexto_heredado += "RECUERDOS RELEVANTES:\n"
            for r in recuerdos[:5]:
                contexto_heredado += f"- {r.get('contenido', '')}\n"
            contexto_heredado += "\n"
        
        mensaje_completo = f"{contexto_heredado}TAREA: {task}"
        
        # Usar AetherService que ya maneja correctamente la ejecución completa
        # (AetherService usa el grafo completo internamente)
        from backend.core.aether_service import AetherService
        resultado = AetherService.process_message(mensaje_completo)
        
        if isinstance(resultado, dict) and "response" in resultado:
            respuesta = resultado["response"]
        else:
            respuesta = str(resultado)
        
        return SubagentResult(task_id=task_id, task=task, success=True, response=respuesta)
    
    except Exception as e:  # noqa: BLE001
        return SubagentResult(task_id=task_id, task=task, success=False, response="", error=f"{type(e).__name__}: {e}")


def lanzar_subagentes(tareas: list[str], parent_mem: dict | None = None) -> list:
    """
    Lanza múltiples sub-agentes secuencialmente para procesar tareas independientes.
    
    Args:
        tareas: Lista de descripciones de tareas para cada sub-agente
        parent_mem: Memoria del agente padre (se usa la memoria global si None)
    
    Returns:
        Lista de SubagentResult, uno por tarea
    """
    if parent_mem is None:
        from core.memory.memory_manager import cargar_memoria
        parent_mem = cargar_memoria()
    
    resultados = []
    for tarea in tareas:
        resultado = _run_subagent(tarea, parent_mem)
        resultados.append(resultado)
    
    return resultados


def lanzar_subagente(task: str, parent_mem: dict | None = None) -> object:
    """Lanza un solo sub-agente (wrapper simple)."""
    if parent_mem is None:
        from core.memory.memory_manager import cargar_memoria
        parent_mem = cargar_memoria()
    return _run_subagent(task, parent_mem)
