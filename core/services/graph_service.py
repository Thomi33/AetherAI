"""
graph_service: entrada única al motor LangGraph de Aether.

Este módulo es deliberadamente delgado y SIN dependencias de CrewAI, de modo
que el CLI/servicio puedan usar el grafo como motor único sin arrastrar el
código legado de core/services/aether_service.py (basado en CrewAI).

Función principal:
    procesar_orden_grafo(orden, mem, modo_autonomo) -> str
"""

from core.memory.memory_manager import registrar_turno, programar_consolidacion


def respuesta_final_o_informe(resultado: dict) -> str:
    """
    Respuesta final del grafo con reemplazo honesto del texto genérico.

    Si el modelo dejó texto final, se devuelve tal cual. Si cerró sin texto
    (respuesta vacía) pero el agent loop ejecutó pasos de herramientas, en
    vez de "Operación completada." se informa cuántos pasos fallaron
    (clasificación 'error'/'sin_resultados' de agent_pasos_log).
    """
    final = ((resultado or {}).get("final_response") or "").strip()
    if final:
        return final

    pasos = (resultado or {}).get("agent_pasos_log") or []
    fallidos = sum(
        1 for p in pasos
        if isinstance(p, dict) and p.get("cls") in ("error", "sin_resultados")
    )
    if fallidos:
        return (f"Operación incompleta: {fallidos} de {len(pasos)} paso(s) de "
                f"herramientas fallaron. Podés pedirme reintentar los que fallaron.")
    return "Operación completada."


def procesar_orden_grafo(orden: str, mem: dict, modo_autonomo: bool = True,
                         imagenes: list[str] | None = None) -> str:
    """
    Procesa una orden invocando el grafo de Aether
    (planner → plan_executor → [error handler] → synthesizer → finalize).

    - Normaliza la memoria y construye el estado con la factory central.
    - Registra el turno del usuario aquí; node_finalize registra el de Aether
      (evita doble registro en la tabla conversaciones).
    - imagenes: thumbnails b64 de adjuntos de imagen → el modelo principal
      los ve incrustados en su mensaje (visión inline).
    - Retorna la respuesta final del grafo.
    """
    # Imports locales para evitar import circular y costos de import en frío.
    from core.agent.graph_builder import get_graph
    from core.agent.graph_state import crear_estado_inicial

    registrar_turno(mem, "usuario", orden)

    grafo = get_graph()
    estado = crear_estado_inicial(orden, mem, modo_autonomo, imagenes=imagenes)
    resultado = grafo.invoke(estado)
    # node_finalize ya dejó registrado el turno de Aether. Programar después
    # de invoke garantiza que el lote vea el intercambio completo y no añade
    # latencia al caller (API/CLI).
    programar_consolidacion(mem)
    return respuesta_final_o_informe(resultado)
