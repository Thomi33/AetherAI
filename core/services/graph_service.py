"""
graph_service: entrada única al motor LangGraph de Aether.

Este módulo es deliberadamente delgado y SIN dependencias de CrewAI, de modo
que el CLI/servicio puedan usar el grafo como motor único sin arrastrar el
código legado de core/services/aether_service.py (basado en CrewAI).

Función principal:
    procesar_orden_grafo(orden, mem, modo_autonomo) -> str
"""

from datetime import datetime

from core.memory.memory_manager import registrar_turno, programar_consolidacion


def _detalle_pasos_fallidos(pasos: list) -> str:
    """
    Detalle legible de los pasos fallidos de agent_pasos_log: qué paso, qué
    tool, qué comando/query pedía y el error (primera línea).

    Fix del mensaje genérico "Operación incompleta: 1 de 6 paso(s) de
    herramientas fallaron": no decía QUÉ falló, así que el usuario (y el
    modelo en el turno siguiente) no podía saber si la conclusión estaba
    afectada.
    """
    lineas: list[str] = []
    for i, p in enumerate(pasos):
        if not isinstance(p, dict) or p.get("cls") not in ("error", "sin_resultados"):
            continue
        tool = str(p.get("tool") or "?")
        args = p.get("args") if isinstance(p.get("args"), dict) else {}
        comando = ""
        for clave in ("instruccion", "command", "query", "url", "path"):
            val = args.get(clave)
            if isinstance(val, str) and val.strip():
                comando = val.strip()
                break
        resultado = str(p.get("resultado") or "").strip()
        primera = resultado.splitlines()[0][:150] if resultado else "(sin salida)"
        linea = f"- Paso {i + 1} (tool: {tool}"
        if comando:
            linea += f", pedido: «{comando[:80]}»"
        linea += f"): {primera}"
        lineas.append(linea)
    return "\n".join(lineas)


def respuesta_final_o_informe(resultado: dict) -> str:
    """
    Respuesta final del grafo con reemplazo honesto del texto genérico.

    Si el modelo dejó texto final, se devuelve tal cual. Si cerró sin texto
    (respuesta vacía) pero el agent loop ejecutó pasos de herramientas, en
    vez de "Operación completada." se informa qué pasos fallaron
    (clasificación 'error'/'sin_resultados' de agent_pasos_log), CON el
    detalle de cada uno y el impacto en la conclusión.
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
        detalle = _detalle_pasos_fallidos(pasos)
        return (
            f"Operación incompleta: {fallidos} de {len(pasos)} paso(s) de "
            f"herramientas fallaron:\n{detalle}\n\n"
            f"La conclusión puede estar afectada por {fallidos} paso(s) "
            f"fallido(s) — no todos los datos pudieron obtenerse. Podés "
            f"pedirme reintentar solo los que fallaron."
        )
    return "Operación completada."


def procesar_orden_grafo(orden: str, mem: dict, modo_autonomo: bool = True,
                         imagenes: list[str] | None = None) -> str:
    """
    Procesa una orden invocando el grafo de Aether
    (planner → plan_executor → [error handler] → synthesizer → finalize).

    - Normaliza la memoria y construye el estado con la factory central.
    - Turnos del historial: los registra node_finalize al completarse la
      inferencia (el turno del usuario con su fecha REAL de envío vía
      AetherState.fecha_envio; el de Aether con la hora de finalización).
      Acá queda SOLO el fallback para los caminos que terminan sin pasar
      por finalize (mismo contrato que tui/engine_bridge.py y
      backend/core/aether_service.py). Antes el turno del usuario se
      pre-registraba acá: además de duplicarse en el store, terminaba en
      el contexto de conversación y el modelo ecoeaba su propio input.
    - imagenes: thumbnails b64 de adjuntos de imagen → el modelo principal
      los ve incrustados en su mensaje (visión inline).
    - Retorna la respuesta final del grafo.
    """
    # Imports locales para evitar import circular y costos de import en frío.
    from core.agent.graph_builder import get_graph
    from core.agent.graph_state import crear_estado_inicial

    # FIX reloj (Plan B): hora REAL de llegada de la orden — viaja por el
    # estado hasta node_finalize, que estampa el turno del usuario con
    # esta fecha y no con la del fin de la inferencia.
    fecha_envio = datetime.now().isoformat()

    grafo = get_graph()
    estado = crear_estado_inicial(orden, mem, modo_autonomo, imagenes=imagenes,
                                  fecha_envio=fecha_envio)

    # Mismo patrón de consumo que los bridges: deltas del stream acumulados
    # en el resultado + detección de finalize para el fallback del turno
    # del usuario.
    resultado: dict = {}
    llego_finalize = False
    for update in grafo.stream(estado, stream_mode="updates"):
        for nodo, delta in update.items():
            if isinstance(delta, dict):
                resultado.update(delta)
            if nodo == "finalize":
                llego_finalize = True

    if not llego_finalize:
        registrar_turno(mem, "usuario", orden, fecha=fecha_envio)

    # node_finalize ya dejó registrado el intercambio completo. Programar
    # después del stream garantiza que el lote de consolidación vea ambos
    # turnos y no añade latencia al caller (API/CLI).
    programar_consolidacion(mem)
    return respuesta_final_o_informe(resultado)
