# AetherAI — Arquitectura actual

> Documento técnico basado en el estado de `main` inspeccionado el 2026-09-09.
> Describe la arquitectura implementada en el repositorio, no una arquitectura futura.

## 1. Resumen

AetherAI es un agente local, terminal-first, orientado a Arch Linux. El motor activo utiliza **LangGraph** para orquestación y **Ollama** como runtime local de modelos. El repositorio conserva componentes históricos y una superficie `backend/`, pero el flujo soportado documentado por el proyecto es el CLI.

La arquitectura actual combina dos estrategias de ejecución:

1. **Fast-path / Tool Planning:** `node_planner` puede construir planes deterministas, especialmente para operaciones simples.
2. **Agent Loop:** para tareas abiertas, el modelo recibe el catálogo estructurado de tools y realiza tool calling incremental; el grafo vuelve al loop mientras `agent_activo` siga activo.

El punto de entrada lógico del motor es `procesar_orden_grafo()`.

## 2. Flujo de alto nivel

```text
Usuario / CLI
    │
    ▼
procesar_orden_grafo()
    │
    ├── registrar turno del usuario
    ├── crear_estado_inicial()
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    LangGraph                             │
│                                                         │
│ START                                                   │
│   │                                                     │
│   ▼                                                     │
│ planner                                                 │
│   │                                                     │
│   ▼                                                     │
│ context_manager                                         │
│   │                                                     │
│   ├──────────────► plan_executor ──► synthesizer ──┐   │
│   │                                                │   │
│   └──────────────► agent_loop ◄───────┐           │   │
│                       │                │           │   │
│                       └────────────────┘           │   │
│                                                    ▼   │
│                                                 finalize│
│                                                    │   │
│                                                    ▼   │
│                                                   END   │
│                                                         │
│  Errores: executor/agent → diagnose → confirm → retry │
│                         ↘ fallback                     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Respuesta final + consolidación de memoria
```

`core/agent/graph_builder.py` define explícitamente estos nodos y sus rutas condicionales. El grafo se compila una sola vez y se reutiliza mediante `get_graph()`; `reset_graph()` permite invalidar esa instancia en procesos que lo necesiten.

## 3. Entrada única al motor

`core/agent/graph_service.py` y la fachada compatible `core/services/graph_service.py` exponen `procesar_orden_grafo(orden, mem, modo_autonomo=True)`.

La función:

1. registra el turno del usuario;
2. obtiene el grafo compilado;
3. crea un `AetherState` completo mediante `crear_estado_inicial()`;
4. ejecuta `grafo.invoke()`;
5. devuelve `final_response`;
6. programa la consolidación de memoria después de completar el intercambio.

La existencia de ambas rutas de `graph_service` es deliberada: `core/agent/graph_service.py` es la implementación del motor y `core/services/graph_service.py` actúa como fachada compatible para callers existentes.

## 4. Estado: `AetherState`

`core/agent/graph_state.py` define el contrato central de estado mediante `TypedDict`.

### Entrada y control

- `orden`: solicitud original del usuario.
- `mem`: memoria normalizada.
- `modo_autonomo`: modo de ejecución.
- `intent`, `done`, `terminado`, `tokens`, `ruta`: control y compatibilidad.

### Tool Planning

- `plan_activo`
- `plan_pasos`
- `plan_index`
- `plan_resultados`

Estos campos sostienen el camino de planificación/fast-path.

### Agent Loop

- `agent_activo`
- `agent_messages`
- `agent_pasos_log`

El transcript sigue el formato de mensajes que utiliza Ollama (`system/user/assistant/tool`) y el log conserva tool, argumentos y resultado para diagnóstico.

### Resultados de tools

El estado reserva campos separados para web, shell, MCP, visión, filesystem y computer use. También conserva `_tool_args` para los argumentos estructurados de la llamada actual.

### Context Manager

- `sesion_id`
- `context_slots`
- `context_dump`
- `tema_actual`
- `historial_filtrado`

### Error Handler

El estado mantiene el contexto del error, número máximo de intentos, autorización y el fix propuesto/diff/fuente, además de aliases legacy para compatibilidad.

La factory `crear_estado_inicial()` es la fuente central para inicializar el estado y normaliza la memoria antes de construirlo.

## 5. Orquestación LangGraph

`core/agent/graph_builder.py` construye el `StateGraph`.

### Ruta principal

```text
START
  → planner
  → context_manager
  → plan_executor | agent_loop
  → plan_synthesizer (cuando corresponde)
  → finalize
  → END
```

### `planner`

El planner conserva fast-paths deterministas y decide si la orden necesita el camino de planificación o el agent loop. El estado producido por esta fase es utilizado por `context_manager` para identificar el tema inicial.

### `context_manager`

Se ejecuta siempre después del planner y antes de la inferencia. Su responsabilidad es seleccionar el contexto relevante, actualizar el tema activo y producir `context_dump` para diagnóstico.

La implementación usa `construir_contexto_memoria()` y `construir_context_dump()` y propaga el resultado mediante `context_slots`.

### `plan_executor`

Ejecuta los pasos de un plan mediante los nodos reales registrados en `TOOL_REGISTRY`. Después de cada ejecución, el router decide si debe continuar el plan, sintetizar el resultado, finalizar o entrar al manejador de errores.

### `agent_loop`

Es el camino de tool calling nativo. El modelo recibe las tools estructuradas y decide incrementalmente qué hacer. El nodo puede volver a sí mismo mientras `agent_activo=True`; cuando el objetivo termina, la ruta continúa hacia `finalize`.

Este diseño evita obligar a las tareas abiertas a producir un plan completo de antemano.

### `plan_synthesizer`

Convierte resultados crudos de herramientas en una respuesta útil cuando todavía es necesario pasar por una etapa de síntesis. El builder evita esta etapa cuando una tool ya produjo una respuesta determinista y no existen datos crudos nuevos que sintetizar.

### `finalize`

Es el cierre del grafo y deja la respuesta final disponible en `final_response`. También participa en el registro del turno de Aether.

## 6. Registro de herramientas

`core/agent/tool_registry.py` es la fuente central de verdad para las tools conocidas por el agente.

Cada entrada vincula:

```text
nombre de tool
    ├── node
    ├── instruccion_requerida
    └── descripcion
```

Además, `TOOL_PARAMETROS` define los schemas de argumentos utilizados por el tool calling nativo.

Actualmente el registro contiene:

| Tool | Función conceptual |
|---|---|
| `text` | conversación/respuesta directa |
| `web` | búsqueda web |
| `shell` | generación y ejecución de comandos |
| `launch` | lanzamiento de aplicaciones |
| `vision` | captura y análisis de pantalla |
| `codigo` | generación/ejecución de código |
| `memory` | gestión de memoria |
| `file_write` | escritura legacy basada en resultado/instrucción |
| `extract` | extracción/limpieza de resultados anteriores |
| `mcp` | invocación de tools MCP |
| `computer_use` | percepción + acción sobre la interfaz |
| `fs_write` | escritura estructurada de uno o varios archivos |
| `fs_read` | lectura de archivo |
| `fs_mkdir` | creación de directorios |
| `fs_list` | listado de directorios |

`validar_tool_call()` realiza una validación rápida de existencia, tipo de argumentos, instrucciones requeridas y, en MCP, `server` + `name`. No sustituye la validación completa de un JSON Schema.

## 7. Herramientas y capas de ejecución

La implementación concreta de las herramientas se encuentra principalmente en `core/tools/`:

- `web_search.py`: búsqueda web.
- `url_reader.py`: lectura de URLs.
- `shell_executor.py`: ejecución de shell.
- `flatpak_manager.py`: lanzamiento de aplicaciones.
- `vision.py`: captura/análisis visual.
- `computer_control.py`: control de interfaz.
- `mcp_client.py`: cliente MCP.
- `file_writer.py`: writer legacy.
- `filesystem_tool.py`: operaciones estructuradas de filesystem.

Los nodos del grafo funcionan como capa de orquestación; las implementaciones de `core/tools/` encapsulan las operaciones concretas.

## 8. Directorio de trabajo y separación de datos

Aether distingue entre:

- **Proyecto:** código fuente y comportamiento versionado.
- **Home de runtime:** `~/Aether`, utilizado para DB, logs, screenshots y otros datos persistentes.
- **Directorio de trabajo:** la ruta sobre la que el usuario abrió Aether o la especificada mediante `--workdir`.

`bin/aether` conserva el `$PWD` original en `AETHER_CWD` antes de entrar al directorio del proyecto. `core/config/settings.py` resuelve `RUTA_TRABAJO` dinámicamente a partir de esa variable o del cwd actual.

Esto permite ejecutar Aether desde un proyecto externo sin mover la instalación del agente.

Los directorios de trabajo nuevos pasan por autorización explícita mediante `core/config/dir_authorization.py`, mientras que rutas sensibles del sistema tienen controles adicionales.

## 9. Configuración y runtime local

`core/config/settings.py` centraliza los valores principales:

- `OLLAMA_HOST`: `http://localhost:11434`.
- `SEARXNG_URL`: `http://localhost:8081`.
- `MODELO`: modelo de texto configurado actualmente.
- `MODELO_VISION`: modelo multimodal configurado actualmente.
- `TOOL_CALLING_NATIVO=True`.
- `NUM_CTX=8192`.
- `OLLAMA_KEEP_ALIVE=-1`.
- opciones de generación y límites de contexto.
- rutas de DB, logs, screenshots, embeddings, backups y skills.

La configuración de modelo de `model_policy.py` actualmente funciona en **modo observación/passthrough**: registra decisiones y métricas, pero devuelve el modelo solicitado por el caller y no realiza switching efectivo. Esto deja preparado el punto de decisión para políticas adaptativas futuras.

## 10. Memoria

El subsistema de memoria vive en `core/memory/`.

### Capas

```text
Memoria RAM
    │
    ├── core
    ├── resumen
    ├── conversación
    └── historial de comandos
          │
          ▼
     current.db
          │
          ├── conversaciones
          ├── comandos
          ├── recuerdos
          ├── core_memory
          └── resumen_memoria
```

`memory_manager.py` utiliza `~/Aether/db/current.db` como DB de producción. La estructura RAM se normaliza mediante `normalizar_mem()`.

El contexto conversacional no se inyecta sin límite: `settings.py` define presupuestos de turnos y caracteres. Para planes multi-tool, el historial conversacional puede recortarse a cero turnos para evitar contaminación entre pasos; los resultados de pasos anteriores constituyen el contexto operativo relevante.

`consolidator.py` programa la actualización del resumen acumulativo después de completar una orden.

El repositorio también contiene un subsistema `core/memory/store/` destinado a controlar esquema, migraciones, snapshots, staging y escritura persistente.

## 11. MCP y extensibilidad

MCP está integrado como una tool del registro (`mcp`) y se ejecuta a través de `core/tools/mcp_client.py`. El contrato de tool calling requiere identificar `server`, `name` y, cuando corresponda, `arguments`.

Esto permite que servidores MCP conectados extiendan las capacidades de Aether sin convertir cada integración externa en un nodo hardcodeado independiente.

## 12. Error handling

Los errores de ejecución tienen una ruta explícita:

```text
error
  │
  ▼
error_diagnose
  │
  ├── fix propuesto → error_confirm → error_retry ──► ejecución
  │
  └── sin fix / límite → error_fallback
```

El estado distingue el contexto (`shell`, `launch`, `codigo`), mantiene el intento actual y limita los reintentos. `graph_builder.py` usa un máximo general de 3 intentos y permite límites específicos por contexto definidos en `graph_state.py`.

## 13. Modelos

La configuración actual separa modelo de texto y modelo de visión. El registro de política de modelos permite clasificar tareas como `chat`, `planner`, `synthesis`, `shell`, `code`, `vision`, `mcp` y `error`.

La política actualmente no cambia el modelo activo: su función es observacional y registra recomendaciones, posible cambio y latencia para futuras estrategias.

## 14. CLI y launcher

Los entrypoints relevantes del repositorio son:

```text
bin/aether
    └── bin/aether_run.py
          └── runtime CLI

run.py
    └── CLI/TUI
          └── core.services.graph_service

jarvis_new.py
    └── entrada alternativa al motor
```

El launcher `bin/aether` soporta ejecución desde cualquier directorio y comandos como `task`, `doctor`, `cli`, `--version` y `--help`.

La carpeta `cli/` contiene un loop interactivo marcado como deprecated en el README; el launcher moderno es la superficie recomendada.

## 15. Componentes históricos / no soportados

El repositorio contiene componentes que no deben confundirse con el motor actual:

- `core/services/aether_service.py`: implementación histórica basada en CrewAI.
- `backend/`: API FastAPI y piezas web existentes, pero el README actual declara que no forman parte del flujo CLI soportado.
- `frontend.tar.gz`: frontend empaquetado asociado a esa superficie web.
- aliases/campos legacy dentro del estado y memoria: se mantienen para compatibilidad.

La arquitectura activa es **LangGraph + Ollama + tools/MCP + memoria local**, no CrewAI.

## 16. Estructura lógica

```text
AetherAI/
├── bin/                         # launcher global y runtime CLI
├── core/
│   ├── agent/                   # grafo, estado, planner, loop, registry
│   ├── config/                  # configuración y autorización de directorios
│   ├── connectors/              # conectores del sistema
│   ├── memory/                  # memoria + store persistente
│   ├── parser/                  # parsing de shell/respuestas
│   ├── services/                # fachadas/servicios compatibles
│   ├── skills/                  # comportamiento reutilizable
│   ├── tools/                   # operaciones concretas
│   ├── utils/                   # utilidades
│   ├── events.py                # eventos/runtime
│   └── state_manager.py         # gestión de estado auxiliar
├── cli/                         # loop legacy/deprecated
├── backend/                     # superficie web fuera del flujo CLI soportado
├── tests/                       # suite de pruebas
└── documentación/               # README, changelog, planes e informes
```

## 17. Principios arquitectónicos observables

1. **Local-first:** el runtime de inferencia principal está en Ollama local.
2. **Orquestación explícita:** LangGraph controla estados y transiciones.
3. **Tool registry único:** las tools se describen y validan desde un registro central.
4. **Tool calling estructurado:** el agent loop usa schemas de argumentos, no depende exclusivamente de texto libre.
5. **Contexto antes de inferencia:** `context_manager` selecciona información antes de los nodos downstream.
6. **Filesystem contextual:** el directorio de trabajo puede ser externo a la carpeta del proyecto.
7. **Memoria persistente separada:** los datos de runtime viven fuera del código versionado.
8. **Compatibilidad gradual:** existen aliases y fachadas legacy mientras se consolida la arquitectura nueva.
9. **Recuperación ante errores:** las ejecuciones pueden diagnosticar, proponer correcciones, reintentar y hacer fallback.
10. **Observabilidad preparada:** el sistema contiene `context_dump`, logs de pasos del agent loop y una política de modelos en modo observación.

## 18. Fuentes de implementación

Este documento se contrastó directamente con los componentes principales de `main`, especialmente:

- `core/agent/graph_builder.py`
- `core/agent/graph_state.py`
- `core/agent/tool_registry.py`
- `core/agent/node_context_manager.py`
- `core/agent/model_policy.py`
- `core/agent/graph_service.py`
- `core/services/graph_service.py`
- `core/config/settings.py`
- `core/memory/memory_manager.py`
- `core/tools/*`
- `README.md`
