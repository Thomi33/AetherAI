# 🗺️ ai_map.md — Mapa de Contexto Semántico · Aether (AetherAI)

> **Propósito:** mapa denso de contexto del repositorio, optimizado para agentes de IA. Leélo ANTES de
> generar o modificar código. Refleja el estado real verificado del árbol (rama `development/aether-next`),
> contrastado contra el código y no solo contra la documentación.
> **Protocolo de lectura rápida:** (1) §0 + §1 para ubicarse; (2) para tocar código ir directo a §9
> (mapa de lectura por tarea); (3) respetar §5 (reglas de arquitectura) sin excepciones.
> Detalle profundo: `docs/ARCHITECTURE.md` (EN) · `docs/es/ARCHITECTURE.es.md` (ES) · `README.md` (uso humano).

---

## 0. HECHOS CLAVE (leer primero)

| Campo | Valor |
|---|---|
| Producto | **Aether**: agente local, terminal-first, en español. Conversa, ejecuta shell, lee/escribe archivos, busca en la web, observa la pantalla y controla mouse/teclado. |
| Motor activo | **LangGraph + Ollama**. ⚠️ NO es CrewAI (es histórico); `crewai-env/` es solo un venv con nombre viejo. |
| Repo / rama | `github.com/Thomi33/AetherAI` · rama activa `development/aether-next` · ≈120 archivos de proyecto (sin venvs). |
| Lenguaje | Python **3.12** (`.python-version`). Sin `pyproject`/packaging: los entry points insertan la raíz en `sys.path` y los imports son absolutos (`core.*`, `backend.api.*`). |
| Idioma del código | **Español** en docstrings, mensajes e identificadores públicos (`procesar_orden_grafo`, `cargar_memoria`). El código nuevo DEBE seguirlo. |
| Plataforma objetivo | Arch Linux + Wayland (Hyprland/Sway). Binarios del sistema: `ollama`, `zsh`, `grim`, `ydotool`, `hyprctl`/`swaymsg`, whisper local (STT). |
| Datos de runtime | Viven **FUERA del repo** en `BASE_AETHER` = `~/Aether` (override `AETHER_DATA_DIR`): memoria, notas, logs, screenshots y skills operativas. |
| venvs (gitignored) | `env/` (TUI, prioridad 1) · `crewai-env/` · `.venv/` (backend). El lanzador elige en ese orden. |
| Backend | `backend/` = API FastAPI **experimental**, fuera del flujo soportado (el soportado es CLI/TUI). |
| Tests | pytest hermético: `conftest.py` (raíz) agrega el repo a `sys.path` y fuerza `AETHER_CENTRAL_MEMORY=0`. |

---

## 1. ÁRBOL DE DIRECTORIOS EXPLICADO

```text
mi_proyecto_crew/            ← raíz del repo (nombre local histórico; el repo real es AetherAI)
│
│ ── ENTRY POINTS ─────────────────────────────────────────────────────────────────
├── run.py               ★ Entry point principal (TUI). Parsea --workdir → fija AETHER_CWD
│                          ANTES de importar settings → asegura esquema de memoria → tui.app.run().
├── bin/aether           ★ Lanzador bash global (instalado en ~/.local/bin). Subcomandos:
│                          tui | task "orden" | doctor | version | web. Auto-localiza el
│                          proyecto (soporta symlinks), captura $PWD → AETHER_CWD y elige venv.
├── bin/aether_run.py    ★ CLI sin TUI: {version | task | doctor}. task = one-shot al grafo.
├── tui/__main__.py        `python -m tui` → tui.app.run() (alias de run.py).
│
│ ── MOTOR ────────────────────────────────────────────────────────────────────────
├── core/                ★ PAQUETE MÁS IMPORTANTE (motor del agente). Imports `core.*`.
│   ├── agent/             Núcleo de razonamiento:
│   │   · graph_builder.py      build_graph()/get_graph()/reset_graph(): nodos y rutas
│   │                            condicionales del grafo LangGraph (compila 1 vez y cachea).
│   │   · graph_nodes.py        TODOS los nodos: node_planner, node_context_manager,
│   │                            node_plan_executor, node_agent_loop, node_plan_synthesizer,
│   │                            node_<tool>*, node_error_*, node_finalize (≈5200 líneas).
│   │   · graph_state.py        AetherState (TypedDict): estado que fluye por el grafo.
│   │   · tool_registry.py      REGISTRO ÚNICO de tools: TOOL_REGISTRY (nombre→nodo+desc),
│   │                            TOOL_PARAMETROS (JSON schemas), TOOLS_VALIDAS.
│   │   · node_context_manager.py  Poda/compacta contexto ANTES de inferir (protege VRAM).
│   │   · prompts.py / streaming.py / grounding.py / keywords_config.json.
│   ├── services/          ★ graph_service.py → procesar_orden_grafo(): PUNTO DE ENTRADA
│   │                        LÓGICO ÚNICO al motor. + stt_service/stt_worker/audio_input
│   │                        (dictado por voz, whisper local).
│   ├── config/            settings.py (constantes; delega en settings_v2) · settings_v2.py
│   │                        (Settings lazy) · config_manager.py (2 capas + hot reload) ·
│   │                        dir_authorization.py (autorización de dirs de trabajo) ·
│   │                        config.json / config.local.json · mcp_servers.json /
│   │                        mcp_servers.local.json (mismo patrón de 2 capas).
│   ├── memory/            memory_manager.py (store JSON de sesión: turnos, recuerdos,
│   │                        resumen acumulativo) · central_store_v2.py (memoria central
│   │                        SQLite compartida por TODOS los runtimes) · context_builder.py
│   │                        (inyecta memoria al prompt) · memoria_store.py · central/ (v1).
│   ├── tools/             Implementaciones: shell_executor · filesystem_tool (fs_read/
│   │                        write/mkdir/list) · file_writer · web_search (SearXNG→DDG) ·
│   │                        url_reader · vision (grim+multimodal) · mcp_client ·
│   │                        ydotool_wrapper (computer_use) · roblox_bridge ·
│   │                        flatpak_manager (launch) · net_audit · subagent.
│   ├── connectors/        Fuentes externas: base, filesystem, github, notion, registry.
│   ├── parser/            shell_parser.py (parsing de salida del shell).
│   ├── skills/            registry.py — skills en <BASE_AETHER>/skills/<n>/SKILL.md.
│   └── utils/             intent_utils.py · response_cleaner.py (compartida TUI/backend).
│
│ ── INTERFAZ ──────────────────────────────────────────────────────────────────────
├── tui/                   TUI Textual: app.py (aplicación principal) · engine_bridge.py
│                          (grafo→TUI: iter_eventos, inicializar_motor, Token/Done/Error) ·
│                          widgets/ (chat, plan, debug, status, paste_input, selector_screens,
│                          slash_completer) · status_messages.py. Comandos /slash + F2 dictado.
│
│ ── WEB EXPERIMENTAL ──────────────────────────────────────────────────────────────
├── backend/               API FastAPI aislada: api/main.py:app (lifespan → AetherService
│                          on-demand) · api/routes/ (14 routers: chat SSE, ws, config, prompt,
│                          models, session, memory, skills, mcp, runtime, roblox, workspace,
│                          account, stt) · api/models/schemas.py (Pydantic) ·
│                          core/aether_service.py (envuelve el grafo) · core/attachments.py
│                          (adjuntos base64) · core/config.py (pydantic-settings) · run.sh
│                          (uvicorn desde RAÍZ) · requirements.txt propios.
│
│ ── SOPORTE ───────────────────────────────────────────────────────────────────────
├── tests/                 Suite pytest (regresión + QA). Convención test_<feature>.py.
├── scripts/               aether_focus.py · qa_verify_18331.py · test_tui_prompt.py.
├── tools/                 hardware_probe.py (diagnóstico de hardware p/el instalador).
├── installer/             install-core.sh = instalador real (perfiles POTATO→ULTRA, elige
│                          modelo, crea venv, instala Ollama y el comando `aether` global).
├── install.sh             Bootstrap público: descarga install-core.sh de GitHub y lo ejecuta.
├── docs/                  ARCHITECTURE.md (EN) · docs/es/ (originales ES).
├── _legacy/               Artefactos retirados (DB vieja, dumps, .bak). NO importar ni usar.
├── cli/                   Vacío (solo __init__.py). Reservado.
│
│ ── CONFIG RAÍZ ───────────────────────────────────────────────────────────────────
├── conftest.py            Pytest global: sys.path raíz + AETHER_CENTRAL_MEMORY=0.
├── requirements.txt       Deps motor: ollama, langgraph, langchain-core, fastapi, uvicorn,
│                          pydantic(-settings), requests, mcp, textual, rich, opencv-headless.
├── tps.py                 Benchmark tok/s contra la API de Ollama.
├── .python-version        3.12.7
├── .gitignore             Ignora venvs, .env, *.db, memoria, *.local.json, backups.
└── env/ · .venv/ · crewai-env/   venvs gitignored (NO analizar su contenido).
```

## 2. TECNOLOGÍAS CLAVE

| Tecnología | Pin (`requirements.txt`) | Rol | Archivos clave |
|---|---|---|---|
| Python | 3.12 (`.python-version`) | Runtime de todo el sistema | — |
| **LangGraph** + langchain-core | ≥1.0 | Orquestación explícita del agente (grafo de estados) | `core/agent/graph_builder.py`, `graph_state.py` |
| **Ollama** (HTTP local :11434) | cliente ≥0.6 | Runtime del LLM local: chat, streaming, **tool calling nativo**, visión multimodal | `core/agent/graph_nodes.py`, `streaming.py`, `tps.py` |
| **Textual** + Rich | ≥6.0 / ≥14.0 | TUI completa (widgets, comandos `/slash`, temas) | `tui/` |
| FastAPI + Uvicorn + Pydantic v2 (+ pydantic-settings) | ≥0.115 / ≥0.30 / ≥2 | API web experimental (REST + SSE + WebSocket) y validación | `backend/` |
| **MCP** (Model Context Protocol) | ≥1.0 | Cliente de servidores MCP externos (tools extra) | `core/tools/mcp_client.py`, `core/config/mcp_servers*.json` |
| SQLite (stdlib) + JSON | — | Memoria central compartida (v2) + store de sesión JSON | `core/memory/central_store_v2.py`, `memoria_store.py` |
| OpenCV headless (+ Pillow en backend) | ≥4.10 / ≥10 | Visión por pantalla y thumbnails de adjuntos | `core/tools/vision.py`, `backend/core/attachments.py` |
| requests | ≥2.31 | Web (SearXNG → fallback DuckDuckGo) y llamadas crudas a Ollama | `core/tools/web_search.py`, `url_reader.py` |
| pytest | dev | Suite de tests hermética | `tests/`, `conftest.py` |
| Binarios del sistema | externos | `zsh` (shell) · `grim` (screenshots) · `ydotool` (computer_use) · `hyprctl`/`swaymsg` (contexto WM) · whisper (STT) | `core/tools/` |

Notas técnicas:
- **Tool calling nativo** de Ollama (`TOOL_CALLING_NATIVO`): el agent loop envía `TOOL_PARAMETROS` como JSON schemas; no parsea texto libre.
- La **visión usa el modelo principal** si es multimodal (no hay modelo de visión separado; Google Vision solo es opt-in en `/play-roblox google`).
- No hay frontend Node en este repo: la Web UI vive en otro repo (`~/aether_web_ui`); `frontend/` solo aparece en `.gitignore` como defensa.

## 3. ENTRY POINTS DEL CÓDIGO

| Entry point | Invocación | Qué hace / a dónde lleva |
|---|---|---|
| `run.py` | `python run.py [--workdir R]` · `aether` | TUI principal. Fija `AETHER_CWD` → `asegurar_esquema()` (memoria) → `tui.app.run()`. |
| `tui/__main__.py` | `python -m tui` | Igual que `run.py` sin `--workdir`. |
| `bin/aether` | `aether [tui\|task\|doctor\|version\|web]` | Lanzador bash global. Exporta `AETHER_CWD=$PWD`, elige venv (`env→crewai-env→.venv`), `cd` al repo y despacha. |
| `bin/aether_run.py` | `python bin/aether_run.py {version\|task\|doctor}` | `task` = one-shot: autoriza dir → `cargar_memoria()` → `procesar_orden_grafo()`. `doctor` = diagnóstico (python/venv/ollama/config/db). |
| **`core/services/graph_service.py::procesar_orden_grafo(orden, mem, modo_autonomo=True)`** | API interna | ★ **PUNTO DE ENTRADA LÓGICO ÚNICO al motor.** Registra turno → `get_graph()` → `crear_estado_inicial()` → `grafo.invoke()` → devuelve `final_response` → agenda consolidación de memoria. TUI, CLI y backend llaman SOLO esto. |
| `core/agent/graph_builder.py` | `build_graph()` · `get_graph()` · `reset_graph()` | Construcción/compilación/cache del grafo. Smoke test: `python -c "from core.agent.graph_builder import build_graph; build_graph(); print('ok')"`. |
| `backend/api/main.py::app` | `.venv/bin/python -m uvicorn backend.api.main:app` **desde la RAÍZ** | API FastAPI (SSE `/api/chat/stream`, WS `/ws/chat`, REST). `AetherService.initialize()` on-demand en lifespan. |
| `tests/` | `python -m pytest tests` | Suite completa (con `conftest.py` raíz). |
| `install.sh` → `installer/install-core.sh` | `./install.sh` | Instalación + perfilado de hardware (POTATO→ULTRA) + descarga de modelo. |
| `tps.py` | `python tps.py [modelo] [prompt]` | Benchmark de tok/s contra Ollama. |

## 4. FLUJO DEL MOTOR (grafo LangGraph)

```text
START → planner → context_manager → ¿hay plan? ──sí──→ plan_executor → plan_synthesizer ─┐
                                        │                                               ▼
                                        └──no──→ agent_loop (itera con tool calling   finalize → END
                                                 nativo mientras `agent_activo` siga)
Errores (executor/loop): error_diagnose → error_confirm → error_retry (acotado) → error_fallback
```

- **planner**: fast-paths deterministas + planes de 1..N pasos.
- **context_manager**: corre SIEMPRE entre planner e inferencia. Decide qué entra al modelo, compacta (protege VRAM) y emite `context_dump` para debug.
- **agent_loop**: tareas abiertas; el modelo elige tools del catálogo (`TOOL_REGISTRY`) con argumentos validados por schema (`TOOL_PARAMETROS`).
- **plan_executor**: despacha cada paso al nodo `node_<tool>` correspondiente.
- **Catálogo de tools** (fuente de verdad: `TOOL_REGISTRY`): `text, web, shell, launch, vision, codigo, memory, file_write, fs_read, fs_write, fs_mkdir, fs_list, computer_use, mcp, net_audit`.
- Cada tool deja "datos crudos" en un campo propio de `AetherState` (mapa `_CAMPO_DATOS_CRUDOS_POR_TOOL` en `graph_builder.py`): si hay datos → sintetiza el LLM; si la tool ya fijó `final_response` determinista → directo a `finalize` (evita reformular/alucinar).
- Estado compartido: `AetherState` (TypedDict) en `core/agent/graph_state.py`.

## 5. REGLAS DE ARQUITECTURA (invariantes — respetar sin excepciones)

| # | Regla | Dónde se aplica |
|---|---|---|
| R1 | **Local-first**: toda inferencia va a Ollama local (`OLLAMA_HOST`). Nada sale a servicios externos (excepción opt-in: Google Vision en `/play-roblox google`). | `core/agent/graph_nodes.py`, settings |
| R2 | **Un solo motor**: cualquier runtime (TUI, `aether task`, backend) invoca `procesar_orden_grafo()`. Nunca crear caminos de inferencia paralelos. | `core/services/graph_service.py` |
| R3 | **Registro único de tools**: para agregar/cambiar una tool tocar SOLO `tool_registry.py` (TOOL_REGISTRY + TOOL_PARAMETROS) → nodo `node_*` en `graph_nodes.py` → conexión en `graph_builder.py` → implementación en `core/tools/`. Nunca bypasear el registro. | `core/agent/`, `core/tools/` |
| R4 | **Contexto antes de inferencia**: `node_context_manager` corre siempre después del planner; ninguna inferencia lo saltea. | `core/agent/graph_builder.py` |
| R5 | **Tool calling estructurado**: el agent loop usa JSON schemas (`TOOL_PARAMETROS`), no parseo de texto libre, cuando `TOOL_CALLING_NATIVO=True`. | `core/agent/graph_nodes.py` |
| R6 | **Configuración en 2 capas**: `config.json` (defaults versionados, solo lectura lógica) + `config.local.json` (overrides locales/hardware, GITIGNORED). Mismo patrón MCP: `mcp_servers.json` (placeholders `{token}`) + `mcp_servers.local.json` (tokens reales, JAMÁS commiteado). Cambios de runtime → `config_manager.py` (validación + hot reload). | `core/config/` |
| R7 | **Datos de runtime fuera del repo**: memoria, notas, logs, screenshots y skills operativas viven en `BASE_AETHER` (`~/Aether`), override `AETHER_DATA_DIR`. El repo solo versiona código/comportamiento. | `core/config/settings.py`, `.gitignore` |
| R8 | **Autorización explícita de directorios**: antes de operar en un dir nuevo se pide confirmación (`dir_authorization.py`); los rechazos NO se persisten. Las rutas relativas de las tools se resuelven contra `settings.RUTA_TRABAJO` (vía `AETHER_CWD`), NUNCA contra el repo. | `core/config/dir_authorization.py` |
| R9 | **Memoria en 2 niveles**: (a) store de sesión JSON (`<BASE_AETHER>/db/memoria.json` + `.bak` rotativo); (b) memoria central SQLite v2 compartida por todos los runtimes (capas: conversación · usuario con `user_update()` e historial, nunca overwrite silencioso · aprendizaje con importance/strength y `consolidate()` · outcomes ~48 h TTL). Los tests NUNCA escriben memoria real (`conftest.py` fuerza `AETHER_CENTRAL_MEMORY=0`). | `core/memory/` |
| R10 | **Recuperación de errores acotada**: diagnose → confirm → retry con límites → fallback explícito. El agent loop cierra tras 3 fallos de una tool de búsqueda; dedup de URLs por turno; la outcome memory evita repetir intentos fallidos. | `graph_nodes.py`, `graph_builder.py` |
| R11 | **Compatibilidad gradual**: facades/alias legacy se mantienen mientras migra la v2 (p. ej. `settings.py` delega en `settings_v2.py`). Lo retirado va a `_legacy/` y NO se importa desde el árbol activo. | `core/`, `_legacy/` |
| R12 | **Español primero**: docstrings, mensajes al usuario e identificadores públicos en español. | todo el repo |
| R13 | **Backend aislado**: `backend/` tiene requirements propios y corre DESDE LA RAÍZ (imports absolutos `backend.api.*`; `cd backend && uvicorn api.main:app` FALLA con `ModuleNotFoundError`). No es parte del flujo soportado. | `backend/` |
| R14 | **Síntesis honesta**: si una tool ya fijó respuesta determinista, no se re-sintetiza con LLM. El consolidador de memoria descarta bloques de thinking del modelo. | `graph_builder.py`, `memory_manager.py` |
| R15 | **Observabilidad no invasiva**: `context_dump`, logs de pasos del agent loop, lifecycle explícito en backend (`QUEUED/RUNNING/COMPLETED/FAILED/CANCELLED`). Métricas de bench solo con `AETHER_BENCH_INSTRUMENT=1`. | varios |
| R16 | **Higiene de repo**: jamás commitear `.env`, `*.db`, memoria, `config.local.json`, `mcp_servers.local.json`, backups ni credenciales. Ver `.gitignore` (defensa en profundidad). | git |

## 6. VARIABLES DE ENTORNO Y CONFIGURACIÓN

| Variable | Efecto | Default |
|---|---|---|
| `AETHER_CWD` | Directorio de trabajo del agente; se fija ANTES de importar `core.config.settings`. | `$PWD` del lanzador |
| `AETHER_DATA_DIR` | Base de datos de runtime (`BASE_AETHER`). | `~/Aether` |
| `AETHER_PYTHON` | Intérprete forzado para `bin/aether`. | autodetección `env→crewai-env→.venv` |
| `AETHER_CENTRAL_MEMORY` | `0` desactiva la memoria central (los tests la apagan por defecto). | activada |
| `AETHER_CENTRAL_MEMORY_PATH` | Ruta de la memoria central SQLite. | `~/.aether/memory/` |
| `OLLAMA_HOST` | Endpoint de Ollama. | `http://localhost:11434` |
| `SEARXNG_URL` | Endpoint SearXNG (web search). | `http://localhost:8081` |
| `AETHER_BENCH_INSTRUMENT` | `1` → métricas a `~/.aether/bench_metrics.jsonl`. | `0` |
| `AETHER_WEB_PORT` / `WEB_UI_DIR` | Puerto del backend web / carpeta de estáticos. | `8000` / autodetección |
| `GOOGLE_APPLICATION_CREDENTIALS` / `GOOGLE_API_KEY` | Solo `/play-roblox google`. NUNCA en `config.json`. | — |

Claves relevantes de `config.json`: `MODELO` (default `ornith:9b`) · `MODELO_VISION` (= MODELO) · `MODO_AUTONOMO` · `TOOL_CALLING_NATIVO` · `TIMEOUT_CMD` · `NUM_CTX`/`NUM_PREDICT` · `MAX_STEPS_COMPUTER_USE` (default 8) · `SEARXNG_URL`. Validación de modelos y rangos: `ConfigValidator` en `config_manager.py`.

## 7. COMANDOS OPERATIVOS (verificación rápida)

```bash
# Dev (desde la raíz; venvs: env/ · crewai-env/ · .venv/)
python -m pytest tests                                     # suite completa
python -c "from core.agent.graph_builder import build_graph; build_graph(); print('ok')"  # smoke grafo
python -m py_compile run.py tui/app.py core/agent/graph_nodes.py
python run.py                                               # TUI

# CLI global
aether doctor                                               # diagnóstico del entorno
aether task "orden"                                         # one-shot

# Backend (DESDE LA RAÍZ; .venv es el que tiene fastapi)
.venv/bin/python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload

# Benchmark
python tps.py "modelo" "prompt"
```

## 8. GOTCHAS / NO ASUMIR SIN VERIFICAR

1. **El motor NO es CrewAI** pese al nombre del workspace (`mi_proyecto_crew`) y del venv `crewai-env/`. Es LangGraph + Ollama.
2. `cd backend && uvicorn api.main:app` **falla** (`ModuleNotFoundError: No module named 'backend'`): correr uvicorn desde la raíz con `backend.api.main:app`.
3. `docs/ARCHITECTURE.md` menciona `core/agent/graph_service.py` y `tools/audit_project.py`: **ya no existen**. Los reales: `core/services/graph_service.py` y `tools/hardware_probe.py`. Verificar contra código, no solo docs.
4. `rich.markup.escape` NO basta para Textual (solo escapa tags minúsculos): usar el saneo de `tui/app.py::_sanear_streaming`.
5. `AETHER_CWD` debe fijarse **antes** de importar `core.config.settings` (settings es lazy, pero así lo ven los procesos hijos).
6. Las skills operativas NO están en el repo: `<BASE_AETHER>/skills/<nombre>/SKILL.md` (carga dinámica).
7. La Web UI está en otro repo (`~/aether_web_ui`); acá el backend solo sirve estáticos.
8. Roblox: runtime en repo separado (`~/aether-roblox`); aquí solo `core/tools/roblox_bridge.py` (primitivas genéricas).
9. `graph_nodes.py` ≈5200 líneas: navegar por `def node_*`, no leer de corrido.
10. Alias/nombres históricos de settings y memoria se conservan a propósito (R11): antes de renombrar algo público, buscar referencias en todo el repo.

## 9. MAPA DE LECTURA POR TAREA (dónde empezar según el trabajo)

| Tarea | Leer en este orden |
|---|---|
| Cambiar razonamiento/grafo | `core/agent/graph_builder.py` → `graph_state.py` → `graph_nodes.py` (sección del nodo) |
| Agregar/modificar una tool | `core/agent/tool_registry.py` → `core/tools/<impl>.py` → `graph_nodes.py::node_*` → `graph_builder.py` |
| Memoria / persistencia | `core/memory/memory_manager.py` → `central_store_v2.py` → `context_builder.py` |
| Configuración / instalación | `core/config/settings.py` → `settings_v2.py` → `config_manager.py` → `installer/install-core.sh` |
| TUI / UX | `tui/app.py` → `engine_bridge.py` → `tui/widgets/*` |
| Backend web | `backend/api/main.py` → `backend/core/aether_service.py` → `backend/api/routes/*` |
| Tests nuevos | `conftest.py` → el test más parecido de `tests/` |
| Documentación | `README.md` (ES) → `docs/ARCHITECTURE.md` (EN) → `docs/es/` |

---
*Mantener este archivo actualizado al cambiar la estructura, un entry point o una regla de arquitectura. Verificado contra código: rama `development/aether-next`.*
