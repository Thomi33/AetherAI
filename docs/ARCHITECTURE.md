# AetherAI — Current Architecture

> Technical document based on the state of `main` inspected on 2026-10-03
> (after PR #4: v2 migration + agent-loop web-search fix).
> It describes the architecture implemented in the repository, not a future
> architecture. English docs live here; the Spanish originals are kept in
> [`docs/es/`](./es/).

## 1. Overview

AetherAI is a local, terminal-first agent oriented to Arch Linux. The active
engine uses **LangGraph** for orchestration and **Ollama** as the local model
runtime. The repository keeps historical components and a `backend/` surface,
but the project's documented supported flow is the CLI/TUI.

The current architecture combines two execution strategies:

1. **Fast-path / Tool Planning:** `node_planner` can build deterministic plans,
   especially for simple operations.
2. **Agent Loop:** for open-ended tasks, the model receives the structured tool
catalog and performs incremental tool calling; the graph returns to the loop
   while `agent_activo` stays active.

The engine's logical entry point is `procesar_orden_grafo()`.

## 2. High-level flow

```text
User / CLI
    │
    ▼
procesar_orden_grafo()
    │
    ├── register user turn
    ├── create_estado_inicial()
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
│  Errors: executor/agent → diagnose → confirm → retry    │
│                         ↘ fallback                      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Final response + memory consolidation
```

`core/agent/graph_builder.py` explicitly defines these nodes and their
conditional routes. The graph is compiled once and reused through
`get_graph()`; `reset_graph()` lets processes that need it invalidate that
instance.

## 3. Single engine entry point

`core/agent/graph_service.py` and the compatible facade
`core/services/graph_service.py` expose
`procesar_orden_grafo(orden, mem, modo_autonomo=True)`.

The function:

1. registers the user's turn;
2. gets the compiled graph;
3. creates a complete `AetherState` through `crear_estado_inicial()`;
4. runs `grafo.invoke()`;
5. returns `final_response`;
6. schedules memory consolidation after the exchange completes.

The existence of both `graph_service` paths is deliberate:
`core/agent/graph_service.py` is the engine implementation and
`core/services/graph_service.py` acts as a compatible facade for existing
callers.

## 4. State: `AetherState`

`core/agent/graph_state.py` defines the central state contract via
`TypedDict`.

### Input and control

- `orden`: the user's original request.
- `mem`: normalized memory.
- `modo_autonomo`: execution mode.
- `intent`, `done`, `terminado`, `tokens`, `ruta`: control and compatibility.

### Tool Planning

- `plan_activo`
- `plan_pasos`
- `plan_index`
- `plan_resultados`

These fields sustain the planning/fast-path route.

### Agent Loop

- `agent_activo`
- `agent_messages`
- `agent_pasos_log`
- `agent_images` (b64 thumbnails of this turn's image attachments,
  embedded into the user message when the model supports vision)

The transcript follows the message format Ollama uses
(`system/user/assistant/tool`), and the log keeps tool, arguments and result
for diagnostics.

### Tool results

The state reserves separate fields for web, shell, MCP, vision, filesystem
and computer use. It also keeps `_tool_args` for the structured arguments of
the current call.

### Context Manager

- `sesion_id`
- `context_slots`
- `context_dump`
- `tema_actual`
- `historial_filtrado`

### Error Handler

The state keeps the error context, maximum attempt count, authorization and
the proposed fix/diff/source, plus legacy aliases for compatibility.

The `crear_estado_inicial()` factory is the central source for initializing
state and normalizes memory before building it.

## 5. LangGraph orchestration

`core/agent/graph_builder.py` builds the `StateGraph`.

### Main route

```text
START
  → planner
  → context_manager
  → plan_executor | agent_loop
  → plan_synthesizer (when applicable)
  → finalize
  → END
```

### `planner`

The planner keeps deterministic fast-paths and decides whether the request
needs the planning route or the agent loop. The state produced by this phase
is used by `context_manager` to identify the initial topic.

### `context_manager`

Always runs after the planner and before inference. Its responsibility is to
select the relevant context, update the active topic and produce
`context_dump` for diagnostics.

The implementation uses `construir_contexto_memoria()` and
`construir_context_dump()` and propagates the result via `context_slots`.

### `plan_executor`

Executes a plan's steps through the real nodes registered in
`TOOL_REGISTRY`. After each execution the router decides whether to continue
the plan, synthesize the result, finish or enter the error handler.

### `agent_loop`

This is the native tool-calling route. The model receives the structured
tools and incrementally decides what to do. The node can loop back to
itself while `agent_activo=True`; when the goal is done the route continues
to `finalize`.

This design avoids forcing open-ended tasks to produce a complete plan up
front.

**Step limit and loop defenses.** The loop has **no fixed step limit by
design** — the user decides when to stop (Stop button / `/api/stop` /
Ctrl+C; cancellation is checked at the start of every step). The safety nets
that keep it from spinning forever:

1. **Exact-duplicate block:** a tool call already executed twice with
   identical arguments is not run again; the model gets closing guidance.
2. **Attempt memory ("if it didn't return a result, it didn't return a
   result"):** negative tool results (`sin_resultados` / `error`) are
   classified, given terminal guidance in the tool message, and persisted
   as outcomes in the shared central memory, so the same dead end isn't
   retried in later turns.
3. **Hard cap:** search/reading tools (`web`, `fs_read`, `fs_list`,
   `vision`) that accumulate 3 failed attempts in the turn are no longer
   executed; the loop closes with whatever it has.
4. **Per-step instruction:** the instruction the model writes in the tool
   call arguments is the step's base text (falling back to the original
   request), so a refined instruction really changes what the step does —
   the fix for the "4 steps searching the same thing" bug.
5. **Web dedup:** `node_web` never re-reads a URL already read in the turn;
   a repeated search reads the first new result URL, and when everything
   has been read it returns `[SIN RESULTADOS NUEVOS]`, which classifies as
   `sin_resultados` and feeds the hard cap. The model can also pass an
   explicit `query` (searched as-is) or `url` (read directly) via the
   tool schema.

### `plan_synthesizer`

Turns raw tool results into a useful answer when a synthesis stage is still
needed. The builder skips this stage when a tool already produced a
deterministic answer and there is no new raw data to synthesize.

### `finalize`

Closes the graph and leaves the final answer available in `final_response`.
It also takes part in registering Aether's turn.

## 6. Tool registry

`core/agent/tool_registry.py` is the central source of truth for the tools
known to the agent.

Each entry links:

```text
tool name
    ├── node
    ├── instruccion_requerida
    └── descripcion
```

In addition, `TOOL_PARAMETROS` defines the argument schemas used by native
tool calling.

The registry currently contains:

| Tool | Conceptual function |
|---|---|
| `text` | conversation/direct answer |
| `web` | web search (optional structured args: `query`, `url`) |
| `shell` | command generation and execution |
| `launch` | application launching |
| `vision` | screen capture and analysis |
| `codigo` | code generation/execution |
| `memory` | memory management |
| `file_write` | legacy result/instruction-based writing |
| `extract` | extraction/cleaning of previous results |
| `mcp` | invocation of MCP tools |
| `computer_use` | perception + action over the interface |
| `subagent` | isolated sub-agents for parallel tasks |
| `fs_write` | structured writing of one or several files |
| `fs_read` | file reading |
| `fs_mkdir` | directory creation |
| `fs_list` | directory listing |

`validar_tool_call()` performs a quick validation of existence, argument
type, required instructions and, for MCP, `server` + `name`. It does not
replace full JSON Schema validation.

## 7. Tools and execution layers

The concrete tool implementations live mostly in `core/tools/`:

- `web_search.py`: web search (SearXNG with DuckDuckGo fallback).
- `url_reader.py`: URL reading.
- `shell_executor.py`: shell execution.
- `flatpak_manager.py`: application launching.
- `vision.py`: capture/visual analysis.
- `ydotool_wrapper.py`: interface control (v2 — `hyprctl` + `ydotool`,
  without VLM fallback or complex verification; replaces the removed
  `computer_control.py`).
- `mcp_client.py`: MCP client.
- `file_writer.py`: legacy writer.
- `filesystem_tool.py`: structured filesystem operations.

The graph nodes act as the orchestration layer; the `core/tools/`
implementations encapsulate the concrete operations.

## 8. Working directory and data separation

Aether distinguishes between:

- **Project:** source code and versioned behavior.
- **Runtime home:** `~/Aether`, used for DB, logs, screenshots and other
  persistent data.
- **Working directory:** the path where the user opened Aether, or the one
  given via `--workdir`.

`bin/aether` preserves the original `$PWD` in `AETHER_CWD` before entering
the project directory. `core/config/settings.py` resolves `RUTA_TRABAJO`
dynamically from that variable or from the current cwd.

This lets you run Aether over an external project without moving the
agent's installation.

New working directories go through explicit authorization via
`core/config/dir_authorization.py`, and sensitive system paths have
additional controls.

## 9. Configuration and local runtime

Configuration is **two-layered**:

- `core/config/config.json` — versioned defaults (shipped with the repo,
  read-only).
- `core/config/config.local.json` — local overrides, ignored by Git.

`core/config/settings.py` centralizes the main values as a compatibility
adapter, while `core/config/settings_v2.py` provides the simplified loader
(JSON load + shallow merge, ~80 lines, no watchers/validators/locks).

Key values:

- `OLLAMA_HOST`: `http://localhost:11434`.
- `SEARXNG_URL`: `http://localhost:8081`.
- `MODELO`: currently configured text model.
- `MODELO_VISION`: currently configured multimodal model.
- `TOOL_CALLING_NATIVO=True`.
- `NUM_CTX=32768` (installer profiles adjust it to the hardware).
- `OLLAMA_KEEP_ALIVE=-1`.
- generation options and context limits.
- paths for DB, logs, screenshots, embeddings, backups and skills.

## 10. Memory

The memory subsystem lives in `core/memory/`.

### Layers

```text
RAM memory
    │
    ├── core
    ├── summary
    ├── conversation
    └── command history
          │
          ▼
     current.db
          │
          ├── conversations
          ├── commands
          ├── memories
          ├── core_memory
          └── summary_memory
```

`memory_manager.py` uses `~/Aether/db/current.db` as the production DB. The
RAM structure is normalized through `normalizar_mem()`.

Conversational context is not injected without limits: `settings.py` defines
per-turn and character budgets. For multi-tool plans, conversational history
can be trimmed to zero turns to avoid contamination between steps; previous
step results are the relevant operating context.

Consolidation of the accumulated summary is scheduled after completing an
order (v2 moved this into `memory_manager`; the old `consolidator.py` was
removed).

### Shared central memory (v2: SQLite)

`core/memory/central_store_v2.py` implements the shared central memory on
stdlib SQLite — same API surface as the previous JSON store, without the
complex dedup/fingerprint/TTL-thread machinery. Tables:

- `learnings`: memories with `importance` and `strength`; use reinforces,
  disuse decays, `forget()` hides without deleting.
- `outcomes`: operational memory of failed attempts (namespace `intentos`,
  ~48 h TTL, daily decay, 200-entry cap) — "if it didn't return a result,
  it didn't return a result".
- `user_facts`: stable user facts (`user_set()` never silently
  overwrites; corrections via `user_update()` with history).
- `conversations`: per-runtime session summaries.

It lives outside any runtime (default `~/.aether/memory/`, override with
`AETHER_CENTRAL_MEMORY_PATH`, opt-out with `AETHER_CENTRAL_MEMORY=0`) and is
shared by every surface: TUI, Web UI, Roblox Player and future runtimes.
The context builder injects learnings into the `[APRENDIZAJES]` slot and
recent failed attempts into the `[INTENTOS RECIENTES]` slot.

## 11. MCP and extensibility

MCP is integrated as a registry tool (`mcp`) and runs through
`core/tools/mcp_client.py`. The tool-calling contract requires `server`,
`name` and, when applicable, `arguments`.

Server configuration is two-layered: `mcp_servers.json` (versioned template)
and `mcp_servers.local.json` (tokens/secrets, ignored by Git) — the same
scheme as `config.json`/`config.local.json`.

This lets connected MCP servers extend Aether's capabilities without turning
every external integration into an independent hardcoded node.

## 12. Error handling

Execution errors have an explicit route:

```text
error
  │
  ▼
error_diagnose
  │
  ├── proposed fix → error_confirm → error_retry ──► execution
  │
  └── no fix / limit → error_fallback
```

The state distinguishes the context (`shell`, `launch`, `codigo`), keeps the
current attempt and limits retries. `graph_builder.py` uses a general
maximum of 3 attempts and allows specific per-context limits defined in
`graph_state.py`.

## 13. Models

The current configuration separates the text model (`MODELO`) from the
vision model (`MODELO_VISION`); when the main model is multimodal, vision
runs on it and no second model is needed.

The v2 migration removed the old observational model policy
(`model_policy.py`): the configured model is used directly. The decision
point for future adaptive policies now lives in the plain config layer.

## 14. CLI and launcher

The relevant entry points are:

```text
bin/aether
    └── bin/aether_run.py
          └── CLI runtime (subcommands: version, task, doctor)

run.py
    └── TUI
          └── core.services.graph_service
```

The `bin/aether` launcher supports running from any directory plus commands
like `task`, `doctor`, `--version` and `--help`. The old deprecated
interactive loop in `cli/` was removed by the v2 migration.

## 15. Historical / unsupported components

The repository contains components that must not be confused with the
current engine:

- `core/services/aether_service.py`: historical CrewAI-based implementation.
- `backend/`: FastAPI API and existing web pieces, but the README declares
  them outside the supported CLI flow. (The backend does run inferences
  with an explicit `QUEUED/RUNNING/COMPLETED/FAILED/CANCELLED` lifecycle.)
- legacy aliases/fields inside state and memory: kept for compatibility.
- `_legacy/`: retired artifacts (old databases/dumps) preserved outside the
  main tree.

The active architecture is **LangGraph + Ollama + tools/MCP + local
memory**, not CrewAI.

## 16. Logical structure

```text
AetherAI/
├── bin/                         # global launcher and CLI runtime
├── core/
│   ├── agent/                   # graph, state, planner, loop, registry
│   ├── config/                  # two-layer config and directory authorization
│   ├── memory/                  # memory + SQLite stores (session + central v2)
│   ├── parser/                  # shell/response parsing
│   ├── services/                # compatible facades/services
│   ├── skills/                  # reusable behavior
│   ├── tools/                   # concrete operations (incl. ydotool_wrapper)
│   └── utils/                   # utilities
├── tui/                         # Textual terminal UI
├── skills/                      # operational SKILL.md files
├── scripts/                     # utilities and QA verification harnesses
├── tests/                       # test suite
├── docs/                        # docs (English) + docs/es (Spanish originals)
├── backend/                     # experimental web surface, outside the CLI flow
└── _legacy/                     # retired artifacts
```

## 17. Observable architectural principles

1. **Local-first:** the main inference runtime is local Ollama.
2. **Explicit orchestration:** LangGraph controls states and transitions.
3. **Single tool registry:** tools are described and validated from a
   central registry.
4. **Structured tool calling:** the agent loop uses argument schemas, not
   exclusively free text.
5. **Context before inference:** `context_manager` selects information
   before the downstream nodes.
6. **Contextual filesystem:** the working directory can be external to the
   project folder.
7. **Separate persistent memory:** runtime data lives outside versioned
   code.
8. **Gradual compatibility:** legacy aliases and facades exist while the
   new architecture settles.
9. **Error recovery:** executions can diagnose, propose fixes, retry and
   fall back.
10. **Prepared observability:** the system ships `context_dump`, agent-loop
    step logs, and an explicit inference lifecycle
    (QUEUED/RUNNING/COMPLETED/FAILED/CANCELLED) in the web backend.
11. **Bounded loops:** the agent loop has no step limit by design (the user
    decides), but duplicate calls, failed-search caps, per-turn URL dedup
    and attempt memory keep it from repeating itself forever.

## 18. Implementation sources

This document was checked directly against the main components of `main`,
especially:

- `core/agent/graph_builder.py`
- `core/agent/graph_state.py`
- `core/agent/graph_nodes.py`
- `core/agent/tool_registry.py`
- `core/agent/node_context_manager.py`
- `core/agent/graph_service.py`
- `core/agent/streaming.py`
- `core/services/graph_service.py`
- `core/config/settings.py` / `core/config/settings_v2.py`
- `core/memory/memory_manager.py`
- `core/memory/central_store_v2.py`
- `core/tools/*` (incl. `ydotool_wrapper.py`)
- `README.md`
