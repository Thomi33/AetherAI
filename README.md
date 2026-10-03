# Aether

> Local, terminal-first agent. It converses in Spanish (by design); these docs
> are in English — the Spanish originals live in [`docs/es/`](./docs/es/).
> **LangGraph + Ollama + real tools + persistent memory.**

Aether is a local assistant that chats, analyzes projects, searches for fresh
information, reads and writes files, runs commands, generates code, watches
the screen and chains multiple actions without sending your project to
external services. The model runs on your own machine through Ollama.

## Table of contents

- [Installation](#installation)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Tools](#tools)
- [Configuration and performance](#configuration-and-performance)
- [Memory and data](#memory-and-data)
- [Project architecture](#project-architecture)
- [Development and tests](#development-and-tests)
- [Troubleshooting](#troubleshooting)

## Installation

### Download Aether

You don't need to manually install Python, pip, Ollama or Aether's
 dependencies before starting. The installer takes care of preparing the
 environment.

You can download the repository in two ways:

**With Git:**

```bash
git clone https://github.com/Thomi33/AetherAI.git
cd AetherAI
```

**Without Git:** open the repository on GitHub, choose **Code → Download
ZIP**, unzip the file and open a terminal inside the `AetherAI` folder.

### Install

From the repository root:

```bash
chmod +x install.sh
./install.sh
```

**That's it.** The installer automatically takes care of:

1. Detecting CPU, RAM, VRAM and available disk space.
2. Choosing the hardware tier (`POTATO`, `LOW`, `MID`, `HIGH` or `ULTRA`).
3. Selecting the model and parameters appropriate for that machine.
4. Installing Aether's core Python dependencies.
5. Preparing Aether's virtual environment.
6. Installing/verifying Ollama when applicable.
7. Downloading the selected model, with your authorization.
8. Setting up memory, SQLite and runtime parameters.
9. Installing the global `aether` command into `~/.local/bin` and making it
   available on the `PATH`.
10. Running a final validation of the environment and the model.

During installation you may only be asked a few normal decisions, such as
the `sudo` password, whether you want to keep data in `~/Aether` and whether
you want to download the selected model.

There is no need to run `pip` commands afterwards, create another virtualenv
or install Aether's dependencies by hand.

### Hardware and profiles

The installer adapts the model and parameters to the detected machine. For
example, a very limited machine may end up as `POTATO` and use
`qwen2.5:1.5b`, while a more powerful machine gets a higher profile.

The recommended context is adjusted automatically according to the hardware,
with a limit of `65536`.

Vision uses the same configured main model when that model supports
multimodal input; no second vision model is downloaded.

## Usage

### Main TUI

After installing:

```bash
aether
```

It also works directly from the project:

```bash
python run.py
```

### Global launcher

The installer creates the `aether` command in `~/.local/bin`, so you can open
Aether from any directory:

```bash
aether
aether task "list the project files"
aether doctor
aether --version
```

You can also run Aether over a specific directory:

```bash
cd ~/Project
aether
aether task "create a README for this project"
aether --workdir ~/Project task "run the tests"
```

Before operating on a new directory, Aether asks for authorization. Denials
are not persisted: if you start Aether again, it will ask again.

### Stopping an inference

During an operation you can use any of these options:

- The **Stop** button in the TUI (or the Web UI's `/api/stop`).
- `Ctrl+C`.
- The `/stop` command.

Cancellation is cooperative: the flow stops when Ollama delivers the next
available event.

## How it works

Every request becomes an `AetherState` and flows through a LangGraph graph:

```text
START
  -> planner
  -> context_manager
  -> plan_executor or agent_loop
  -> plan_synthesizer/finalize
  -> END
```

- **Planner:** resolves simple fast-paths and prepares single or multi-step
  plans.
- **Agent loop:** uses Ollama's native tool calling for open-ended tasks. It
  has no fixed step limit — you decide when to stop. Several safety nets keep
  it from spinning forever:
  - the same tool call with identical arguments is never repeated;
  - negative results ("no results", errors) are classified, given closing
    guidance, and persisted as outcomes so they aren't retried in later turns;
  - search/reading tools that accumulate 3 failed attempts in a turn force
    the loop to close ("if it didn't return a result, it didn't return a
    result");
  - the instruction the model writes for each step is what that step
    actually does, so refining the instruction really changes the search;
  - `web` never re-reads a URL already read in the same turn: repeated
    searches read the next new source, and when there's nothing new they
    return `[SIN RESULTADOS NUEVOS]` and count toward the hard cap.
- **Executor:** dispatches each plan step to the tool's real implementation.
- **Synthesizer:** combines results when there are multiple actions.
- **Error handler:** diagnoses errors, can search for a fix, retries within
  limits and leaves the error explicit if it can't be resolved.
- **Context manager:** prunes and compacts context to avoid unnecessary VRAM
  consumption.

When Aether isn't certain, its prompt tells it to search the internet first
via `web` when the fact is current, versioned or externally verifiable. For
local facts it should prefer files, shell and system tools.

The TUI shows real execution states, for example:

```text
[•] Analyzing request...
[•] Preparing plan...
[•] Running tool: shell
compacting context to save your VRAM...
[✓] Operation completed
```

Personality messages are optional, brief and secondary; they never replace
real states or invent actions.

## Tools

| Tool | Function |
|---|---|
| `text` | Direct conversation with streaming. |
| `web` | Web search and reading. Optional args: `query` (exact search, not rewritten) and `url` (read that specific source). |
| `shell` | Runs `zsh` commands with barriers and timeout. |
| `launch` | Opens applications and Flatpak packages. |
| `vision` | Captures and analyzes the screen with the multimodal main model. |
| `codigo` | Generates, saves and runs Python, Bash or Java. |
| `memory` | Queries, stores or deletes memories. |
| `file_write` | Saves a step's result to a file. |
| `extract` | Extracts/cleans content from a previous step's raw result. |
| `fs_read` / `fs_write` | Reads and writes files with explicit paths (`fs_write` supports multi-file atomic writes). |
| `fs_mkdir` / `fs_list` | Creates directories and lists contents. |
| `subagent` | Spawns isolated sub-agents for parallel tasks. |
| `computer_use` | Controls mouse/keyboard via `ydotool` + `hyprctl`. |
| `mcp` | Invokes configured MCP servers. |

Relative paths resolve inside the authorized working directory, not inside
the repository folder. By default memory, notes and logs live separately in
`~/Aether`; if you choose not to use a separate home during onboarding, they
live in `<project>/.aether-data/`.

### computer_use (v2: ydotool wrapper)

Interface control lives in `core/tools/ydotool_wrapper.py`. Window search,
focus and workspace switching go straight to Hyprland (`hyprctl`); clicks,
typing, mouse movement and key holds go through `ydotool`. The v2 wrapper is
deliberately simple: the old implementation's VLM fallback, verification
dataclasses, complex retries and context caching were removed — an action
either runs or fails fast, and visual diagnosis is left to the explicit
`vision` tool when you ask for it.

## Configuration and performance

Configuration follows a **two-layer** scheme:

- [`core/config/config.json`](./core/config/config.json) — versioned defaults
  (read-only; ships with the repo).
- `core/config/config.local.json` — your local overrides, ignored by Git.

Manually edited values always take priority over the program defaults. MCP
servers follow the same pattern: `mcp_servers.json` (versioned template) plus
`mcp_servers.local.json` for tokens and secrets, which never get committed.

The most important values are:

- `MODELO`: main model for text, tool calling and vision.
- `AETHER_DATA_DIR`: folder where memory, notes and logs are kept.
- `NUM_CTX`: effective context; the installer adjusts it to the hardware.
- `NUM_PREDICT`: maximum generated tokens.
- `OLLAMA_KEEP_ALIVE`, threads and batch: latency and memory parameters.
- `STT_ENABLED`: optional voice dictation.

A bigger context is not always faster: it depends on RAM, VRAM,
quantization and the chosen model.

## Memory and data

Aether keeps temporary memory in RAM and persistent memory through the
controlled subsystem in [`core/memory/store`](./core/memory/store):

- Production: `$AETHER_DATA_DIR/db/current.db`.
- Tests: `$AETHER_DATA_DIR/db/staging.db`.
- Snapshots: `$AETHER_DATA_DIR/db/snapshots/`.
- Backups: `$AETHER_DATA_DIR/db/backups/`.

Writes go through `MemoryStore`, versioned migrations and an integrity guard.
The old `memoria.db` at the repository root was a legacy artifact unused by
the current runtime; it was preserved outside the main tree in `_legacy/data/`
together with the old dump so no historical data is lost.

During long inferences, the consolidated summary is also saved to
`notes/contexto_importante.md` before compacting the context. That way
important data survives even when the prompt shrinks. During onboarding, the
installer asks whether Aether should have its own folder. If you choose no,
it uses `<project>/.aether-data/` for the database and its notes.

### Shared central memory

On top of the per-session storage above, Aether has a **shared central
memory** (`core/memory/central_store_v2.py`) implemented on stdlib SQLite
that lives outside any runtime (default `~/.aether/memory/`, override with
`AETHER_CENTRAL_MEMORY_PATH`) and is the same for everyone: terminal, Web
UI, Roblox Player and future runtimes. Four tables back its layers:

- **Conversations** (`conversations`): per-session temporal context,
  compactable.
- **User** (`user_facts`): stable facts about the user; `user_set()` never
  silently overwrites, corrections go through `user_update()` with history.
- **Learnings** (`learnings`): memories with `importance` (1–10) and
  `strength` (0–1). Use reinforces them (asymptotically: never saturates),
  disuse weakens them (`consolidate()`, scheduled after every turn), and
  `forget()` makes them inaccessible **without deleting** (only `hard=True`
  deletes). Personality is cumulative: `personality_signals()` adds patterns,
  never replaces anything.
- **Outcomes** (`outcomes`): operational, ephemeral memory of what was
  already tried and gave no result — "if it didn't return a result, it
  didn't return a result". Entries have a ~48 h TTL, daily decay and a cap
  of 200. The agent loop classifies negative tool results, appends closing
guidance to the tool message, and force-closes the loop after 3 failed
attempts of a search/reading tool in the turn; the outcome is persisted so
  the same dead end isn't retried in later turns.

The context builder injects relevant learnings into the `[APRENDIZAJES]`
slot and recent failed attempts into the `[INTENTOS RECIENTES]` slot (both
disableable with `AETHER_CENTRAL_MEMORY=0`; injection never reinforces
memories — only real use does).

## Project architecture

```text
.
├── run.py                    # Main entry point
├── install.sh                # Installer and hardware profiling
├── installer/                # Installer internals (install-core.sh)
├── bin/                      # Global aether launcher
├── core/
│   ├── agent/                # Graph, planner, loop and tool registry
│   ├── config/               # Config (two-layer) and directory authorization
│   ├── memory/               # Memory: RAM, SQLite store, central memory v2
│   ├── services/             # Single entry point to the graph
│   ├── skills/               # Reusable behavior
│   ├── tools/                # Shell, web, vision, filesystem, ydotool
│   ├── parser/               # Shell/response parsing
│   └── utils/                # Utilities
├── tui/                      # Textual interface and execution states
├── skills/                   # Reusable instructions (SKILL.md)
├── scripts/                  # Utility and QA verification scripts
├── tests/                    # Automated tests
├── docs/                     # English docs + Spanish originals (docs/es/)
├── backend/                  # Experimental FastAPI backend, not installed by default
└── _legacy/                  # Retired artifacts kept for reference
```

Operational skills live in `skills/<name>/SKILL.md`. They stay as separate
files because Aether discovers and loads them dynamically; they are not
redundant documentation.

## Optional backend

The FastAPI backend is experimental and **not part of the normal TUI
installation**. It is kept separate so its older dependencies don't
interfere with the current runtime. The future WebUI will be developed in a
separate repository.

The backend runs inferences with an explicit lifecycle —
`QUEUED → RUNNING → COMPLETED/FAILED/CANCELLED` — with per-inference logs and
no orphan threads, so the Web UI can always show and stop what is actually
running.

If you need to work with the experimental backend manually, start it **from
the repository root** (package imports are absolute, `backend.api…`):

```bash
crewai-env/bin/python -m pip install -r backend/requirements.txt
.venv/bin/python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000 --reload
```

> Note: the old command (`cd backend && uvicorn api.main:app`) fails with
> `ModuleNotFoundError: No module named 'backend'`.

## Development and tests

The normal installer already creates and configures the virtual environment
(`.venv`). For development:

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m py_compile run.py tui/app.py core/agent/graph_nodes.py
```

### Roblox runtime

Roblox-specific automation lives in the separate repository
`~/aether-roblox`. Aether keeps the generic control primitives and connects
to the runtime through `core.tools.roblox_bridge`.

From the TUI everything is controlled with a single command:

```text
/play-roblox
/play-roblox google
/play-roblox status
/play-roblox stop
```

Startup looks for and focuses `org.vinegarhq.Sober` before launching the
runtime. It uses Ollama by default; `/play-roblox google` selects Google
Cloud Vision for that run. The credential must exist in the environment,
never stored in `config.json`:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
# or: export GOOGLE_API_KEY="..."
```

Install the runtime extra once:

```bash
cd ~/aether-roblox
venv/bin/pip install -e '.[google]'
```

The runtime coordinates perception, reaction, WASD movement and camera so
several loops never fight over focus or input devices.

To check that the graph compiles:

```bash
.venv/bin/python -c "from core.agent.graph_builder import build_graph; build_graph(); print('ok')"
```

The audit script lives in [`tools/audit_project.py`](./tools/audit_project.py)
and serves as manual diagnostics; it is not part of normal startup.

## Troubleshooting

### `aether` doesn't show up after installing

The installer adds `~/.local/bin` to the `PATH`. If a terminal was already
open during installation, restart that terminal so it inherits the new
environment.

### Ollama doesn't respond

```bash
ollama serve
ollama list
```

In a normal installation the installer already verifies Ollama and downloads
the selected model when you authorize it.

### The model responds slowly

Lower `NUM_CTX` or `NUM_PREDICT`, check available VRAM and verify Ollama is
using the GPU. Hardware and model are the main limits; Aether avoids
redundant work, compacts context and caps loops, but it can't speed up the
model's intrinsic generation.

### Directory rejected

A rejection closes that run and is not persisted. Start Aether again to
authorize it whenever you want.

### Vision doesn't work

Check `grim`, Wayland/Hyprland and that the configured model supports images:

```bash
command -v grim
ollama show "$(.venv/bin/python -c 'import json; print(json.load(open("core/config/config.json"))["MODELO"])')"
```

## License and status

Project under active development. Review changes directly in Git and never
commit credentials, `.env` files, personal databases or local
configurations.
