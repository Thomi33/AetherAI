# Approved models for Aether

> Locally-hosted models **officially evaluated and approved** for use with
> Aether (Ollama). Compatibility documentation: what was tested, what was
> observed, and which limitations are on record. Runtime details:
> [`ARCHITECTURE.md`](./ARCHITECTURE.md) · agent-dense map:
> [`../../ai_map.md`](../../ai_map.md).

Status: **Ornith** and **Granite** passed the evaluations performed by the
maintainer (2026-10-10) and are listed as compatible with Aether. This does NOT
imply identical performance or the same hardware requirements. Completed
evaluations are not repeated; their conclusions are kept as recorded. No
normalized benchmarks are versioned in this repo: the standing results are
those observed in the maintainer's evaluation sessions.

---

## Ornith — approved

| Field | Recorded value |
|---|---|
| Exact Ollama name | `ornith-1.5:9b` |
| Quantization / architecture | `Q4_K_M` / `qwen35` |
| Parameters / context | 9.0B / 262,144 tokens |
| Capabilities (Ollama `/api/show`) | `tools`, `thinking`, `completion`, `vision` |
| On-disk size (measured) | 6.6 GB |
| Role in Aether | Native, default model of the project (`MODELO` default in `config.json`); the runtime (planner, agent loop, synthesis) is developed and validated on it |

**Recorded strengths**

- Native multimodal (`vision`): sees attached images directly (inline
  thumbnails) and feeds the video-frame analysis pipeline.
- Native reasoning (`thinking`) and structured tool calling: the agent loop is
  tuned around its behavior.
- Long context (262k) with configurable `NUM_CTX`; primary model recommended by
  the hardware probe for HIGH/ULTRA tiers (≈12 GB+ VRAM or 32 GB RAM).

**Tested configuration:** `ornith-1.5:9b` as `MODELO` (HIGH/ULTRA probe tiers,
`installer/install-core.sh`). Variants recorded in
`ConfigValidator.VALID_MODELS`: `ornith:9b`, `ornith:7b`, `ornith-1.5:9b`.

---

## Granite — approved (with a documented autonomy limitation)

| Field | Recorded value |
|---|---|
| Exact Ollama name | `hf.co/ibm-granite/granite-4.2-8b-GGUF:Q6_K` |
| Quantization / architecture | `Q6_K` / `granite` |
| Parameters / context | 8.79B / 131,072 tokens |
| Capabilities (Ollama `/api/show`) | `tools`, `thinking`, `completion` — **no `vision`** |
| On-disk size (measured) | 7.2 GB |

**Recorded strengths**

- Structured tool calling and native reasoning (`tools` + `thinking`).
- Fully functional as Aether's primary model: planner, agent loop and
  synthesis operate normally with it.
- Since it does not declare `vision`, Aether's multimodal pipeline compensates
  automatically: images and video frames are analyzed through
  `MODELO_VISION` (captions) instead of inline (see
  `core/config/capabilities.py`).

### Recorded limitation — does not affect approval

During an agentic task, Granite was observed to **stop before completing the
work and require manual intervention** to continue. Documented as an autonomy
limitation; **the cause is not attributed** — it may lie in the model, the
runtime, or context management.

The relevant distinction:

| Dimension | Status |
|---|---|
| **Functional compatibility** (tool calling, graph flow, tool usage, instruction following) | ✅ Approved |
| **Sustained autonomy on long tasks** (completing without intervention) | ⚠ Not guaranteed |

### Proposed recovery mechanisms (NOT implemented)

Recorded for a future runtime task — not part of this compatibility
documentation nor of the installer:

1. **Detect** when a task is interrupted without completing (a stop distinct
   from a normal turn ending).
2. **Preserve** the original objective and the graph's execution state.
3. **Retry** with the task's pertinent context (objective + last step +
   partial result), avoiding an ambiguous `retry` the model may read as a new
   order.
4. **Verify real progress** after the retry (new steps/evidence, not just a
   model reply).
5. **Bound retries** (e.g. N=2) and **escalate to human intervention** when
   there is no progress, with an explicit state message.

---

## Quick comparison

| | Ornith 1.5 | Granite 4.2 |
|---|---|---|
| Approved for Aether | ✅ | ✅ |
| Native vision | ✅ (`vision`) | ❌ (uses `MODELO_VISION`) |
| Thinking / tools | ✅ / ✅ | ✅ / ✅ |
| Sustained autonomy on long tasks | No limitation recorded | ⚠ May stop and require intervention |
| Recorded requirement (probe) | ≈12 GB+ VRAM or 32 GB RAM (HIGH tier) | 8.79B `Q6_K`, 7.2 GB on disk; no formal requirement versioned in the repo |
| Installer | Recommended option (probe default HIGH/ULTRA) | Selectable approved option |

## How to choose

- **~12 GB+ GPU / ample RAM** → Ornith 1.5 (`ornith-1.5:9b`): native
  multimodal and the project's reference model.
- **You prefer Granite / tighter hardware** →
  `hf.co/ibm-granite/granite-4.2-8b-GGUF:Q6_K`: approved; accept the autonomy
  note for long agentic tasks.
- Very limited hardware → the probe recommends compact models (POTATO/LOW/MID
  tiers); you can still configure **any custom Ollama model** via `--model` or
  `config.local.json`.

The installer (`installer/install-core.sh`) offers both approved models,
detects already-installed ones, and keeps the custom-model option.
