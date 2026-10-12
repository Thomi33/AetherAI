# Modelos aprobados para Aether

> Modelos locales **oficialmente evaluados y aprobados** para usarse con Aether
> (Ollama). Documentación de compatibilidad: qué se probó, qué se observó y qué
> limitaciones quedaron registradas. Detalle técnico del runtime:
> [`ARCHITECTURE.es.md`](./ARCHITECTURE.es.md) · mapa denso para agentes:
> [`../../ai_map.md`](../../ai_map.md).

Estado: **Ornith** y **Granite** aprobaron las evaluaciones realizadas por el
mantenedor (2026-10-10) y figuran como modelos compatibles con Aether. Esto NO
implica idéntico rendimiento ni los mismos requisitos de hardware. Las
evaluaciones ya completadas no se repiten; sus conclusiones se conservan tal
como fueron registradas. No hay métricas normalizadas (benchmarks formales)
versionadas en este repo: los resultados vigentes son los observados en las
sesiones de evaluación del mantenedor.

---

## Ornith — aprobado

| Dato | Valor registrado |
|---|---|
| Nombre exacto en Ollama | `ornith-1.5:9b` |
| Cuantización / arquitectura | `Q4_K_M` / `qwen35` |
| Parámetros / contexto | 9.0B / 262 144 tokens |
| Capabilities (Ollama `/api/show`) | `tools`, `thinking`, `completion`, `vision` |
| Peso en disco (medido) | 6.6 GB |
| Rol en Aether | Modelo nativo y por defecto del proyecto (`MODELO` default de `config.json`); el runtime (planner, agent loop, síntesis) se desarrolla y valida sobre él |

**Fortalezas registradas**

- Multimodal nativo (`vision`): ve imágenes adjuntas directamente (thumbnails
  inline) y alimenta el pipeline de análisis de video por frames.
- Razonamiento nativo (`thinking`) y tool calling estructurado: el agent loop
  está afinado sobre su comportamiento.
- Contexto largo (262k) con `NUM_CTX` configurable; es el primario recomendado
  por el hardware probe para tiers HIGH/ULTRA (≈12 GB+ de VRAM o 32 GB RAM).

**Configuración probada:** `ornith-1.5:9b` como `MODELO` (tiers HIGH/ULTRA del
probe, `installer/install-core.sh`). Variantes registradas en
`ConfigValidator.VALID_MODELS`: `ornith:9b`, `ornith:7b`, `ornith-1.5:9b`.

---

## Granite — aprobado (con limitación de autonomía documentada)

| Dato | Valor registrado |
|---|---|
| Nombre exacto en Ollama | `hf.co/ibm-granite/granite-4.2-8b-GGUF:Q6_K` |
| Cuantización / arquitectura | `Q6_K` / `granite` |
| Parámetros / contexto | 8.79B / 131 072 tokens |
| Capabilities (Ollama `/api/show`) | `tools`, `thinking`, `completion` — **sin `vision`** |
| Peso en disco (medido) | 7.2 GB |

**Fortalezas registradas**

- Tool calling estructurado y razonamiento nativo (`tools` + `thinking`).
- Funcional como modelo primario de Aether: planner, agent loop y síntesis
  operan con él con normalidad.
- Al no declarar `vision`, el pipeline multimodal de Aether lo compensa de
  forma automática: las imágenes y los frames de video se analizan con
  `MODELO_VISION` (captions) en vez de inline (ver `core/config/capabilities.py`).

### Limitación registrada — sin efecto sobre la aprobación

Durante una tarea agéntica se observó que Granite **puede detenerse antes de
completar el trabajo y requerir una intervención manual** para continuar.
Se documentó como limitación de autonomía; **la causa no está atribuida** —
puede estar en el modelo, en el runtime o en la gestión del contexto.

La distinción relevante:

| Dimensión | Estado |
|---|---|
| **Compatibilidad funcional** (tool calling, flujo del grafo, uso de herramientas, respuesta a instrucciones) | ✅ Aprobada |
| **Autonomía sostenida durante tareas largas** (completar sin intervención) | ⚠ No garantizada |

### Propuesta de mecanismos de recuperación (NO implementada)

Registrada para una tarea futura del runtime — no forma parte de esta
documentación de compatibilidad ni del instalador:

1. **Detectar** cuándo una tarea se interrumpe sin haberse completado
   (corte distinto de finalización normal del turno).
2. **Conservar** el objetivo original y el estado de ejecución del grafo.
3. **Reintentar** con el contexto pertinente de la tarea (objetivo + último
   paso + resultado parcial), evitando un `retry` ambiguo que el modelo pueda
   interpretar como una orden nueva.
4. **Verificar progreso real** tras el reintento (nuevos pasos/evidencia,
   no solo una respuesta del modelo).
5. **Limitar los reintentos** (p. ej. N=2) y **escalar a intervención
   humana** cuando no hay progreso, con un mensaje explícito del estado.

---

## Comparativa rápida

| | Ornith 1.5 | Granite 4.2 |
|---|---|---|
| Aprobado para Aether | ✅ | ✅ |
| Vision nativa | ✅ (`vision`) | ❌ (usa `MODELO_VISION`) |
| Thinking / tools | ✅ / ✅ | ✅ / ✅ |
| Autonomía sostenida en tareas largas | Sin limitación registrada | ⚠ Puede detenerse y requerir intervención |
| Requisito registrado (probe) | ≈12 GB+ VRAM o 32 GB RAM (tier HIGH) | 8.79B `Q6_K`, 7.2 GB en disco; sin requisito formal versionado en el repo |
| Instalador | Opción recomendada (default del probe HIGH/ULTRA) | Opción aprobada seleccionable |

## Cómo elegir

- **GPU con ~12 GB+ / RAM amplia** → Ornith 1.5 (`ornith-1.5:9b`): multimodal
  nativo y referencia del proyecto.
- **Prefieras Granite / hardware ajustado** → `hf.co/ibm-granite/granite-4.2-8b-GGUF:Q6_K`:
  aprobado; aceptá la nota de autonomía para tareas agénticas largas.
- Hardware muy limitado → el probe recomienda modelos compactos (tiers
  POTATO/LOW/MID); seguís pudiendo configurar **cualquier modelo custom de
  Ollama** vía `--model` o `config.local.json`.

El instalador (`installer/install-core.sh`) ofrece ambos modelos aprobados,
detecta los ya instalados y mantiene la opción de modelo personalizado.
