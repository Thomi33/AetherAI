# Semantic routing — datos de prototipos

Fuente única de ejemplos para el router semántico OPT-IN de Aether
(`core/agent/semantic_router.py`).

## Contenido

- `router_examples_v2.jsonl` — **394 ejemplos**, copia byte a byte del split
  `train` del dataset v2 de Aether
  (SHA-256: `dc63169964a343b60e33bae8c92a4229b009d6e8aa784ab20d866b8c83f27196`).

## Reglas

1. **SOLO train.** Este archivo es exclusivamente el split de entrenamiento.
   NUNCA agregar acá ejemplos del split `test` de v2 (es holdout del
   benchmark `/mnt/nvme/data/semantic-router-bench/` y de Laya).
2. El runtime NO depende de ninguna ruta externa (`/mnt/nvme/...`): este
   archivo viaja con el repo y es la única fuente de ejemplos.
3. Provenance: los prototipos se calculan con estos ejemplos embebiéndolos
   con el modelo configurado (`PLANNER_ROUTER_SEMANTIC_MODEL`, default
   `embeddinggemma:latest`) la primera vez que se activa
   `PLANNER_ROUTER = semantic`; el resultado se cachea en
   `~/.aether/semantic_router/prototypes__<modelo>.json`.
4. Formato: JSONL con `orden` (texto) y `tool` (label), generado para el
   benchmark v2 (español rioplatense). Los 10 labels son las tools de Aether:
   text, web, shell, launch, vision, fs_write, fs_read, fs_mkdir, fs_list,
   computer_use.

## Benchmark de referencia

Configuración validada en `/mnt/nvme/data/semantic-router-bench/`
(evaluada sobre el test holdout de v2, 86 ejemplos):

- embeddinggemma + prototype (texto crudo): accuracy 87.21%, macro 87.64%,
  top-3 97.7% — vs Laya v2 (fine-tuning): 75.58% / 76.47%.
- Margen top1−top2 como señal de confianza: predicciones ambiguas
  (margin < 0.03) → 58.8% accuracy vs 94.2% en las confiables.
