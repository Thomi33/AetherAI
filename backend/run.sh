#!/bin/bash
# Aether Backend — inicia la API con el venv del proyecto
set -e

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$BACKEND_DIR")"
cd "$PROJECT_ROOT"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
    echo "⚠️  .venv no encontrado, usando: $PYTHON"
fi

echo "🚀 Aether API: http://localhost:${API_PORT:-8000}  (docs: /docs)"
exec "$PYTHON" -m uvicorn backend.api.main:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}"
