#!/usr/bin/env python
"""
AETHER - Entrypoint Principal (TUI moderna con Textual)

Ejecuta: python run.py [--workdir RUTA]
"""
import os
import sys
from pathlib import Path

# --workdir puede venir en argv: fijar AETHER_CWD ANTES de importar settings
# (settings.RUTA_TRABAJO es lazy, pero así también lo ven los hijos).
for _i, _a in enumerate(sys.argv):
    if _a == "--workdir" and _i + 1 < len(sys.argv):
        os.environ["AETHER_CWD"] = sys.argv[_i + 1]
        del sys.argv[_i:_i + 2]
        break
    if _a.startswith("--workdir="):
        os.environ["AETHER_CWD"] = _a.split("=", 1)[1]
        del sys.argv[_i]
        break

# Asegurar que el directorio raíz esté en sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# La memoria necesita su esquema antes de que la TUI cargue/cambie datos.
# Es idempotente y crea current.db/tablas en el AETHER_DATA_DIR configurado.
from core.memory.memory_manager import asegurar_esquema
asegurar_esquema()

# Importar y ejecutar la TUI moderna
from tui.app import run

if __name__ == "__main__":
    run(workdir=os.environ.get("AETHER_CWD"))
