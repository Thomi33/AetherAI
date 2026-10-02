"""Configuración global de pytest para mi_proyecto_crew.

Agrega el directorio raíz al sys.path para que los módulos bajo `core/`
sean importables sin instalación del paquete.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

# Hermeticidad de la memoria central compartida: los tests NUNCA deben
# escribir en ~/.aether/memory/learning.json. Con esto queda desactivada
# por defecto en toda la suite; los tests que la necesitan la activan
# explícitamente (monkeypatch.setenv) apuntando AETHER_CENTRAL_MEMORY_PATH
# a un tmp_path (ver la fixture central_tmp de test_outcome_memory.py).
os.environ["AETHER_CENTRAL_MEMORY"] = "0"
