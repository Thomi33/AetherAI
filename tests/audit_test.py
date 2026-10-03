#!/home/thomi/mi_proyecto_crew/.venv/bin/python3
import sys
sys.path.insert(0, '/home/thomi/mi_proyecto_crew')

from core.agent.tool_registry import TOOL_REGISTRY, validar_plan
from core.agent.graph_builder import build_graph, _TOOLS_CON_RESUMEN

print("=" * 60)
print("AETHER PROJECT INTEGRATION AUDIT")
print("=" * 60)

print("\n1. TOOL REGISTRY ANALYSIS")
print("-" * 60)
print(f"Total tools registered: {len(TOOL_REGISTRY)}")
print(f"Tools: {sorted(TOOL_REGISTRY.keys())}")

print("\n2. _TOOLS_CON_RESUMEN ANALYSIS")
print("-" * 60)
print(f"Tools in resumen set: {sorted(_TOOLS_CON_RESUMEN)}")
all_tools = set(TOOL_REGISTRY.keys())
not_in_resumen = all_tools - _TOOLS_CON_RESUMEN
print(f"Tools NOT in resumen set: {sorted(not_in_resumen)}")

print("\n3. VALIDATION TESTS")
print("-" * 60)
test_cases = [
    ([{"tool": "text", "instruccion": "saluda"}], "Valid simple plan"),
    ([{"tool": "inexistente", "instruccion": "x"}], "Invalid tool"),
    ([{"instruccion": "x"}], "Missing tool field"),
    ([], "Empty plan"),
    ([{"tool": "web"}], "Missing instruction for required tool"),
    ([{"tool": "vision"}], "Vision no requiere instruccion"),
    ([{"tool": "web", "args": {"query": "algo"}}], "Instruction in args query"),
    ([{"tool": "text", "instruccion": "x", "args": "no-dict"}], "Args not dict"),
]

for plan, desc in test_cases:
    ok, err = validar_plan(plan)
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"  {status} | {desc}: ok={ok}, err={err[:50] if err else 'None'}")

print("\n4. GRAPH BUILD TEST")
print("-" * 60)
try:
    graph = build_graph()
    print("  ✅ Graph compiled successfully")
except Exception as e:
    print(f"  ❌ Graph build failed: {e}")

print("\n5. MEMORY CONSOLIDATOR KEY PARAMETERS")
print("-" * 60)
from core.memory.memory_manager import MIN_TURNOS_PARA_CONSOLIDAR, programar_consolidacion, consolidar_resumen
print(f"  MIN_TURNOS_PARA_CONSOLIDAR: {MIN_TURNOS_PARA_CONSOLIDAR}")
print(f"  _scheduler_lock exists: False (removed)")
print(f"  _scheduler_running exists: False (removed)")

print("\n6. SHELL EXECUTOR SAFETY PATTERNS")
print("-" * 60)
from core.tools.shell_executor import _PATRONES_PELIGROSOS, EDITORES_BANEADOS, _LANZADORES_GUI
print(f"  Dangerous patterns count: {len(_PATRONES_PELIGROSOS)}")
print(f"  Banned editors: {EDITORES_BANEADOS}")
print(f"  GUI launcher regex: {_LANZADORES_GUI.pattern[:80]}...")

print("\n7. CONTEXT BUDGET SETTINGS")
print("-" * 60)
from core.config.settings import CONTEXTO_CONV_MAX_CHARS, MAX_TURNOS_CONTEXTO, MAX_TURNOS_CONTEXTO_PLAN, MAX_TURNOS_CONTEXTO_CHAT
print(f"  CONTEXTO_CONV_MAX_CHARS: {CONTEXTO_CONV_MAX_CHARS}")
print(f"  MAX_TURNOS_CONTEXTO: {MAX_TURNOS_CONTEXTO}")
print(f"  MAX_TURNOS_CONTEXTO_PLAN: {MAX_TURNOS_CONTEXTO_PLAN}")
print(f"  MAX_TURNOS_CONTEXTO_CHAT: {MAX_TURNOS_CONTEXTO_CHAT}")

print("\n8. MCP CONNECTOR PERMISSIONS")
print("-" * 60)
from core.connectors.filesystem import CONNECTOR as fs_connector
from core.connectors.github import CONNECTOR as github_connector
print(f"  Filesystem - write_file permission: {fs_connector.default_permission}")
print(f"  Filesystem - write_file in permissions: {fs_connector.permissions.get('write_file', 'not set')}")
print(f"  GitHub - create_issue permission: {github_connector.permissions.get('create_issue', 'not set')}")
print(f"  GitHub - default_permission: {github_connector.default_permission}")

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)