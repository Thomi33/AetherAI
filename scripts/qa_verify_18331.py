"""
QA HARNESS — verificación en vivo del fix del bug "4 pasos buscando lo
mismo" (búsquedas web repetidas en el agent loop).

NO es código de producción ni un test de la suite: es verificación
end-to-end del escenario reportado, con SOLO los bordes externos
fakeados:

  - _llm_chat_agente (modelo agente): scripted → emite tool calls web
    parafraseadas, como en el log real (el modelo insistía variando la
    instruccion).
  - buscar_web: fake → siempre los mismos resultados (IMPO primero),
    como en el log real.
  - leer_url: fake → contenido de portada que NO contiene lo buscado.

TODO lo demás es de PRODUCCIÓN y REAL: node_agent_loop, node_web,
_generar_query_busqueda y el condensador (_llm_chat → Ollama real,
MODELO configurado). Verifica en vivo:

  F-1  la instruccion del MODELO llega al condensador (no la orden
       original con la wake word)
  F-3  una URL ya leída no se relee: cada paso lee una URL NUEVA, y
       agotadas todas devuelve [SIN RESULTADOS NUEVOS]
  F-4  las repeticiones clasifican sin_resultados → tope duro → cierre
       forzado (no hay bucle infinito)
  F-5  las queries que salen están saneadas (sin wake word, sin
       fragmentos de marcador, ≥2 palabras) CON EL CONDENSADOR REAL

Uso: .venv/bin/python scripts/qa_verify_18331.py
(Salida completa del trace en stdout; sale con código 0/1.)
"""
import os
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# Sin tocar la memoria central real del Creador (igual que conftest.py)
os.environ["AETHER_CENTRAL_MEMORY"] = "0"

import core.agent.graph_nodes as gn                              # noqa: E402
from core.agent import tool_registry as tr                        # noqa: E402

ORDEN = "aether, que dice ley numero 18331 uruguay?"
URL_IMPO = "https://www.impo.com.uy/bases/leyes/18331-2008"
URL_IMPO_29 = "https://www.impo.com.uy/bases/leyes/18331-2008/29"

# Los 10 resultados del log real colapsados a los 2 relevantes (el
# buscador devolvía siempre lo mismo y node_web leía el primero).
RESULTADOS = (
    "[SEARXNG]\n"
    "- Título: Ley N° 18331 - IMPO\n"
    f"  URL: {URL_IMPO}\n"
    "  Resumen: Ley de Protección de Datos Personales, portada.\n"
    "- Título: Ley N° 18331 - IMPO (texto)\n"
    f"  URL: {URL_IMPO_29}\n"
    "  Resumen: Texto de la ley 18331, página 29.\n"
)

# Instrucciones del log real (4) + la insistencia típica (5ª).
INSTRUCCIONES = [
    "buscar el texto exacto del artículo 18331 del Código Penal uruguayo (ley 18331). "
    "Necesito el contenido completo del artículo.",
    "buscar el artículo 18331 específico en el texto de la ley. "
    "Necesito el número exacto del artículo y su contenido.",
    "buscar en el texto completo de la ley 18331 la frase 'derecho al olvido' "
    "o 'olvido' o 'eliminación de datos'.",
    "buscar el texto completo de la ley 18331 de Uruguay en formato PDF o texto plano.",
    "reintentar la búsqueda del texto de la ley 18331 con otras palabras clave.",
]


def main() -> int:
    busquedas: list = []
    lecturas: list = []
    condensador_io: list = []

    # ── Borde 1: modelo agente scripted (como el log) ────────────────────
    def fake_agente(messages, tools, min_predict=None):
        i = fake_agente.i
        fake_agente.i += 1
        if i < len(INSTRUCCIONES):
            return {"content": "", "tool_calls": [{"function": {
                "name": "web",
                "arguments": {"instruccion": INSTRUCCIONES[i]},
            }}]}
        return {"content": "listo, respondí con lo que tenía", "tool_calls": []}
    fake_agente.i = 0
    gn._llm_chat_agente = fake_agente
    tr.construir_tools_ollama = lambda: []

    # ── Borde 2: espía del condensador REAL (llama a Ollama de verdad) ───
    real_llm_chat = gn._llm_chat

    def spy_llm_chat(system=None, user=None, messages=None, **kw):
        condensador_io.append({"system": system or "", "user": user or ""})
        return real_llm_chat(system=system, user=user, messages=messages, **kw)
    gn._llm_chat = spy_llm_chat

    # ── Borde 3: buscador + lector fakeados (los resultados del log) ──────
    gn.buscar_web = SimpleNamespace(
        invoke=lambda q: busquedas.append(q) or RESULTADOS)

    def _leer(url):
        lecturas.append(url)
        portada = ("LEY 18.331 — PROTECCIÓN DE DATOS PERSONALES. "
                   "Artículo 1º.- La presente ley tiene por objeto la protección "
                   "de los datos personales... ")
        return f"[CONTENIDO DE {url}]\n" + (portada * 40)[:7000]
    gn.leer_url = SimpleNamespace(invoke=_leer)

    # ── Ejecución: TODO el flujo real de producción ───────────────────────
    print("═" * 72)
    print("VERIFICACIÓN EN VIVO — escenario del log, condensador REAL (Ollama)")
    print("═" * 72)
    state = {"orden": ORDEN, "mem": {}, "agent_messages": None,
             "agent_pasos_log": None}
    rondas = 0
    while not state.get("done") and rondas < 12:
        rondas += 1
        out = gn.node_agent_loop(state)
        state.update(out)

    pasos = state.get("agent_pasos_log") or []
    condensados = [io for io in condensador_io
                   if "generador de queries" in io["system"]]

    # ── Veredicto ─────────────────────────────────────────────────────────
    fallos = []

    def check(nombre, cond, detalle=""):
        print(f"  {'✅' if cond else '❌'} {nombre}"
              + (f"\n      └─ {detalle}" if detalle else ""))
        if not cond:
            fallos.append(nombre)

    print("\n─── Queries enviadas al buscador (condensador REAL) ───")
    for q in busquedas:
        print(f"  → '{q}'")
    print("─── URLs leídas ───")
    for u in lecturas:
        print(f"  → {u}")
    print("─── Input del condensador (F-1: instruccion del modelo, sin contexto) ───")
    for io in condensados:
        print(f"  → {io['user'].splitlines()[0][:90]}")

    print("\n─── Veredicto ───")
    check("F-1: el condensador condensa la instruccion del MODELO "
          "(sin la orden original ni wake word)",
          bool(condensados) and all(
              "aether" not in io["user"].lower() for io in condensados),
          f"{len(condensados)} llamadas al condensador")
    check("F-5: el condensador NO vio el [CONTEXTO DE PASOS PREVIOS]",
          all(gn._MARCADOR_CTX_PREVIO not in io["user"] for io in condensados))
    check("F-5: todas las queries saneadas (sin wake word, sin '[', ≥2 palabras)",
          all("aether" not in q.lower() and "[" not in q and len(q.split()) >= 2
              for q in busquedas))
    check("F-3: no se releyó ninguna URL (5 búsquedas → 2 lecturas)",
          lecturas == [URL_IMPO, URL_IMPO_29], f"lecturas: {lecturas}")
    check("F-4: repetidas clasifican sin_resultados y agotan el tope",
          [p.get("cls") for p in pasos] == ["ok", "ok", "sin_resultados",
                                            "sin_resultados", "sin_resultados"]
          and gn._fallos_de_tool(pasos, "web") == 3,
          f"cls: {[p.get('cls') for p in pasos]}")
    check("F-4: cierre forzado (done) — no hay bucle infinito",
          state.get("done") is True
          and "No pude conseguir resultados" in (state.get("final_response") or ""),
          f"rondas={rondas}, pasos={len(pasos)}")

    print("\n" + "═" * 72)
    if fallos:
        print(f"❌ VERIFICACIÓN FALLÓ: {fallos}")
        return 1
    print("✅ VERIFICACIÓN EN VIVO OK: 5 búsquedas insistidas → 2 lecturas "
          "únicas, queries saneadas y cierre forzado. El loop ya no se "
          "repite infinito.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
