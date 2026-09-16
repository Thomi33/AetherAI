"""Tests de la memoria central compartida de Aether (core.memory.central).

Cubre: aprendizaje, búsqueda, refuerzo/debilitamiento (forget != delete),
consolidación por desuso, corrección explícita, memoria de usuario sin
sobrescritura silenciosa, memoria de conversación y señales de personalidad.
"""
from datetime import date, timedelta

import pytest

from core.memory.central import CentralMemory


@pytest.fixture
def mem(tmp_path):
    return CentralMemory(str(tmp_path / "memory"))


# ── Aprendizaje ──────────────────────────────────────────────────────────

def test_learn_formato_simple_y_legible(mem):
    m = mem.learn("Salta con SPACE cuando hay un obstáculo",
                  importance=6, tags=["roblox", "movimiento"], runtime="roblox")
    assert m["id"] == "mem_000001"
    assert m["summary"].startswith("Salta")
    assert m["date"] == date.today().isoformat()
    assert 0.0 < m["strength"] <= 1.0
    assert m["weakened"] is False


def test_learn_rechaza_summary_vacio(mem):
    with pytest.raises(ValueError):
        mem.learn("   ")


def test_ids_se_crean_secuenciales(mem):
    a = mem.learn("uno")
    b = mem.learn("dos")
    assert a["id"] == "mem_000001"
    assert b["id"] == "mem_000002"


# ── Búsqueda y recuperación ─────────────────────────────────────────────

def test_search_encuentra_por_relevancia(mem):
    mem.learn("El vision model es ambiguo con poca luz", tags=["vision"])
    mem.learn("La tecla E interactúa con NPCs", tags=["roblox"])
    res = mem.search("descripción ambigua del vision model")
    assert len(res) == 1
    assert "vision" in res[0]["tags"]


def test_search_refuerza_lo_que_recupera(mem):
    m = mem.learn("dato reutilizable", importance=5)
    antes = mem.get(m["id"])["strength"]
    mem.search("dato reutilizable")
    despues = mem.get(m["id"])["strength"]
    assert despues > antes
    assert mem.get(m["id"])["uses"] == 1


def test_recall_refuerza_mas_que_search(mem):
    m = mem.learn("recuerdo por id", importance=5)
    antes = mem.get(m["id"])["strength"]
    mem.recall(m["id"])
    assert mem.get(m["id"])["strength"] > antes


def test_search_filtra_por_namespace_y_kind(mem):
    mem.learn("a", namespace="roblox:speed_run", kind="strategy")
    mem.learn("b", namespace="roblox:obby", kind="strategy")
    res = mem.search(namespace="roblox:speed_run")
    assert [r["summary"] for r in res] == ["a"]


# ── Debilitamiento: olvidar NO es borrar ────────────────────────────────

def test_forget_debilita_pero_conserva(mem):
    m = mem.learn("recuerdo a debilitar")
    assert mem.forget(m["id"]) is True
    # No aparece en búsquedas normales…
    assert mem.search("recuerdo a debilitar") == []
    # …pero sigue existiendo en disco y es recuperable explícitamente.
    debiles = mem.search("recuerdo a debilitar", include_weak=True)
    assert len(debiles) == 1
    assert debiles[0]["weakened"] is True
    assert mem.get(m["id"]) is not None


def test_recuerdo_debilitado_se_refuerza_al_reusarse(mem):
    m = mem.learn("recuerdo rescatable")
    mem.forget(m["id"])
    mem.recall(m["id"])
    mem.recall(m["id"])
    rescate = mem.get(m["id"])
    assert rescate["strength"] >= 0.15
    assert rescate["weakened"] is False
    assert mem.search("recuerdo rescatable") != []


def test_forget_hard_es_la_unica_forma_de_borrar(mem):
    m = mem.learn("borrado fisico")
    assert mem.forget(m["id"], hard=True) is True
    assert mem.get(m["id"]) is None


# ── Consolidación: decaimiento por desuso, nunca borrado ────────────────

def test_consolidate_debilita_recuerdos_viejos_sin_borrarlos(mem):
    m = mem.learn("recuerdo del pasado", importance=3)
    futuro = (date.today() + timedelta(days=365)).isoformat()
    stats = mem.consolidate(today=futuro)
    assert stats["total"] == 1
    despues = mem.get(m["id"])
    assert despues is not None                      # sigue existiendo
    assert despues["strength"] < 0.15               # muy debilitado
    assert despues["strength"] > 0.0                # pero nunca cero
    assert despues["weakened"] is True


def test_consolidate_no_toca_recuerdos_recientes(mem):
    m = mem.learn("recuerdo fresco")
    antes = mem.get(m["id"])["strength"]
    mem.consolidate()
    assert mem.get(m["id"])["strength"] == antes


# ── Corrección explícita ────────────────────────────────────────────────

def test_update_es_la_via_de_correccion(mem):
    m = mem.learn("la tecla E abre la puerta roja")
    assert mem.update(m["id"], summary="la tecla F abre la puerta roja") is True
    assert mem.get(m["id"])["summary"] == "la tecla F abre la puerta roja"
    assert mem.update("mem_999999", summary="x") is False


# ── Memoria del usuario ─────────────────────────────────────────────────

def test_user_set_no_sobrescribe_silenciosamente(mem):
    ok, _ = mem.user_set("editor_preferido", "neovim", runtime="terminal")
    assert ok is True
    ok, msg = mem.user_set("editor_preferido", "vscode", runtime="roblox")
    assert ok is False
    assert "conflicto" in msg
    assert mem.user_get("editor_preferido") == "neovim"


def test_user_update_corrige_con_historial(mem):
    mem.user_set("editor_preferido", "neovim")
    assert mem.user_update("editor_preferido", "vscode") is True
    assert mem.user_get("editor_preferido") == "vscode"


def test_user_forget_quita_del_perfil(mem):
    mem.user_set("nombre", "thomi")
    assert mem.user_forget("nombre") is True
    assert mem.user_get("nombre") is None
    assert mem.user_forget("inexistente") is False


# ── Memoria de conversación ─────────────────────────────────────────────

def test_conversacion_append_get_y_compact(mem):
    for i in range(10):
        mem.conversation_append("s1", "usuario", f"mensaje {i}", runtime="terminal")
    conv = mem.conversation_get("s1")
    assert len(conv["turns"]) == 10
    assert mem.conversation_compact("s1", "resumen de la charla", keep_last=3)
    conv = mem.conversation_get("s1")
    assert conv["summary"] == "resumen de la charla"
    assert len(conv["turns"]) == 3
    assert mem.conversation_get("no_existe")["turns"] == []


# ── Personalidad acumulativa ────────────────────────────────────────────

def test_personality_signals_es_solo_lectura_y_acumulativa(mem):
    mem.learn("conviene re-observar antes de actuar si es ambiguo", importance=8)
    debil = mem.learn("señal débil", importance=1)
    mem.forget(debil["id"])
    signals = mem.personality_signals()
    assert signals == ["conviene re-observar antes de actuar si es ambiguo"]
    # Nada fue modificado por consultar las señales.
    assert mem.get(debil["id"])["weakened"] is True


# ── Persistencia compartida entre instancias (runtimes) ─────────────────

def test_otra_instancia_ve_lo_aprendido(tmp_path):
    root = str(tmp_path / "memory")
    a = CentralMemory(root)
    a.learn("aprendido por un runtime", runtime="roblox")
    b = CentralMemory(root)          # otro runtime, mismo almacenamiento
    assert b.search("aprendido por un runtime") != []
    assert b.stats()["learnings"] == 1



# ── QA extendido: EP/BVA sobre importance ───────────────────────────────

def test_importance_clamps_en_limites(mem):
    bajo = mem.learn("cero", importance=0)          # por debajo del mínimo
    alto = mem.learn("once", importance=11)         # por encima del máximo
    ok = mem.learn("cinco", importance=5)
    assert bajo["importance"] == 1
    assert alto["importance"] == 10
    assert ok["importance"] == 5
    # La fuerza inicial crece con la importancia (monótona en el rango).
    assert bajo["strength"] < ok["strength"] < alto["strength"]


def test_search_limit_cero_devuelve_vacio(mem):
    mem.learn("algo")
    assert mem.search("", limit=0) == []


# ── QA extendido: tabla de decisión de filtros de search ─────────────────

def test_search_filtros_combinados(mem):
    mem.learn("a", tags=["x"], namespace="n1", kind="k1", runtime="terminal")
    mem.learn("b", tags=["y"], namespace="n1", kind="k1", runtime="roblox")
    mem.learn("c", tags=["x"], namespace="n2", kind="k1", runtime="terminal")
    assert [r["summary"] for r in mem.search(tags=["x"])] == ["a", "c"]
    assert [r["summary"] for r in mem.search(runtime="roblox")] == ["b"]
    assert [r["summary"] for r in mem.search(namespace="n1", runtime="terminal")] == ["a"]
    assert mem.search(tags=["inexistente"]) == []


def test_search_con_acentos_y_unicode(mem):
    mem.learn("La descripción visual era ambigua y difícil")
    assert mem.search("descripción ambigua") != []


# ── QA extendido: boundary del umbral de accesibilidad ──────────────────

def test_recuerdo_en_el_limite_del_umbral_sigue_accesible(mem):
    m = mem.learn("recuerdo limítrofe", importance=9)
    central = mem._load_learning()["memories"][m["id"]]
    central["strength"] = 0.15                      # exactamente el umbral
    mem._save_learning()
    assert mem.search("recuerdo limítrofe") != []
    central["strength"] = 0.149                     # justo por debajo
    mem._save_learning()
    assert mem.search("recuerdo limítrofe") == []


# ── QA extendido: efectividad de decaimiento a distintas edades ─────────

def test_decaimiento_es_proporcional_a_la_antiguedad(mem):
    reciente = mem.learn("recuerdo casi nuevo", importance=5)
    central = mem._load_learning()["memories"][reciente["id"]]
    central["last_used"] = (date.today() - timedelta(days=7)).isoformat()
    viejo = mem.learn("recuerdo vetusto", importance=5)
    central = mem._load_learning()["memories"][viejo["id"]]
    central["last_used"] = (date.today() - timedelta(days=90)).isoformat()
    mem._save_learning()
    mem.consolidate()
    s_reciente = mem.get(reciente["id"])["strength"]
    s_viejo = mem.get(viejo["id"])["strength"]
    assert 0 < s_viejo < s_reciente


# ── QA extendido: concurrencia entre instancias (lost update) ───────────

def test_escrituras_intercaladas_no_pierden_datos(tmp_path):
    root = str(tmp_path / "memory")
    a = CentralMemory(root)                         # runtime A
    b = CentralMemory(root)                         # runtime B
    a.learn("de A")
    b.learn("de B")                                  # B escribe después de A
    a.learn("de A otra vez")                         # A escribe con caché viejo
    final = CentralMemory(root)
    assert final.stats()["learnings"] == 3


def test_hilos_concurrentes_no_rompen_el_archivo(tmp_path):
    import threading
    mem = CentralMemory(str(tmp_path / "memory"))
    def worker(i):
        for j in range(10):
            mem.learn(f"hilo {i} memoria {j}")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert mem.stats()["learnings"] == 40


# ── QA extendido: corrupción en disco nunca pierde datos en silencio ────

def test_json_corrupto_se_aparta_y_no_pisa_datos(tmp_path):
    root = tmp_path / "memory"
    mem = CentralMemory(str(root))
    mem.learn("dato valioso")
    path = root / "learning.json"
    original = path.read_text(encoding="utf-8")
    path.write_text("{ esto no es json valido !!!", encoding="utf-8")

    mem2 = CentralMemory(str(root))                  # otro runtime lo lee
    mem2.learn("dato nuevo")
    assert mem2.stats()["learnings"] == 1
    # El archivo corrupto NO se borró: quedó apartado para forense.
    apartados = list(root.glob("learning.json.corrupt-*"))
    assert len(apartados) == 1
    assert "dato valioso" in original                # el original sigue íntegro
    assert apartados[0].read_text(encoding="utf-8").startswith("{ esto no es")


# ── QA extendido: user_set idempotente con el mismo valor ───────────────

def test_user_set_mismo_valor_es_idempotente(mem):
    assert mem.user_set("editor", "nvim")[0] is True
    assert mem.user_set("editor", "nvim")[0] is True
    assert mem.user_get("editor") == "nvim"


# ── QA extendido: raíz por variable de entorno ──────────────────────────

def test_root_por_variable_de_entorno(tmp_path, monkeypatch):
    monkeypatch.setenv("AETHER_CENTRAL_MEMORY_PATH", str(tmp_path / "custom"))
    from core.memory.central.store import default_root
    assert default_root() == str(tmp_path / "custom")

