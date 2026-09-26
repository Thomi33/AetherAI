"""Tests del semantic router OPT-IN de Aether.

Cubre (ver core/agent/semantic_router.py y el hook en
core/agent/graph_nodes.py::_intentar_ruta_semantica):

1. Selección correcta de tools.
2. Cálculo del margin (top1 − top2).
3. Fallback cuando margin < threshold (y umbral configurable).
4. Ollama no disponible.
5. embeddinggemma (modelo) no disponible.
6. Modelo incorrecto / respuesta inválida de embeddings.
7. Compatibilidad con el planner actual (modo default → agent loop).
8. El comportamiento por defecto no cambia (config sin la clave nueva).

Además: prototipos + cache, métricas opcionales (AETHER_BENCH_INSTRUMENT),
override por env de sesión y validadores del ConfigManager.

Técnica: monkeypatch de la capa Ollama (`sr._ollama_embed`) y del config
manager (`sr.get_config_manager`) — no se requiere Ollama corriendo, no se
toca config.local.json y ningún test depende de /mnt/nvme.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

import ollama

import core.agent.semantic_router as sr
import core.agent.graph_nodes as g
from core.config.config_manager import ConfigManager


# ══════════════════════════════════════════════════════════════════════
# Helpers: vectores sintéticos y entorno fake
# ══════════════════════════════════════════════════════════════════════

DIM = 16
ORDEN_LAUNCH = "abrime el notepad"
ORDEN_FS_LIST = "listame los archivos de la carpeta descargas"
ORDEN_DUDOSA = "revisame el informe de ventas del trimestre"


def _vec_tool(tool: str) -> list[float]:
    """Vector one-hot determinista para la tool (dim=DIM)."""
    v = [0.0] * DIM
    v[sr.TOOL_LABELS.index(tool)] = 1.0
    return v


def _mezcla(tool_a: str, tool_b: str, peso_a: float = 0.5) -> list[float]:
    a, b = _vec_tool(tool_a), _vec_tool(tool_b)
    return [peso_a * x + (1.0 - peso_a) * y for x, y in zip(a, b)]


class _FakeConfig:
    """ConfigManager mínimo: solo .get(key, default)."""

    def __init__(self, datos: dict):
        self._datos = dict(datos)

    def get(self, key, default=None):
        return self._datos.get(key, default)


def _montar_entorno(
    monkeypatch,
    tmp_path,
    datos_config: dict | None = None,
    mapa_consultas: dict | None = None,
    error_embed: Exception | None = None,
    respuesta: str | None = None,
    contador: list | None = None,
):
    """Prepara el módulo semantic_router para un test aislado.

    - Ejemplos sintéticos (uno por tool, "ejemplo <tool>") en tmp_path.
    - Cache de prototipos en tmp_path (nunca ~/.aether).
    - Config fake inyectada en el módulo.
    - Capa Ollama reemplazada por un embed falso determinista.
    """
    ejemplos_path = tmp_path / "ejemplos.jsonl"
    with open(ejemplos_path, "w", encoding="utf-8") as fh:
        for tool in sr.TOOL_LABELS:
            fh.write(json.dumps({"orden": f"ejemplo {tool}", "tool": tool}) + "\n")

    monkeypatch.setattr(sr, "EJEMPLOS_PATH", ejemplos_path)
    monkeypatch.setattr(sr, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv(sr.ENV_OVERRIDE, raising=False)
    sr.reset_semantic_router()

    cfg = {
        "PLANNER_ROUTER": "semantic",
        "PLANNER_ROUTER_SEMANTIC_MODEL": "embeddinggemma:latest",
        "PLANNER_ROUTER_SEMANTIC_MARGIN": 0.03,
    }
    cfg.update(datos_config or {})
    monkeypatch.setattr(sr, "get_config_manager", lambda: _FakeConfig(cfg))

    mapa = dict(mapa_consultas or {})

    def _embed_fake(model, textos):
        if error_embed is not None:
            raise error_embed
        if contador is not None:
            contador.append(list(textos))
        if respuesta == "vacia":
            return []
        if respuesta == "corta":
            return [_vec_tool("text") for _ in textos[:-1]]
        out = []
        for texto in textos:
            if texto.startswith("ejemplo "):
                out.append(_vec_tool(texto.split(" ", 1)[1]))
            elif texto in mapa:
                v = list(mapa[texto])
                if respuesta == "dim_mala":
                    v = v + [0.0] * DIM  # consulta con dim 2x → error en coseno
                out.append(v)
            else:
                out.append(_vec_tool("text"))
        return out

    monkeypatch.setattr(sr, "_ollama_embed", _embed_fake)


def _state(orden: str) -> dict:
    """Estado mínimo para node_planner (mismo patrón que test_tool_planner)."""
    return {"orden": orden, "mem": {"conversacion": []}, "sesion_id": ""}


def _es_agent_loop(update: dict) -> bool:
    """node_planner en su camino actual activa el agent loop."""
    return (
        update.get("agent_activo") is True
        and update.get("plan_activo") is False
        and update.get("plan_pasos") == []
    )


# ══════════════════════════════════════════════════════════════════════
# 1) Selección correcta de tools + 2) cálculo del margin (unidad)
# ══════════════════════════════════════════════════════════════════════

class TestSeleccionYMargin:
    def test_selecciona_la_tool_correcta(self, monkeypatch, tmp_path):
        """La consulta cercana al prototipo de launch rutea a launch."""
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        res = sr.ruta_semantica(ORDEN_LAUNCH)
        assert res is not None
        assert res.tool == "launch"
        assert res.model == "embeddinggemma:latest"
        assert res.direct is True
        assert res.top_score == pytest.approx(1.0)
        assert res.second_score == pytest.approx(0.0)

    def test_calculo_de_margin(self, monkeypatch, tmp_path):
        """margin = coseno(top1) − coseno(top2), calculado con vectores
        mezcla 0.6/0.4 → (0.6 − 0.4) / sqrt(0.6² + 0.4²)."""
        mezcla = _mezcla("launch", "web", 0.6)
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_DUDOSA: mezcla})
        router = sr._get_router()
        res = router.route(ORDEN_DUDOSA, margin_threshold=0.03)
        esperado = (0.6 - 0.4) / (0.6 ** 2 + 0.4 ** 2) ** 0.5
        assert res.tool == "launch"
        assert res.second_tool == "web"
        assert res.margin == pytest.approx(esperado, abs=1e-9)
        assert res.margin == pytest.approx(res.top_score - res.second_score, abs=1e-12)

    def test_empate_deterministico(self, monkeypatch, tmp_path):
        """Empate exacto → margin 0 y desempate alfabético (launch < web)."""
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_DUDOSA: _mezcla("launch", "web", 0.5)})
        router = sr._get_router()
        res = router.route(ORDEN_DUDOSA, margin_threshold=0.03)
        assert res.margin == pytest.approx(0.0, abs=1e-12)
        assert res.tool == "launch" and res.second_tool == "web"
        assert res.ranked_tools[0] == "launch"


# ══════════════════════════════════════════════════════════════════════
# 3) Fallback por margin bajo (+ umbral configurable)
# ══════════════════════════════════════════════════════════════════════

class TestFallbackMargin:
    def test_margin_bajo_hace_fallback(self, monkeypatch, tmp_path):
        """Mezcla 50/50 → margin 0 < 0.03 → None (jamás una predicción dudosa)."""
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_DUDOSA: _mezcla("launch", "web", 0.5)})
        assert sr.ruta_semantica(ORDEN_DUDOSA) is None

    def test_margin_exactamente_en_umbral_rutea(self, monkeypatch, tmp_path):
        """margin == umbral (>=) rutea; la condición es '<' estricto."""
        margen_umbral = (0.6 - 0.4) / (0.6 ** 2 + 0.4 ** 2) ** 0.5  # ≈ 0.2774
        _montar_entorno(monkeypatch, tmp_path,
                        datos_config={"PLANNER_ROUTER_SEMANTIC_MARGIN": margen_umbral},
                        mapa_consultas={ORDEN_DUDOSA: _mezcla("launch", "web", 0.6)})
        res = sr.ruta_semantica(ORDEN_DUDOSA)
        assert res is not None and res.tool == "launch"

    def test_umbral_es_configurable(self, monkeypatch, tmp_path):
        """Misma consulta: con umbral 0.03 rutea; con 0.9 hace fallback."""
        mezcla = _mezcla("launch", "web", 0.6)  # margin ≈ 0.2774
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_DUDOSA: mezcla})
        assert sr.ruta_semantica(ORDEN_DUDOSA) is not None

        _montar_entorno(monkeypatch, tmp_path,
                        datos_config={"PLANNER_ROUTER_SEMANTIC_MARGIN": 0.9},
                        mapa_consultas={ORDEN_DUDOSA: mezcla})
        assert sr.ruta_semantica(ORDEN_DUDOSA) is None

    def test_margin_invalido_cae_en_default(self, monkeypatch, tmp_path):
        """Valor de config ilegible → usa el default 0.03 (no rompe)."""
        mezcla = _mezcla("launch", "web", 0.6)
        _montar_entorno(monkeypatch, tmp_path,
                        datos_config={"PLANNER_ROUTER_SEMANTIC_MARGIN": "alto"},
                        mapa_consultas={ORDEN_DUDOSA: mezcla})
        res = sr.ruta_semantica(ORDEN_DUDOSA)
        assert res is not None
        assert res.margin_threshold == pytest.approx(sr.DEFAULT_MARGIN)


# ══════════════════════════════════════════════════════════════════════
# Tools fs_*: requieren args estructurados → fallback (no error handler)
# ══════════════════════════════════════════════════════════════════════

class TestToolsEstructuradas:
    def test_fs_list_confiable_NO_rutea_directo(self, monkeypatch, tmp_path):
        """fs_list con margin alto igualmente NO se ejecuta directo: su nodo
        requiere 'path' → fallback al agent loop que la invoca con args."""
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_FS_LIST: _vec_tool("fs_list")})
        assert sr.ruta_semantica(ORDEN_FS_LIST) is None

    def test_todas_las_fs_quedan_excluidas(self):
        assert sr.HERRAMIENTAS_ENRUTABLES_DIRECTO == frozenset({
            "text", "web", "shell", "launch", "vision", "computer_use",
        })
        for tool in ("fs_write", "fs_read", "fs_mkdir", "fs_list"):
            assert tool not in sr.HERRAMIENTAS_ENRUTABLES_DIRECTO


# ══════════════════════════════════════════════════════════════════════
# 4/5/6) Fallos de Ollama / modelo / respuesta inválida → fallback limpio
# ══════════════════════════════════════════════════════════════════════

class TestFallosOllama:
    def test_ollama_no_disponible(self, monkeypatch, tmp_path):
        """Ollama caído (ConnectionError) → None, sin excepción al planner."""
        _montar_entorno(monkeypatch, tmp_path,
                        error_embed=ConnectionError("connection refused"))
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None

    def test_modelo_no_disponible(self, monkeypatch, tmp_path):
        """embeddinggemma no instalado (ollama.ResponseError 404) → None."""
        _montar_entorno(monkeypatch, tmp_path,
                        error_embed=ollama.ResponseError("model not found", 404))
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None

    def test_respuesta_vacia(self, monkeypatch, tmp_path):
        """Respuesta sin embeddings → inválida → None."""
        _montar_entorno(monkeypatch, tmp_path, respuesta="vacia")
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None

    def test_respuesta_incompleta(self, monkeypatch, tmp_path):
        """Menos embeddings que textos → inválida → None."""
        _montar_entorno(monkeypatch, tmp_path, respuesta="corta")
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None

    def test_consulta_con_dimension_inconsistente(self, monkeypatch, tmp_path):
        """Prototipos de 16 dims vs consulta de 32 → error en coseno → None."""
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")},
                        respuesta="dim_mala")
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None

    def test_error_en_runtime_degra_a_fallback_sin_lanzar(self, monkeypatch, tmp_path):
        """Cualquier otra excepción de la capa de embeddings → None."""
        _montar_entorno(monkeypatch, tmp_path,
                        error_embed=RuntimeError("boom inesperado"))
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None


# ══════════════════════════════════════════════════════════════════════
# 7/8) Modo default: comportamiento por defecto INTACTO
# ══════════════════════════════════════════════════════════════════════

class TestModoDefault:
    def test_default_no_toca_ollama(self, monkeypatch, tmp_path):
        """PLANNER_ROUTER=default → ni un solo embed; costo cero."""
        contador: list = []
        _montar_entorno(monkeypatch, tmp_path,
                        datos_config={"PLANNER_ROUTER": "default"},
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")},
                        contador=contador)
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None
        assert contador == []  # nunca llamó a Ollama

    def test_config_vieja_sin_la_clave_comporta_como_default(self, monkeypatch, tmp_path):
        """Una config sin PLANNER_ROUTER (instalaciones previas) → default."""
        contador: list = []
        _montar_entorno(monkeypatch, tmp_path,
                        datos_config={"PLANNER_ROUTER": None},
                        contador=contador)
        # quitar la clave por completo (no valor None)
        monkeypatch.setattr(
            sr, "get_config_manager",
            lambda: _FakeConfig({}),  # config legacy: sin claves nuevas
        )
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None
        assert contador == []

    def test_valor_invalido_en_config_cae_en_default(self, monkeypatch, tmp_path):
        """PLANNER_ROUTER="otro cosa" → tratado como default (nunca crashea)."""
        _montar_entorno(monkeypatch, tmp_path,
                        datos_config={"PLANNER_ROUTER": "neuronal"},
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None

    def test_env_de_sesion_gana_a_la_config(self, monkeypatch, tmp_path):
        """AETHER_PLANNER_ROUTER=semantic activa aunque la config diga default."""
        _montar_entorno(monkeypatch, tmp_path,
                        datos_config={"PLANNER_ROUTER": "default"},
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        monkeypatch.setenv(sr.ENV_OVERRIDE, "semantic")
        res = sr.ruta_semantica(ORDEN_LAUNCH)
        assert res is not None and res.tool == "launch"

    def test_env_de_sesion_default_desactiva(self, monkeypatch, tmp_path):
        """AETHER_PLANNER_ROUTER=default desactiva aunque la config diga semantic."""
        contador: list = []
        _montar_entorno(monkeypatch, tmp_path, contador=contador,
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        monkeypatch.setenv(sr.ENV_OVERRIDE, "default")
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None
        assert contador == []

    def test_env_invalido_se_ignora(self, monkeypatch, tmp_path):
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        monkeypatch.setenv(sr.ENV_OVERRIDE, "banana")
        assert sr.resolver_modo_router() == "default"
        assert sr.ruta_semantica(ORDEN_LAUNCH) is None


# ══════════════════════════════════════════════════════════════════════
# Prototipos + cache + métricas opcionales
# ══════════════════════════════════════════════════════════════════════

class TestPrototiposYCache:
    def test_prototipos_se_construyen_y_cachean(self, monkeypatch, tmp_path):
        """1er uso: embebe ejemplos + consulta. 2do uso (otro singleton):
        carga de cache → solo embebe la consulta."""
        contador: list = []
        _montar_entorno(monkeypatch, tmp_path, contador=contador,
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        assert sr.ruta_semantica(ORDEN_LAUNCH) is not None
        assert len(contador) == 2  # batch de ejemplos + consulta

        sr.reset_semantic_router()
        contador.clear()
        assert sr.ruta_semantica(ORDEN_LAUNCH) is not None
        assert len(contador) == 1  # solo la consulta: prototipos desde cache

        cache = tmp_path / "cache" / "prototypes__embeddinggemma_latest.json"
        assert cache.exists()
        data = json.loads(cache.read_text(encoding="utf-8"))
        assert data["model"] == "embeddinggemma:latest"
        assert set(data["prototypes"].keys()) == set(sr.TOOL_LABELS)

    def test_cache_corrupta_se_recalcula(self, monkeypatch, tmp_path):
        """Cache con JSON roto → recompute transparente; sigue routenado."""
        contador: list = []
        _montar_entorno(monkeypatch, tmp_path, contador=contador,
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        assert sr.ruta_semantica(ORDEN_LAUNCH) is not None

        cache = tmp_path / "cache" / "prototypes__embeddinggemma_latest.json"
        cache.write_text("{ json corrupto", encoding="utf-8")
        sr.reset_semantic_router()
        contador.clear()
        res = sr.ruta_semantica(ORDEN_LAUNCH)
        assert res is not None and res.tool == "launch"
        assert len(contador) == 2  # reconstruyó prototipos

    def test_cambio_de_modelo_reconstruye(self, monkeypatch, tmp_path):
        """Si cambia PLANNER_ROUTER_SEMANTIC_MODEL, se recalculan prototipos."""
        contador: list = []
        _montar_entorno(monkeypatch, tmp_path, contador=contador,
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        assert sr.ruta_semantica(ORDEN_LAUNCH) is not None

        _montar_entorno(monkeypatch, tmp_path, contador=contador,
                        datos_config={"PLANNER_ROUTER_SEMANTIC_MODEL": "nomic-embed-text:latest"},
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        contador.clear()
        res = sr.ruta_semantica(ORDEN_LAUNCH)
        assert res is not None and res.model == "nomic-embed-text:latest"
        assert len(contador) == 2  # prototipos nuevos para el modelo nuevo


class TestMetricasOpcionales:
    def test_metricas_se_registran_si_instrument_encendido(self, monkeypatch, tmp_path):
        """AETHER_BENCH_INSTRUMENT=1 → JSONL con scores/margins/latencias."""
        log = tmp_path / "bench_metrics.jsonl"
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={
                            ORDEN_LAUNCH: _vec_tool("launch"),
                            ORDEN_DUDOSA: _mezcla("launch", "web", 0.5),
                        })
        monkeypatch.setattr(sr, "BENCH_INSTRUMENT", True)
        monkeypatch.setattr(sr, "BENCH_LOG_PATH", log)

        assert sr.ruta_semantica(ORDEN_LAUNCH) is not None      # direct
        assert sr.ruta_semantica(ORDEN_DUDOSA) is None          # fallback

        lineas = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
        assert len(lineas) == 2
        directo, fallback = lineas

        assert directo["mode"] == "semantic-router"
        assert directo["outcome"] == "direct"
        assert directo["fallback_reason"] is None
        assert directo["selected_tool"] == "launch"
        assert directo["top_score"] == pytest.approx(1.0)
        assert directo["second_score"] == pytest.approx(0.0)
        assert directo["margin"] == pytest.approx(1.0)
        assert directo["embedding_latency_ms"] >= 0.0
        assert directo["routing_latency_ms"] >= 0.0

        assert fallback["outcome"] == "fallback"
        assert fallback["fallback_reason"] == "low_margin"
        assert fallback["margin_threshold"] == pytest.approx(0.03)

    def test_sin_instrument_no_genera_archivo(self, monkeypatch, tmp_path):
        """Modo normal: cero ruido (ni archivo de métricas)."""
        log = tmp_path / "bench_metrics.jsonl"
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        monkeypatch.setattr(sr, "BENCH_INSTRUMENT", False)
        monkeypatch.setattr(sr, "BENCH_LOG_PATH", log)
        assert sr.ruta_semantica(ORDEN_LAUNCH) is not None
        assert not log.exists()


class TestValidadoresConfig:
    def test_claves_nuevas_validan_correctamente(self, tmp_path):
        base = tmp_path / "config.json"
        base.write_text(json.dumps({}), encoding="utf-8")
        manager = ConfigManager(base)

        assert manager.set("PLANNER_ROUTER", "semantic") is True
        assert manager.set("PLANNER_ROUTER", "default") is True
        assert manager.set("PLANNER_ROUTER", "neuronal") is False

        assert manager.set("PLANNER_ROUTER_SEMANTIC_MODEL", "embeddinggemma:latest") is True

        assert manager.set("PLANNER_ROUTER_SEMANTIC_MARGIN", 0.05) is True
        assert manager.set("PLANNER_ROUTER_SEMANTIC_MARGIN", 1.5) is False
        assert manager.set("PLANNER_ROUTER_SEMANTIC_MARGIN", -0.1) is False


# ══════════════════════════════════════════════════════════════════════
# Integración con node_planner (el planner actual y su compatibilidad)
# ══════════════════════════════════════════════════════════════════════

class TestIntegracionPlanner:
    def test_planner_default_mantiene_comportamiento(self, monkeypatch, tmp_path):
        """PLANNER_ROUTER=default → node_planner va al agent loop igual que
        antes de esta integración (ningún cambio observable)."""
        _montar_entorno(monkeypatch, tmp_path,
                        datos_config={"PLANNER_ROUTER": "default"},
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        update = g.node_planner(_state(ORDEN_LAUNCH))
        assert _es_agent_loop(update)

    def test_planner_semantic_confiable_rutea_directo(self, monkeypatch, tmp_path):
        """PLANNER_ROUTER=semantic + margin alto → plan de 1 paso a la tool."""
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_LAUNCH: _vec_tool("launch")})
        update = g.node_planner(_state(ORDEN_LAUNCH))

        assert update["plan_activo"] is True
        assert update.get("agent_activo") is not True
        assert update["plan_index"] == 0
        assert update["plan_pasos"] == [{
            "tool": "launch", "instruccion": ORDEN_LAUNCH, "args": {},
        }]
        assert "mem" in update  # la memoria viaja igual que en los fast-paths

    def test_planner_semantic_dudoso_fallback_al_planner_actual(self, monkeypatch, tmp_path):
        """margin 0 → el planner cae al agent loop (fallback automático)."""
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_DUDOSA: _mezcla("launch", "web", 0.5)})
        update = g.node_planner(_state(ORDEN_DUDOSA))
        assert _es_agent_loop(update)

    def test_planner_semantic_fs_fallback_al_planner_actual(self, monkeypatch, tmp_path):
        """Predicción fs_list (confiable pero no enrutable directo) → agent
        loop, que la invoca con args estructurados (no error handler)."""
        _montar_entorno(monkeypatch, tmp_path,
                        mapa_consultas={ORDEN_FS_LIST: _vec_tool("fs_list")})
        update = g.node_planner(_state(ORDEN_FS_LIST))
        assert _es_agent_loop(update)

    def test_planner_semantic_ollama_caido_fallback_limpio(self, monkeypatch, tmp_path):
        """Ollama caído con semantic activo → el turno NO se rompe: agent loop."""
        _montar_entorno(monkeypatch, tmp_path,
                        error_embed=ConnectionError("connection refused"))
        update = g.node_planner(_state(ORDEN_LAUNCH))
        assert _es_agent_loop(update)

    def test_planner_semantic_modelo_caido_fallback_limpio(self, monkeypatch, tmp_path):
        """Modelo de embeddings inexistente → fallback limpio al agent loop."""
        _montar_entorno(monkeypatch, tmp_path,
                        error_embed=ollama.ResponseError("model not found", 404))
        update = g.node_planner(_state(ORDEN_LAUNCH))
        assert _es_agent_loop(update)

    def test_los_datos_empaquetados_solo_contienen_train(self):
        """El JSONL empaquetado es el train de v2: 394 líneas, 10 tools."""
        ruta = pathlib.Path(sr.EJEMPLOS_PATH)
        assert ruta.exists(), f"falta el archivo de ejemplos: {ruta}"
        with open(ruta, encoding="utf-8") as fh:
            lineas = [json.loads(l) for l in fh if l.strip()]
        assert len(lineas) == 394
        assert {r["tool"] for r in lineas} == set(sr.TOOL_LABELS)




