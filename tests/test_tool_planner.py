"""Tests del tool planner de Aether (core/agent/graph_nodes.py).

Técnicas aplicadas (ver qa-engineer/references/test-design-techniques.md):

- **Partición de equivalencias + valores límite**: detectores deterministas
  (_es_charla_simple, _es_confirmacion_aceptacion, _es_oferta_busqueda_web,
  _parece_correccion_usuario, _posible_referencia_anaforica).
- **Tabla de decisión**: routing anafórico launch vs shell (combina
  sintaxis-shell × contexto-de-lanzamiento).
- **Transición de estados**: _resolver_referencia_anaforica sobre distintas
  configuraciones de memoria conversacional.
- **Regresión**: falso positivo "sin duda..." en _confirmar_referencia_anaforica_llm
  y misrouting de `pip cache purge` a node_launch.

Las rutas que dependen del LLM se testean con monkeypatch sobre _llm_chat /
_confirmar_referencia_anaforica_llm: no se requiere Ollama corriendo.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pytest

import core.agent.graph_nodes as g


def _mem_con_turnos(turnos: list[dict]) -> dict:
    return {"conversacion": turnos}


def _state(orden: str, mem: dict | None = None) -> dict:
    return {
        "orden": orden,
        "mem": mem if mem is not None else _mem_con_turnos([]),
        "sesion_id": "",
    }


class TestCharlaSimple:
    @pytest.mark.parametrize("orden", [
        "hola", "HOLA", "¡Hola!", "buenas", "gracias", "ok",
        "buen día!!", "¿cómo estás?",
    ])
    def test_saludos_son_charla_simple(self, orden):
        assert g._es_charla_simple(orden) is True

    @pytest.mark.parametrize("orden", [
        "hola, listame los archivos",
        "gracias por ejecutar eso",
        "ok ejecutalo",
        "",
        "hola mundo de la programación",
    ])
    def test_no_son_charla_simple(self, orden):
        assert g._es_charla_simple(orden) is False


class TestConfirmacionAceptacion:
    @pytest.mark.parametrize("orden", [
        "dale", "sí", "si", "ok", "confirmado", "dale nomás", "dale si", "claro",
    ])
    def test_confirmaciones_simples(self, orden):
        assert g._es_confirmacion_aceptacion(g._normalizar(orden)) is True

    @pytest.mark.parametrize("orden", [
        "no dale",
        "no",
        "dale no",
        "seguro que no",
        "dale, pero antes borrá todo el directorio",
        "",
        "   ",
    ])
    def test_no_son_confirmacion(self, orden):
        assert g._es_confirmacion_aceptacion(g._normalizar(orden)) is False


class TestOfertaBusquedaWeb:
    @pytest.mark.parametrize("texto", [
        "¿Querés que lo busque?",
        "¿Querés que busque eso en internet?",
        "¿Te lo busco?",
        "¿Busco un precio actualizado?",
        "Puedo buscarlo si querés. ¿Te interesa?",
    ])
    def test_ofertas_reales(self, texto):
        assert g._es_oferta_busqueda_web(texto) is True

    @pytest.mark.parametrize("texto", [
        "queres que lo busque",        # sin signos de pregunta
        "Lo busco siempre en Google.",  # descriptiva, no oferta
        "Ya busqué eso ayer.",
        "",
        "   ",
    ])
    def test_no_son_ofertas(self, texto):
        assert g._es_oferta_busqueda_web(texto) is False


class TestCorreccionUsuario:
    @pytest.mark.parametrize("orden", [
        "me mentiste, no buscaste nada",
        "eso no es cierto",
        "te equivocaste, eso no pasó",
    ])
    def test_reclamos(self, orden):
        assert g._parece_correccion_usuario(orden.lower()) is True

    @pytest.mark.parametrize("orden", [
        "buscame la letra de esa canción",
        "ejecutá ls",
        "",
    ])
    def test_ordenes_normales_no_son_reclamo(self, orden):
        assert g._parece_correccion_usuario(orden.lower()) is False


class TestPosibleReferenciaAnaforica:
    @pytest.mark.parametrize("orden", [
        "ejecutalo", "dale", "hacelo", "guardalo", "eso último",
        "si, ejecutalo",
    ])
    def test_referencias(self, orden):
        assert g._posible_referencia_anaforica(orden.lower()) is True

    @pytest.mark.parametrize("orden", [
        "mostrame el archivo",
        "buscá el clima de mañana",
        "",
    ])
    def test_ordenes_autosuficientes(self, orden):
        assert g._posible_referencia_anaforica(orden.lower()) is False


class TestExtraerComandoDeTexto:
    def test_formato_protocolo_shell(self):
        assert g._extraer_comando_de_texto("probá [SHELL]ls -la[/SHELL]") == "ls -la"

    def test_bloque_markdown(self):
        assert g._extraer_comando_de_texto("```bash\npip cache purge\n```") == "pip cache purge"

    def test_backtick_simple(self):
        assert g._extraer_comando_de_texto("ejecutá `flatpak list` y listo") == "flatpak list"

    def test_texto_sin_comando(self):
        assert g._extraer_comando_de_texto("no hay comandos acá") is None

    @pytest.mark.parametrize("vacio", ["", None])
    def test_entrada_vacia(self, vacio):
        assert g._extraer_comando_de_texto(vacio) is None


class TestResolverReferenciaAnaforica:
    def test_turno_aether_con_comando_devuelve_payload(self):
        mem = _mem_con_turnos([
            {"rol": "usuario", "texto": "cómo limpio la caché de pip?"},
            {"rol": "aether", "texto": "Ejecutá `pip cache purge`.", "tema": "shell"},
        ])
        ref = g._resolver_referencia_anaforica(mem)
        assert ref is not None
        assert ref["command"] == "pip cache purge"
        assert ref["tema"] == "shell"
        assert ref["previo_usuario"] == "cómo limpio la caché de pip?"

    def test_turno_aether_sin_comando_devuelve_texto_sin_command(self):
        mem = _mem_con_turnos([
            {"rol": "usuario", "texto": "cuánto está el dólar?"},
            {"rol": "aether", "texto": "Está $1200 aprox.", "tema": "web"},
        ])
        ref = g._resolver_referencia_anaforica(mem)
        assert ref is not None
        assert ref["command"] is None
        assert "1200" in ref["last_texto"]

    def test_turnos_de_otra_sesion_se_excluyen(self):
        mem = _mem_con_turnos([
            {"rol": "aether", "texto": "[SHELL]rm -rf /tmp/x[/SHELL]", "sesion_id": "vieja"},
        ])
        ref = g._resolver_referencia_anaforica(mem, sesion_id="nueva")
        assert ref is None

    def test_sin_turnos_devuelve_none(self):
        assert g._resolver_referencia_anaforica(_mem_con_turnos([])) is None
        assert g._resolver_referencia_anaforica({}) is None

    def test_ignora_turnos_del_usuario_al_buscar_payload(self):
        mem = _mem_con_turnos([
            {"rol": "aether", "texto": "Acá va: `ls -la`", "tema": "shell"},
            {"rol": "usuario", "texto": "esperá, todavía no"},
        ])
        ref = g._resolver_referencia_anaforica(mem)
        assert ref is not None
        assert ref["command"] == "ls -la"


class TestConfirmarReferenciaAnaforicaLlm:
    def _confirmar(self, monkeypatch, respuesta_llm: str) -> bool:
        monkeypatch.setattr(g, "_llm_chat", lambda **kw: respuesta_llm)
        return g._confirmar_referencia_anaforica_llm("ejecutalo")

    def test_si_simple(self, monkeypatch):
        assert self._confirmar(monkeypatch, "si") is True

    def test_si_con_explicacion(self, monkeypatch):
        assert self._confirmar(monkeypatch, "sí, depende del turno anterior") is True

    def test_no(self, monkeypatch):
        assert self._confirmar(monkeypatch, "no") is False

    def test_regresion_sin_duda_no_es_afirmacion(self, monkeypatch):
        """'sin duda es autosuficiente' empezaba con 'si' y antes se leía
        como afirmación — la orden se desviaba a resolución anafórica."""
        assert self._confirmar(monkeypatch, "sin duda, es autosuficiente") is False
        assert self._confirmar(monkeypatch, "sin contexto previo no se puede") is False

    def test_fallback_seguro_si_llm_falla(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("ollama caído")
        monkeypatch.setattr(g, "_llm_chat", _boom)
        assert g._confirmar_referencia_anaforica_llm("dale") is False

class TestNodePlannerRouting:
    """Tabla de decisión del routing anafórico: sintaxis-shell × contexto-launch.

    La confirmación LLM se mockea en True: estos tests ejercitan el routing
    POSTERIOR a la confirmación.
    """

    def _plan(self, monkeypatch, orden: str, mem: dict) -> dict:
        monkeypatch.setattr(g, "_confirmar_referencia_anaforica_llm", lambda o: True)
        return g.node_planner(_state(orden, mem))

    def test_charla_simple_rutea_a_text(self):
        upd = g.node_planner(_state("hola"))
        assert upd["plan_pasos"][0]["tool"] == "text"
        assert upd["plan_activo"] is True

    def test_reclamo_rutea_a_text_sin_tools(self):
        upd = g.node_planner(_state("me mentiste, no ejecutaste nada"))
        assert upd["plan_pasos"][0]["tool"] == "text"

    def test_comando_multitoken_va_a_shell_aunque_diga_ejecuta(self, monkeypatch):
        """REGRESIÓN: 'ejecutá `pip cache purge`' + 'dale' terminaba en
        node_launch, que buscaba una app llamada 'pip cache purge' en
        flatpak y fallaba. Debe ir a shell."""
        mem = _mem_con_turnos([
            {"rol": "usuario", "texto": "cómo limpio pip?"},
            {"rol": "aether", "texto": "Para limpiar la caché ejecutá `pip cache purge`", "tema": "shell"},
        ])
        upd = self._plan(monkeypatch, "dale", mem)
        paso = upd["plan_pasos"][0]
        assert paso["tool"] == "shell"
        assert paso["args"]["command"] == "pip cache purge"

    def test_app_de_un_token_con_contexto_launch_va_a_launch(self, monkeypatch):
        mem = _mem_con_turnos([
            {"rol": "usuario", "texto": "quiero jugar roblox"},
            {"rol": "aether", "texto": "Podés jugar abriendo `sober`", "tema": "launch"},
        ])
        upd = self._plan(monkeypatch, "dale", mem)
        paso = upd["plan_pasos"][0]
        assert paso["tool"] == "launch"
        # node_launch recibe el nombre REAL, no la orden cruda "dale".
        assert "sober" in paso["instruccion"]

    def test_app_de_un_token_sin_contexto_launch_va_a_shell(self, monkeypatch):
        """Comando de un token sin contexto de lanzamiento (pwd, ls, df):
        shell es el default seguro (ejecuta apps igual)."""
        mem = _mem_con_turnos([
            {"rol": "usuario", "texto": "en qué directorio estoy?"},
            {"rol": "aether", "texto": "Fijate con `pwd`", "tema": "shell"},
        ])
        upd = self._plan(monkeypatch, "ejecutalo", mem)
        assert upd["plan_pasos"][0]["tool"] == "shell"
        assert upd["plan_pasos"][0]["args"]["command"] == "pwd"

    def test_comando_con_pipe_va_a_shell(self, monkeypatch):
        mem = _mem_con_turnos([
            {"rol": "aether", "texto": "Probá `cat log.txt | grep error`", "tema": "shell"},
        ])
        upd = self._plan(monkeypatch, "dale", mem)
        assert upd["plan_pasos"][0]["tool"] == "shell"

    def test_anaforica_sin_antecedente_pide_aclaracion(self, monkeypatch):
        upd = self._plan(monkeypatch, "ejecutalo", _mem_con_turnos([]))
        assert upd["done"] is True
        assert upd["plan_activo"] is False
        assert upd["final_response"]

    def test_confirmacion_de_oferta_busqueda_rutea_a_web(self, monkeypatch):
        mem = _mem_con_turnos([
            {"rol": "usuario", "texto": "cuánto está el dólar blue?"},
            {"rol": "aether", "texto": "No tengo datos en tiempo real. ¿Querés que lo busque?", "tema": "text"},
        ])
        upd = self._plan(monkeypatch, "dale", mem)
        paso = upd["plan_pasos"][0]
        assert paso["tool"] == "web"
        # El prefill incluye el pedido original del usuario para la query.
        assert "dólar" in upd["plan_resultados"][0]


class TestConstruirOrdenPaso:
    def test_inyecta_args_estructurados(self):
        orden = g._construir_orden_paso("buscá info", {"query": "clima rivera"}, [])
        assert "buscá info" in orden
        assert "[query] clima rivera" in orden

    def test_no_duplica_arg_ya_presente_en_instruccion(self):
        orden = g._construir_orden_paso("ejecutá ls -la", {"command": "ls -la"}, [])
        assert orden.count("ls -la") == 1

    def test_incluye_contexto_de_resultados_previos(self):
        orden = g._construir_orden_paso("guardalo", {}, ["resultado del paso 1"])
        assert "[CONTEXTO DE PASOS PREVIOS]" in orden
        assert "resultado del paso 1" in orden

    def test_args_no_dict_no_rompe(self):
        orden = g._construir_orden_paso("hacé algo", None, [])
        assert orden == "hacé algo"


class TestExtraerResultadoPaso:
    def test_extrae_resultado_no_vacio(self):
        salida = {"shell_output": "", "final_response": "todo ok"}
        assert g._extraer_resultado_paso(salida, "text")

    def test_salida_no_dict_devuelve_vacio(self):
        assert g._extraer_resultado_paso(None, "shell") == ""
        assert g._extraer_resultado_paso("texto crudo", "shell") == ""

