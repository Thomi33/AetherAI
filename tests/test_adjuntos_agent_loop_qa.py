"""Tests QA adversariales: romper el agent loop con adjuntos (imagen, PDF,
video, documento, binario) y con las pseudo-llamadas exactas observadas en
los logs de producción.

Regresiones cubiertas:
  1. shell(instruccion='vision("Describe ... adjunta: <path>")') → vision real.
  2. shell(instruccion="vision -- Imagen 'x.jpg'") → vision real (no shell).
  3. shell(instruccion='{"instruccion": "Describe ..."}') → vision real, y el
     JSON jamás llega a zsh.
  4. [SHELL]{...}[/SHELL] (payload JSON) → extraer_comando_shell devuelve None.
  5. cat de un binario ya no lanza UnicodeDecodeError.
  6. vision(path=<pdf>) → error explícito SIN captura de pantalla.
  7. vision con path inexistente → error explícito SIN captura de pantalla.
  8. El system prompt del agent loop ya NO enseña el protocolo [SHELL].
"""
from __future__ import annotations

import subprocess

import pytest

from core.agent.graph_nodes import _rescatar_llamada_confundida
from core.parser.shell_parser import extraer_comando_shell


PATH_IMG_LOG = ("/home/thomi/Aether/adjuntos/"
                "20260916-213632_0_Screenshot_20260909_202804_Gallery.jpg")


# ── 1-3. Rescue de pseudo-llamadas anidadas en shell (logs reales) ───────────


class TestRescueVariantesReales:
    def test_formato_parentesis_exacto_del_log(self):
        tool, args = _rescatar_llamada_confundida(
            "shell", {"instruccion":
                      f'vision("Describe el contenido de la imagen adjunta: {PATH_IMG_LOG}")'})
        assert tool == "vision"
        assert PATH_IMG_LOG in args["instruccion"]

    def test_formato_dash_sin_parentesis(self):
        tool, args = _rescatar_llamada_confundida(
            "shell", {"instruccion":
                      "vision -- Imagen 'Screenshot_20260909_202944_Gallery.jpg'"})
        assert tool == "vision"
        assert "Screenshot_20260909_202944_Gallery.jpg" in args["instruccion"]

    def test_formato_dash_describe(self):
        tool, args = _rescatar_llamada_confundida(
            "shell", {"instruccion":
                      "vision -- Describe el contenido de la imagen 'x.jpg'"})
        assert tool == "vision"
        assert "Describe el contenido" in args["instruccion"]

    def test_payload_json_anidado_se_desempaqueta(self):
        tool, args = _rescatar_llamada_confundida(
            "shell", {"instruccion":
                      '{"instruccion": "vision(\\"Describe: /tmp/x.jpg\\")"}'})
        assert tool == "vision"

    def test_comando_shell_legitimo_no_se_rescata(self):
        for cmd in ("ls -la /tmp", "cat /etc/os-release",
                    "echo '{\"a\": 1}'", "grep foo archivo.txt"):
            tool, args = _rescatar_llamada_confundida(
                "shell", {"instruccion": cmd})
            assert tool == "shell", f"comando legítimo re-ruteado: {cmd}"
            assert args["instruccion"] == cmd

    def test_solo_shell_es_rescatable(self):
        tool, _ = _rescatar_llamada_confundida(
            "web", {"instruccion": 'vision("x")'})
        assert tool == "web"


# ── 4. Un payload JSON NUNCA se ejecuta como comando zsh ─────────────────────


class TestJsonNuncaSeEjecutaEnZsh:
    def test_payload_json_en_bloque_shell_devuelve_none(self):
        texto = ('[SHELL]\n{\n  "instruccion": "Describe el contenido de la '
                 f'imagen adjunta: {PATH_IMG_LOG}"\n}}\n[/SHELL]')
        assert extraer_comando_shell(texto) is None

    def test_payload_json_en_fence_bash_devuelve_none(self):
        texto = '```bash\n{"instruccion": "vision x"}\n```'
        assert extraer_comando_shell(texto) is None

    def test_comandos_reales_siguen_pasando(self):
        assert extraer_comando_shell("[SHELL] ls -la /tmp [/SHELL]") == "ls -la /tmp"
        assert extraer_comando_shell("[SHELL] file /tmp/x.jpg [/SHELL]") \
            == "file /tmp/x.jpg"
        # Llaves de zsh legítimas no son JSON → pasan.
        assert extraer_comando_shell("[SHELL] { ls; pwd; } [/SHELL]") is not None


# ── 5. cat de binario: sin UnicodeDecodeError, mensaje claro ─────────────────


class TestCatBinario:
    def test_salida_binaria_no_crachea(self, tmp_path):
        from core.tools.shell_executor import ejecutar_comando
        binario = tmp_path / "foto.jpg"
        binario.write_bytes(b"\xff\xd8\xff\xe0" + bytes(range(256)) * 4)
        salida, hubo_error = ejecutar_comando(f"cat {binario}")
        assert isinstance(salida, str)  # nunca excepción
        # cat de un binario sale OK (rc=0): el aviso truncado ayuda al modelo.
        assert "salida binaria truncada" in salida or len(salida) > 0

    def test_salida_de_texto_normal_intacta(self):
        from core.tools.shell_executor import ejecutar_comando
        salida, hubo_error = ejecutar_comando("echo hola-mundo")
        assert "hola-mundo" in salida
        assert hubo_error is False


# ── 6-7. node_vision con archivos que NO son imagen ─────────────────────────


class TestVisionConArchivosNoImagen:
    def test_path_pdf_no_captura_pantalla(self, tmp_path, monkeypatch):
        import core.agent.graph_nodes as gn

        def _no_screenshot(*a, **k):
            raise AssertionError("¡node_vision capturó la pantalla con un PDF!")

        monkeypatch.setattr(gn, "ver_pantalla", _no_screenshot)
        pdf = tmp_path / "informe.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        out = gn.node_vision({"orden": "describí el adjunto",
                              "_tool_args": {"path": str(pdf)}})
        assert "[ERROR]" in out["vision_result"]
        assert "no es una imagen" in out["vision_result"]

    def test_path_inexistente_no_captura_pantalla(self, monkeypatch):
        import core.agent.graph_nodes as gn

        def _no_screenshot(*a, **k):
            raise AssertionError("¡node_vision capturó la pantalla con path inválido!")

        monkeypatch.setattr(gn, "ver_pantalla", _no_screenshot)
        out = gn.node_vision({"orden": "describí la imagen",
                              "_tool_args": {"path": "/tmp/no_existe_12345.jpg"}})
        assert "[ERROR]" in out["vision_result"]
        assert "no existe" in out["vision_result"]

    def test_path_imagen_via_args_se_describe(self, tmp_path, monkeypatch):
        import core.agent.graph_nodes as gn
        import backend.core.attachments as att
        img = tmp_path / "foto.png"
        img.write_bytes(b"\x89PNG\r\n")
        llamadas = []

        def fake_describir(ruta, pregunta=None):
            llamadas.append(str(ruta))
            return "un gato"

        monkeypatch.setattr(att, "describir_imagen", fake_describir)
        out = gn.node_vision({"orden": "qué hay en la imagen?",
                              "_tool_args": {"path": str(img),
                                             "instruccion": "qué hay"}})
        assert llamadas == [str(img)]
        assert "gato" in out["vision_result"]

    def test_path_video_no_captura_pantalla(self, tmp_path, monkeypatch):
        import core.agent.graph_nodes as gn
        monkeypatch.setattr(gn, "ver_pantalla",
                            lambda *a, **k: pytest.fail("screenshot con video"))
        mp4 = tmp_path / "clip.mp4"
        mp4.write_bytes(b"\x00\x00\x00\x18ftyp")
        out = gn.node_vision({"orden": "mirá el video",
                              "_tool_args": {"path": str(mp4)}})
        assert "[ERROR]" in out["vision_result"]


# ── 8. System prompt del agent loop sin protocolo legado ─────────────────────


class TestPromptAgentLoop:
    def test_no_enseña_protocolo_shell_legado(self):
        from core.agent.prompts import construir_prompt_agent_loop
        prompt = construir_prompt_agent_loop("")
        # Sin instrucciones del formato de texto legacy (solo mencionado para PROHIBIR).
        assert "[SHELL] <comando" not in prompt
        assert "cat -n" not in prompt
        # Sí enseña el manejo de adjuntos.
        assert "[ADJUNTOS DEL USUARIO]" in prompt
        assert "vision" in prompt and "path" in prompt

    def test_agent_loop_usa_prompt_dedicado(self):
        import core.agent.graph_nodes as gn
        from core.agent import prompts
        assert hasattr(prompts, "construir_prompt_agent_loop")
        # El helper del nodo existe y devuelve el prompt dedicado.
        out = gn._system_prompt_agent_loop({}, None)
        assert "tool calling nativo" in out or "AGENTE" in out


# ── Reproducción end-to-end del log de producción ────────────────────────────


class TestAgentLoopNoEjecutaVisionEnShell:
    def test_vision_anidada_en_shell_se_re_rutea_y_no_se_ejecuta_comando(
            self, tmp_path, monkeypatch):
        import core.agent.graph_nodes as gn
        import core.agent.tool_registry as tr

        img = tmp_path / "foto.jpg"
        img.write_bytes(b"\xff\xd8\xff")
        img_path = str(img)
        llamadas = []

        # El modelo (como en el log) responde primero la pseudo-llamada
        # anidada en shell, y luego, tras el resultado de visión, responde.
        def fake_llm(messages, tools, min_predict=None):
            if not llamadas:
                return {"content": "", "tool_calls": [{"function": {
                    "name": "shell",
                    "arguments": {"instruccion":
                                  f'vision("Describe: {img_path}")'},
                }}]}
            return {"content": "Es una foto de un gato.", "tool_calls": []}

        def fake_node(state):
            llamadas.append(state.get("_tool_args"))
            return {"vision_result": "un gato, sin shell"}

        def node_que_falla_si_es_shell(state):
            raise AssertionError("¡node_shell fue invocado con vision()!")

        monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
        nodos = {"vision": fake_node, "shell": node_que_falla_si_es_shell}
        monkeypatch.setattr(tr, "get_node_func", lambda t: nodos[t])

        state = {"orden": f"Describí la imagen adjunta: {img_path}",
                 "mem": {}, "agent_messages": None, "agent_pasos_log": None}

        out = gn.node_agent_loop(state)
        state.update(out)
        assert llamadas, "el nodo vision debió ejecutarse"
        assert img_path in str(llamadas[0]), (
            "el path de la imagen debía llegar a vision")
        out2 = gn.node_agent_loop(state)
        assert out2.get("final_response") == "Es una foto de un gato."

    def test_shell_con_json_payload_no_llega_a_zsh(self, monkeypatch):
        """Si el modelo emite [SHELL]{json}[/SHELL] dentro de node_shell,
        no se ejecuta nada: el parser devuelve None."""
        import core.agent.graph_nodes as gn
        ejecutado = []
        monkeypatch.setattr(gn, "ejecutar_comando",
                            lambda c: (ejecutado.append(c) or ("ok", False)))
        monkeypatch.setattr(
            gn, "_llm_chat",
            lambda system, user: ('[SHELL]\n{"instruccion": "Describe '
                                  'la imagen"}\n[/SHELL]'))
        monkeypatch.setattr(gn, "_system_prompt", lambda mem, state=None: "x")
        out = gn.node_shell({"orden": "describí la imagen", "mem": {}})
        assert ejecutado == [], "un payload JSON jamás debe llegar a zsh"
        assert out.get("shell_command") is None


# ── Tool 'text' cierra el loop (bug: bucle de respuestas repetidas) ─────────


class TestToolTextCierraElLoop:
    RESPUESTA = ("Es un personaje de caricatura con casco amarillo y "
                 "chaleco verde, sobre una barra de madera, con una copa "
                 "de vino y una pinta de cerveza.")

    def test_text_con_respuesta_completa_termina_de_inmediato(self, monkeypatch):
        """Reproducción del log: modelo llama text con la descripción
        completa 4 veces seguidas hasta MAX_AGENT_STEPS. Ahora la PRIMERA
        llamada a text cierra el loop con esa respuesta, literal."""
        import core.agent.graph_nodes as gn
        llamadas_llm = []

        def fake_llm(messages, tools, min_predict=None):
            llamadas_llm.append(1)
            return {"content": "", "tool_calls": [{"function": {
                "name": "text",
                "arguments": {"instruccion": self.RESPUESTA},
            }}]}

        def node_que_no_debe_llamarse(state):
            raise AssertionError("text no debe ejecutar node_text: ya tiene "
                                 "la respuesta")

        monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
        import core.agent.tool_registry as tr
        monkeypatch.setattr(tr, "get_node_func",
                            lambda t: node_que_no_debe_llamarse)

        state = {"orden": "describí la imagen", "mem": {},
                 "agent_messages": None, "agent_pasos_log": None}
        out = gn.node_agent_loop(state)

        assert out["final_response"] == self.RESPUESTA
        assert out["done"] is True and out["agent_activo"] is False
        assert len(llamadas_llm) == 1, "no puede haber una 2ª vuelta del loop"

    def test_text_en_formato_legado_tambien_cierra(self, monkeypatch):
        """El modelo que escribe [text] respuesta [/text] como texto también
        cierra el loop, no lo convierte en otro paso más."""
        import core.agent.graph_nodes as gn

        def fake_llm(messages, tools, min_predict=None):
            return {"content": f"[text] {self.RESPUESTA} [/text]",
                    "tool_calls": []}

        monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
        import core.agent.tool_registry as tr
        monkeypatch.setattr(tr, "get_node_func",
                            lambda t: pytest.fail("no debió ejecutarse ningún nodo"))

        state = {"orden": "mirá qué ves", "mem": {},
                 "agent_messages": None, "agent_pasos_log": None}
        out = gn.node_agent_loop(state)
        assert out["final_response"] == self.RESPUESTA
        assert out["agent_activo"] is False

    def test_text_vacio_no_cierra_pero_no_loopea_para_siempre(self, monkeypatch):
        """Una llamada text con instrucción vacía/meta no cierra el loop,
        recibe guía, y el modelo termina respondiendo en texto plano."""
        import core.agent.graph_nodes as gn
        rondas = []

        def fake_llm(messages, tools, min_predict=None):
            rondas.append(1)
            if len(rondas) == 1:
                return {"content": "", "tool_calls": [{"function": {
                    "name": "text", "arguments": {"instruccion": "ok"}}}]}
            return {"content": "Respuesta cortés final al usuario.",
                    "tool_calls": []}

        monkeypatch.setattr(gn, "_llm_chat_agente", fake_llm)
        state = {"orden": "hola", "mem": {},
                 "agent_messages": None, "agent_pasos_log": None}
        out = gn.node_agent_loop(state)
        state.update(out)
        out2 = gn.node_agent_loop(state)
        assert out2["final_response"] == "Respuesta cortés final al usuario."
        assert len(rondas) == 2  # como mucho 2 rondas, sin bucle
