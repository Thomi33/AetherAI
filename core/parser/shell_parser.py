"""
Parsing de comandos shell en formato [SHELL] ... [/SHELL].
"""
import re


def _es_payload_tool_call(cmd: str) -> bool:
    """True si el "comando" es en realidad un payload JSON de tool call.

    Bug real (logs de producción): el modelo generó
        [SHELL] {"instruccion": "Describe la imagen adjunta: ..."} [/SHELL]
    y el JSON se ejecutó literalmente en zsh ('command not found:
    instruccion:'). Un objeto/array JSON NUNCA es un comando válido de zsh:
    se rechaza antes de ejecutar.
    """
    texto = cmd.strip()
    if not texto.startswith(("{", "[")):
        return False
    import json
    try:
        return isinstance(json.loads(texto), (dict, list))
    except (ValueError, TypeError):
        return False


def extraer_comando_shell(texto: str) -> str | None:
    """
    Extrae comando shell del formato [SHELL]comando[/SHELL].
    Fallback a bloques de código Markdown CON lenguaje shell explícito.

    NOTA: el fallback exige un tag de lenguaje (```bash/sh/zsh/shell). Antes
    el tag era opcional y capturaba CUALQUIER bloque ``` — incluido el output
    que el LLM alucina entre fences (p.ej. una salida fake de `lscpu`), que
    terminaba ejecutándose como comando ('zsh: no matches found: ...').

    NOTA 2: si lo extraído es un payload JSON de tool call
    ({"instruccion": ...}), se devuelve None: eso NO es un comando y jamás
    debe ejecutarse en la shell.
    """
    m = re.search(r"\[SHELL\]\s*(.*?)\s*\[/SHELL\]", texto, re.DOTALL)
    if m:
        cmd = m.group(1).strip()
        return None if _es_payload_tool_call(cmd) else cmd

    m = re.search(r"```(?:bash|sh|zsh|shell)\s*\n(.*?)\n```", texto, re.DOTALL)
    if m:
        cmd = m.group(1).strip()
        if cmd:
            return None if _es_payload_tool_call(cmd) else cmd

    return None
