"""
Tests de la tool net_audit (auditoría de red read-only) y del tope de
salida de la shell. Todo con `ss`/`journalctl` MOCKEADOS vía runner
inyectable: la suite NO requiere permisos de root ni Ollama.
"""
import json

import pytest


SS_ESTAB = """Netid State  Recv-Q Send-Q Local Address:Port  Peer Address:Port  Process
tcp   ESTAB  0      0      127.0.0.1:43124        127.0.0.1:8000
tcp   ESTAB  0      0      192.168.1.10:47892    140.82.121.4:443    users:(("chrome",pid=1234,fd=89))
tcp   ESTAB  0      0      192.168.1.10:52341    10.0.0.5:53          users:(("systemd-resolved",pid=998,fd=23))
udp   UNCONN 0      0      0.0.0.0:68            0.0.0.0:0
"""

SS_LISTEN = """Netid State  Recv-Q Send-Q Local Address:Port  Peer Address:Port  Process
tcp   LISTEN 0      128    0.0.0.0:22           0.0.0.0:0            users:(("sshd",pid=841,fd=3))
tcp   LISTEN 0      128    127.0.0.1:631        0.0.0.0:0
"""

JOURNAL_SSH_FALLAS = """-- Journal begins at ...
oct 06 10:11:12 host sshd[988]: Failed password for root from 185.99.1.1 port 54321 ssh2
oct 06 10:11:14 host sshd[989]: Invalid user admin from 185.99.1.1 port 54322 ssh2
oct 06 10:12:00 host sshd[990]: Accepted publickey for thomi from 192.168.1.5
"""

LAST_LOGINS = """thomi   pts/0        192.168.1.5     Sat Oct  4 21:12   still logged in
reboot  system boot  6.x             Sat Oct  4 20:55   still running
"""


def _runner(ok_journal=True, ss_estab_rc=0, ss_estab_out=SS_ESTAB):
    def run(cmd):
        if cmd.startswith("ss -tunap"):
            return ss_estab_out, ss_estab_rc
        if cmd.startswith("ss -tun") and "l" not in cmd:
            # degradación: sin -p
            return "\n".join(l.split("users:")[0].rstrip() for l in SS_ESTAB.splitlines()), 0
        if cmd.startswith("ss -tunl"):
            return SS_LISTEN, 0
        if cmd.startswith("journalctl"):
            if ok_journal:
                return JOURNAL_SSH_FALLAS, 0
            return "Failed to add match: Unit sshd.service could not be found.", 1
        if cmd.startswith("last"):
            return LAST_LOGINS, 0
        return "", 1
    return run


# ══════════════════════════════════════════════════════════════════════
# Clasificación de conexiones y puertos expuestos
# ══════════════════════════════════════════════════════════════════════

def test_clasifica_loopback_lan_y_externa():
    from core.tools.net_audit import auditar_red

    informe = auditar_red(runner=_runner())
    clases = {c["remoto"]: c["clase"]
              for c in informe["conexiones_establecidas"]}
    assert clases["127.0.0.1:8000"] == "loopback"
    assert clases["140.82.121.4:443"] == "externa"
    assert clases["10.0.0.5:53"] == "lan"
    # El UNCONN de DHCP (0.0.0.0:68) no es una conexión establecida.
    assert all(c["estado"] == "ESTAB" for c in informe["conexiones_establecidas"])


def test_marca_puertos_expuestos_vs_loopback():
    from core.tools.net_audit import auditar_red

    informe = auditar_red(runner=_runner())
    puerto_22 = next(p for p in informe["puertos_escucha"] if p["puerto"] == "22")
    puerto_631 = next(p for p in informe["puertos_escucha"] if p["puerto"] == "631")
    assert puerto_22["expuesto"] is True       # 0.0.0.0:22 → accesible afuera
    assert puerto_631["expuesto"] is False    # 127.0.0.1:631 → loopback


def test_ssh_fallidos_con_journal_disponible():
    from core.tools.net_audit import auditar_red

    informe = auditar_red(runner=_runner(ok_journal=True))
    assert informe["ssh_fallidos"] == 2        # Failed password + Invalid user
    assert "185.99.1.1" in informe["ssh_ultimo_fallo"]


def test_si_journalctl_falla_el_resto_sigue_valido():
    from core.tools.net_audit import auditar_red

    informe = auditar_red(runner=_runner(ok_journal=False))
    # ssh_fallidos es null con motivo, NO rompe el resto del JSON.
    assert informe["ssh_fallidos"] is None
    assert "motivo" in informe.get("ssh_fallidos_motivo", "").lower() \
        or "permisos" in informe.get("ssh_fallidos_motivo", "")
    assert len(informe["conexiones_establecidas"]) == 3
    assert informe["puertos_escucha"]
    assert informe["last_logins"]
    assert "authorized_keys" in informe


def test_degradacion_sin_permisos_de_proceso_avisa():
    from core.tools.net_audit import auditar_red

    informe = auditar_red(runner=_runner(ss_estab_rc=1, ss_estab_out=""))
    assert len(informe["conexiones_establecidas"]) == 3
    # Avisa la degradación y NUNCA menciona sudo.
    assert "nota_procesos" in informe
    assert "sudo" in informe["nota_procesos"]


def test_authorized_keys_aislado():
    from core.tools.net_audit import _sub_authorized_keys

    info = _sub_authorized_keys()
    assert "ruta" in info and "existe" in info  # existe o no, no rompe


# ══════════════════════════════════════════════════════════════════════
# Nodo del grafo + routing del planner
# ══════════════════════════════════════════════════════════════════════

def test_node_net_audit_devuelve_json_estructurado(monkeypatch):
    import core.tools.net_audit as na
    import core.agent.graph_nodes as gn

    monkeypatch.setattr(na, "auditar_red", lambda: {
        "conexiones_establecidas": [{"remoto": "140.82.121.4:443", "clase": "externa"}],
        "puertos_escucha": [{"puerto": "22", "expuesto": True}],
        "ssh_fallidos": 0,
    })
    out = gn.node_net_audit({"orden": "revisá la red", "mem": {}})
    informe = json.loads(out["net_audit_result"])
    assert informe["puertos_escucha"][0]["expuesto"] is True
    assert out["final_response"] is None        # el LLM solo resume después


def test_planner_envia_revision_de_red_a_net_audit(monkeypatch):
    import core.agent.graph_nodes as gn

    def _boom(*a, **k):
        raise AssertionError("El planner no debía llamar al LLM para esto")

    monkeypatch.setattr(gn, "_llm_chat", _boom)
    monkeypatch.setattr(gn, "_llm_chat_agente", _boom)

    out = gn.node_planner({
        "orden": "revisa mi trafico de red, a ver si no tengo algo sospechoso",
        "mem": {}, "sesion_id": "s"})
    assert [p["tool"] for p in out["plan_pasos"]] == ["net_audit"]
    assert not out.get("agent_activo")


# ══════════════════════════════════════════════════════════════════════
# P1-6: normalización de relevadores y tope de salida de la shell
# ══════════════════════════════════════════════════════════════════════

def test_normalizar_comando_red_agrega_t_u():
    from core.tools.shell_executor import normalizar_comando_red

    # netstat/ss genéricos: agrega -t -u al primer segmento del pipe.
    assert (normalizar_comando_red("netstat -an | grep ESTABLISHED")
            == "netstat -t -u -an | grep ESTABLISHED")
    # Los que ya filtran protocolo quedan intactos.
    assert normalizar_comando_red("ss -tunap") == "ss -tunap"
    assert normalizar_comando_red("netstat -tln") == "netstat -tln"
    # Otros comandos ni se tocan.
    assert normalizar_comando_red("ls -la | head") == "ls -la | head"


def test_salida_de_shell_se_trunca_con_aviso(monkeypatch):
    """P1-6: un volcado enorme (ej. netstat completo) se trunca al tope
    configurable con aviso honesto, sin perder las primeras líneas."""
    import core.tools.shell_executor as se

    class FakeResultado:
        stdout = "\n".join(f"linea {i}" for i in range(500)).encode()
        stderr = b""
        returncode = 0

    monkeypatch.setattr(se.subprocess, "run", lambda *a, **k: FakeResultado())
    salida, error = se.ejecutar_comando("seq 1 500")
    lineas = salida.splitlines()
    assert len(lineas) == 151                    # 150 líneas + aviso
    assert "[truncado: 350 lineas omitidas]".replace("lineas", "") in \
        lineas[-1].replace("líneas", "")
    assert lineas[0] == "linea 0" and lineas[149] == "linea 149"
    assert error is False
