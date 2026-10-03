"""Tests del bridge Aether → runtime Roblox."""
from __future__ import annotations

from pathlib import Path

from core.tools.roblox_bridge import RobloxRuntime


def test_roblox_runtime_focuses_sober_before_start(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        "core.tools.roblox_bridge.buscar_ventana",
        lambda name: calls.append(("find", name)) or ("0x1", None),
    )
    monkeypatch.setattr(
        "core.tools.roblox_bridge.enfocar_ventana",
        lambda window: calls.append(("focus", window)) or ("OK", False),
    )

    class FakeProcess:
        pid = 123

        def poll(self):
            return None

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout=None):
            calls.append(("wait", timeout))

    monkeypatch.setattr("core.tools.roblox_bridge.subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    runtime = RobloxRuntime(str(tmp_path))

    ok, message = runtime.start()

    assert ok is True
    assert "iniciado" in message
    assert calls[:2] == [("find", "org.vinegarhq.Sober"), ("focus", "0x1")]
    assert runtime.running is True


def test_roblox_runtime_rejects_missing_runtime(monkeypatch, tmp_path):
    runtime = RobloxRuntime(str(Path(tmp_path) / "missing"))
    ok, message = runtime.start()
    assert ok is False
    assert "No existe" in message


def test_roblox_runtime_injects_config_as_env_vars(monkeypatch, tmp_path):
    """El config snapshot se pasa como variables de entorno al subproceso."""
    captured_env: dict = {}

    monkeypatch.setattr(
        "core.tools.roblox_bridge.buscar_ventana",
        lambda name: ("0x1", None),
    )
    monkeypatch.setattr(
        "core.tools.roblox_bridge.enfocar_ventana",
        lambda window: ("OK", False),
    )

    class FakeProcess:
        pid = 999

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_popen(*args, **kwargs):
        captured_env.update(kwargs.get("env", {}))
        return FakeProcess()

    monkeypatch.setattr("core.tools.roblox_bridge.subprocess.Popen", fake_popen)
    runtime = RobloxRuntime(str(tmp_path))

    config = {"MODELO": "test-model:7b", "OLLAMA_HOST": "http://localhost:9999"}
    ok, _ = runtime.start(config=config)

    assert ok is True
    assert captured_env.get("AETHER_MODELO") == "test-model:7b"
    assert captured_env.get("AETHER_OLLAMA_HOST") == "http://localhost:9999"


def test_roblox_runtime_no_hardcoded_sys_path_in_subprocess_command(monkeypatch, tmp_path):
    """El comando lanzado al subproceso no debe contener sys.path.insert hardcodeado."""
    captured_cmd: list = []

    monkeypatch.setattr(
        "core.tools.roblox_bridge.buscar_ventana",
        lambda name: ("0x1", None),
    )
    monkeypatch.setattr(
        "core.tools.roblox_bridge.enfocar_ventana",
        lambda window: ("OK", False),
    )

    class FakeProcess:
        pid = 42

        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout=None):
            pass

    def fake_popen(cmd, **kwargs):
        captured_cmd.extend(cmd)
        return FakeProcess()

    monkeypatch.setattr("core.tools.roblox_bridge.subprocess.Popen", fake_popen)
    runtime = RobloxRuntime(str(tmp_path))
    runtime.start()

    cmd_str = " ".join(str(c) for c in captured_cmd)
    assert "sys.path.insert" not in cmd_str
    assert "aether_lab.orchestrator" in cmd_str


def test_roblox_runtime_forwards_google_credentials(monkeypatch, tmp_path):
    captured_env: dict = {}

    monkeypatch.setattr(
        "core.tools.roblox_bridge.buscar_ventana",
        lambda name: ("0x1", None),
    )
    monkeypatch.setattr(
        "core.tools.roblox_bridge.enfocar_ventana",
        lambda window: ("OK", False),
    )

    class FakeProcess:
        pid = 43

        def poll(self):
            return None

    def fake_popen(*args, **kwargs):
        captured_env.update(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr("core.tools.roblox_bridge.subprocess.Popen", fake_popen)
    runtime = RobloxRuntime(str(tmp_path), vision_provider="google")

    ok, _ = runtime.start(config={"GOOGLE_API_KEY": "test-key"})

    assert ok is True
    assert captured_env["GOOGLE_API_KEY"] == "test-key"


def test_roblox_runtime_reads_provider_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AETHER_ROBLOX_VISION_PROVIDER", "google")
    assert RobloxRuntime(str(tmp_path)).vision_provider == "google"
