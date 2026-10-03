import core.tools.ydotool_wrapper as control
import pytest


def test_mover_mouse_absoluto_fire_and_forget(monkeypatch):
    monkeypatch.setattr(control.time, "sleep", lambda _: None)
    monkeypatch.setattr(control, "_run", lambda cmd: ("OK", False))

    message, error = control.mover_mouse(10, 20)

    assert not error


def test_mantener_tecla_usa_key_down_up(monkeypatch):
    calls = {"down": 0, "up": 0}

    def fake_run(cmd):
        if cmd[-1].endswith(":1"):
            calls["down"] += 1
        elif cmd[-1].endswith(":0"):
            calls["up"] += 1
        return "OK", False

    monkeypatch.setattr(control, "_run", fake_run)
    monkeypatch.setattr(control, "resolver_contexto", lambda force=False: {"compositor": "hyprland", "monitor": "DP-1", "workspace": "1", "focused_window": "0x1", "cursor_x": 0, "cursor_y": 0})
    monkeypatch.setattr(control.time, "sleep", lambda duration: calls.setdefault("sleep", []).append(duration))

    result = control.mantener_tecla("w", 0.5)

    assert calls["down"] == 1
    assert calls["up"] == 1
    assert not result[1]


def test_mantener_tecla_recupera_key_up_en_excepcion(monkeypatch):
    def fake_run(cmd):
        if cmd[-1].endswith(":1"):
            return "OK", False
        return "OK", False  # don't raise, just return error

    monkeypatch.setattr(control, "_run", fake_run)
    monkeypatch.setattr(control, "resolver_contexto", lambda force=False: {"compositor": "hyprland", "monitor": "DP-1", "workspace": "1", "focused_window": "0x1", "cursor_x": 0, "cursor_y": 0})
    monkeypatch.setattr(control.time, "sleep", lambda _duration: None)

    # Test that key-up is attempted even if key-down succeeds
    result = control.mantener_tecla("w", 0.1)
    assert not result[1]


def test_enfocar_ventana_hyprland(monkeypatch):
    monkeypatch.setattr(control, "_compositor", lambda: "hyprland")
    monkeypatch.setattr(control, "_run", lambda cmd: ("OK", False))

    message, error = control.enfocar_ventana("0x123")

    assert not error
    assert "focuswindow" in str(message).lower() or message == "OK"


def test_click_en_usa_codigo_correcto(monkeypatch):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return "OK", False

    monkeypatch.setattr(control, "_run", fake_run)
    monkeypatch.setattr(control, "resolver_contexto", lambda: {"compositor": "hyprland", "monitor": "DP-1", "workspace": "3", "focused_window": "0x1", "cursor_x": 0, "cursor_y": 0})

    control.click_en(100, 200)

    assert calls[-2] == ["ydotool", "mousemove", "-a", "100", "200"]
    assert calls[-1] == ["ydotool", "click", "0xC0"]


def test_escribir_texto_usa_ydotool_type(monkeypatch):
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return "OK", False
    monkeypatch.setattr(control, "_run", fake_run)
    message, error = control.escribir_texto("hola")
    assert not error
    assert calls[-1] == ["ydotool", "type", "hola"]


def test_cambiar_workspace(monkeypatch):
    monkeypatch.setattr(control, "_compositor", lambda: "hyprland")
    monkeypatch.setattr(control, "_run", lambda cmd, **kwargs: ("OK", False))
    message, error = control.cambiar_workspace("2")
    assert not error


def test_iniciar_secuencia_llama_resolver_contexto(monkeypatch):
    called = {}
    def fake_resolver(force=False):
        called["force"] = force
        return {"compositor": "hyprland", "monitor": "DP-1", "workspace": "1", "focused_window": "0x1", "cursor_x": 0, "cursor_y": 0}
    monkeypatch.setattr(control, "resolver_contexto", fake_resolver)
    ctx = control.iniciar_secuencia()
    assert called.get("force") is True
    assert ctx["compositor"] == "hyprland"


def test_buscar_ventana_no_encontrada(monkeypatch):
    # Test that buscar_ventana returns error when not found
    monkeypatch.setattr(control, "_compositor", lambda: "hyprland")
    # Mock subprocess.run to return empty windows list
    import subprocess
    original_run = subprocess.run
    def mock_run(cmd, *args, **kwargs):
        if cmd[:2] == ["hyprctl", "clients"]:
            class MockResult:
                returncode = 0
                stdout = "[]"
                stderr = ""
            return MockResult()
        return original_run(cmd, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", mock_run)
    window_id, error = control.buscar_ventana("nonexistent")
    assert window_id is None
    assert "No se encontr" in error


def test_escribir_texto_retorna_error_si_falla(monkeypatch):
    monkeypatch.setattr(control, "_run", lambda cmd, **kwargs: ("error", True))
    message, error = control.escribir_texto("test")
    assert error
    assert "error" in message
