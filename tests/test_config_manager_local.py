import json
from pathlib import Path

from core.config.config_manager import ConfigManager


def test_local_overrides_never_modify_base(tmp_path: Path):
    base = tmp_path / "config.json"
    base.write_text(
        json.dumps({"MODELO": "ornith-1.5:9b", "NUM_CTX": 32768}),
        encoding="utf-8",
    )

    manager = ConfigManager(base)
    assert manager.get("MODELO") == "ornith-1.5:9b"
    assert manager.get("NUM_CTX") == 32768

    assert manager.set("NUM_CTX", 16384)

    assert json.loads(base.read_text(encoding="utf-8"))["NUM_CTX"] == 32768
    local = base.with_name("config.local.json")
    assert json.loads(local.read_text(encoding="utf-8"))["NUM_CTX"] == 16384

    reloaded = ConfigManager(base)
    assert reloaded.get("NUM_CTX") == 16384

    assert reloaded.reset("NUM_CTX")
    assert reloaded.get("NUM_CTX") == 32768
    assert "NUM_CTX" not in json.loads(local.read_text(encoding="utf-8"))


def test_local_config_is_layered_over_base(tmp_path: Path):
    base = tmp_path / "config.json"
    local = tmp_path / "config.local.json"
    base.write_text(json.dumps({"MODELO": "base", "DEBUG": False}), encoding="utf-8")
    local.write_text(json.dumps({"MODELO": "local"}), encoding="utf-8")

    manager = ConfigManager(base)
    assert manager.get("MODELO") == "local"
    assert manager.get("DEBUG") is False
