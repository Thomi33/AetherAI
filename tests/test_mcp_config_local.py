"""
Tests del sistema de dos capas de MCP (plantilla + local), mismo patrón que
config.json / config.local.json:

  - mcp_servers.json       → plantilla versionada con placeholders "{token}".
  - mcp_servers.local.json → tokens reales + servers custom + tombstones
    (ignorado por Git; TODO lo que escribe el runtime vive acá).

El merge es por-server a nivel campo: local pisa campo a campo, los servers
solo-locales se agregan, y "__deleted__": true oculta un server de la
plantilla sin tocarla.
"""
import json

import pytest


PLANTILLA = {
    "github": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "{token}"},
        "enabled": True,
    },
    "notion": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@notionhq/notion-mcp-server"],
        "env": {"NOTION_TOKEN": "{token}"},
        "enabled": True,
    },
}


@pytest.fixture
def capas(tmp_path, monkeypatch):
    """Dos archivos tmp (plantilla + local) inyectados en mcp_client."""
    import core.tools.mcp_client as mc

    base = tmp_path / "mcp_servers.json"
    local = tmp_path / "mcp_servers.local.json"
    base.write_text(json.dumps(PLANTILLA, indent=2), encoding="utf-8")
    local.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(mc, "RUTA_BASE", base)
    monkeypatch.setattr(mc, "RUTA_LOCAL", local)
    return {"base": base, "local": local, "mc": mc}


@pytest.fixture
def client(capas):
    """App FastAPI mínima con el router de MCP (patrón de test_memoria_json)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.api.routes import mcp_routes

    app = FastAPI()
    app.include_router(mcp_routes.router, prefix="/api")
    return TestClient(app)


class TestMergeDosCapas:

    def test_local_pisa_token_y_conserva_resto_de_plantilla(self, capas):
        capas["local"].write_text(json.dumps({
            "github": {"env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_REAL123"}}
        }), encoding="utf-8")
        merged = capas["mc"].cargar_merge()
        assert merged["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_REAL123"
        assert merged["github"]["command"] == "npx"
        assert merged["github"]["args"] == PLANTILLA["github"]["args"]
        assert merged["notion"]["env"]["NOTION_TOKEN"] == "{token}"

    def test_server_solo_local_se_agrega(self, capas):
        capas["local"].write_text(json.dumps({
            "mi-custom": {"command": "npx", "args": ["-y", "@x/mcp"]}
        }), encoding="utf-8")
        merged = capas["mc"].cargar_merge()
        assert set(merged) == {"github", "notion", "mi-custom"}

    def test_tombstone_oculta_server_de_plantilla(self, capas):
        capas["local"].write_text(json.dumps({
            "notion": {"__deleted__": True}
        }), encoding="utf-8")
        merged = capas["mc"].cargar_merge()
        assert "notion" not in merged
        assert "github" in merged
        plantilla = json.loads(capas["base"].read_text(encoding="utf-8"))
        assert "notion" in plantilla

    def test_local_vacio_da_solo_la_plantilla(self, capas):
        merged = capas["mc"].cargar_merge()
        assert merged == PLANTILLA

    def test_sin_archivo_local_la_plantilla_funciona_sola(self, capas):
        capas["local"].unlink()
        merged = capas["mc"].cargar_merge()
        assert merged == PLANTILLA


class TestCapaLocal:

    def test_guardar_cargar_roundtrip(self, capas):
        data = {"x": {"command": "npx", "env": {"K": "secreto"}}}
        capas["mc"].guardar_local(data)
        assert capas["mc"].cargar_local() == data
        assert not capas["local"].with_suffix(".json.tmp").exists()

    def test_set_enabled_en_server_de_plantilla_crea_override_parcial(self, capas):
        capas["mc"].set_enabled("github", False)
        local = capas["mc"].cargar_local()
        assert local["github"] == {"enabled": False}
        merged = capas["mc"].cargar_merge()
        assert merged["github"]["enabled"] is False
        assert merged["github"]["command"] == "npx"
        assert merged["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "{token}"

    def test_set_enabled_en_server_custom_preserva_config(self, capas):
        capas["mc"].guardar_local({
            "mi-custom": {"command": "npx", "args": ["-y", "@x/mcp"],
                          "env": {"K": "secreto"}, "enabled": True}
        })
        capas["mc"].set_enabled("mi-custom", False)
        local = capas["mc"].cargar_local()
        assert local["mi-custom"]["enabled"] is False
        assert local["mi-custom"]["command"] == "npx"
        assert local["mi-custom"]["env"] == {"K": "secreto"}


class TestRutasMcp:

    def test_get_lista_mergeado_y_enmascara_tokens(self, client, capas):
        capas["local"].write_text(json.dumps({
            "github": {"env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_REAL123"}}
        }), encoding="utf-8")
        r = client.get("/api/mcps")
        assert r.status_code == 200 and r.json()["ok"]
        servers = {s["name"]: s for s in r.json()["servers"]}
        assert "ghp_REAL123" not in r.text
        assert servers["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"]["valor"] == "********"

    def test_post_agrega_solo_al_local(self, client, capas):
        r = client.post("/api/mcps", json={
            "name": "mi-server", "transport": "stdio",
            "command": "npx", "args": ["-y", "@x/mcp"],
            "env": {"API_KEY": "sk-REAL-987654321"},
        })
        assert r.status_code == 200 and r.json()["ok"]
        local = json.loads(capas["local"].read_text(encoding="utf-8"))
        assert local["mi-server"]["env"]["API_KEY"] == "sk-REAL-987654321"
        plantilla = json.loads(capas["base"].read_text(encoding="utf-8"))
        assert "mi-server" not in plantilla
        assert plantilla == PLANTILLA

    def test_post_409_si_ya_existe_en_vista_mergeada(self, client, capas):
        r = client.post("/api/mcps", json={
            "name": "github", "transport": "stdio", "command": "npx",
        })
        assert r.status_code == 409

    def test_patch_conserva_token_real_con_mask(self, client, capas):
        capas["local"].write_text(json.dumps({
            "github": {"env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_REAL123"}}
        }), encoding="utf-8")
        r = client.patch("/api/mcps/github", json={
            "name": "github", "transport": "stdio", "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-github"],
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "********"},
        })
        assert r.status_code == 200 and r.json()["ok"]
        local = json.loads(capas["local"].read_text(encoding="utf-8"))
        assert local["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "ghp_REAL123"
        plantilla = json.loads(capas["base"].read_text(encoding="utf-8"))
        assert plantilla["github"]["env"]["GITHUB_PERSONAL_ACCESS_TOKEN"] == "{token}"

    def test_delete_server_de_plantilla_crea_tombstone(self, client, capas):
        r = client.delete("/api/mcps/notion")
        assert r.status_code == 200 and r.json()["ok"]
        local = json.loads(capas["local"].read_text(encoding="utf-8"))
        assert local["notion"] == {"__deleted__": True}
        plantilla = json.loads(capas["base"].read_text(encoding="utf-8"))
        assert "notion" in plantilla
        r2 = client.get("/api/mcps")
        nombres = {s["name"] for s in r2.json()["servers"]}
        assert "notion" not in nombres

    def test_delete_server_custom_lo_saca_del_local(self, client, capas):
        capas["local"].write_text(json.dumps({
            "mi-custom": {"command": "npx", "args": ["-y", "@x/mcp"]}
        }), encoding="utf-8")
        r = client.delete("/api/mcps/mi-custom")
        assert r.status_code == 200 and r.json()["ok"]
        local = json.loads(capas["local"].read_text(encoding="utf-8"))
        assert "mi-custom" not in local

    def test_patch_renombra_y_tombstonea_nombre_viejo(self, client, capas):
        r = client.patch("/api/mcps/notion", json={
            "name": "notion-pro", "transport": "stdio", "command": "npx",
            "args": ["-y", "@notionhq/notion-mcp-server"],
            "env": {"NOTION_TOKEN": "secret_REAL_99"},
        })
        assert r.status_code == 200
        local = json.loads(capas["local"].read_text(encoding="utf-8"))
        assert local["notion"] == {"__deleted__": True}
        assert local["notion-pro"]["env"]["NOTION_TOKEN"] == "secret_REAL_99"
        r2 = client.get("/api/mcps")
        nombres = {s["name"] for s in r2.json()["servers"]}
        assert "notion" not in nombres and "notion-pro" in nombres
