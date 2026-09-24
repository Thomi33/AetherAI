"""Tests del store JSON unificado de memoria (backlog Objetivos 2/3).

Ciclo exigido para cada tipo de dato:
    crear → guardar → "reiniciar" (releer desde disco) → recuperar.

Más: recuperación ante JSON corrupto (restaura .bak), cuarentena cuando todo
falla, migración one-shot desde la DB sqlite legacy, y el contrato REST.
"""
from __future__ import annotations

import json

import pytest

import core.memory.memoria_store as store
import core.memory.memory_manager as mm


@pytest.fixture
def mem_tmp(tmp_path, monkeypatch):
    """Aísla el store en tmp_path (las rutas son atributos de módulo)."""
    ruta = tmp_path / "db" / "memoria.json"
    monkeypatch.setattr(store, "RUTA_JSON", ruta)
    monkeypatch.setattr(store, "RUTA_BAK", ruta.with_suffix(".json.bak"))
    # En los tests, sin DB legacy salvo que un test la cree explícitamente.
    monkeypatch.setattr(store, "BASE_AETHER", tmp_path)
    return ruta


class TestRecuerdos:
    def test_crear_guardar_reiniciar_recuperar(self, mem_tmp):
        assert mm.cargar_memoria()["conversacion"] == []
        mm.guardar_recuerdo("al usuario le gusta el café sin azúcar",
                            categoria="preferencias", importancia=6)
        # "Reinicio": cada llamada relee el archivo desde disco (stateless).
        recs = mm.obtener_recuerdos()
        assert len(recs) == 1
        assert recs[0]["contenido"] == "al usuario le gusta el café sin azúcar"
        assert recs[0]["importancia"] == 6 and recs[0]["id"] >= 1
        # El archivo existe de verdad y es JSON válido.
        data = json.loads(mem_tmp.read_text(encoding="utf-8"))
        assert data["recuerdos"][0]["contenido"].startswith("al usuario")

    def test_ids_monotonos_tras_borrar(self, mem_tmp):
        mm.guardar_recuerdo("uno")
        mm.guardar_recuerdo("dos")
        a, b = mm.obtener_recuerdos(limit=10)
        # (misma importancia → el más reciente primero)
        assert (b["contenido"], a["contenido"]) == ("uno", "dos")
        assert mm.borrar_recuerdo(a["id"]) is True
        mm.guardar_recuerdo("tres")
        assert mm.obtener_recuerdos()[0]["id"] > b["id"]
        assert mm.borrar_recuerdo(99999) is False

    def test_filtros_y_vacio(self, mem_tmp):
        mm.guardar_recuerdo("gato", categoria="mascotas", importancia=8)
        mm.guardar_recuerdo("té", categoria="comida", importancia=3)
        assert [r["contenido"] for r in mm.obtener_recuerdos(categoria="mascotas")] == ["gato"]
        assert [r["contenido"] for r in mm.obtener_recuerdos(importancia_min=5)] == ["gato"]
        mm.guardar_recuerdo("   ")  # vacío no se guarda
        assert len(mm.obtener_recuerdos(limit=10)) == 2


class TestRestoDeSecciones:
    def test_turnos_y_sesiones(self, mem_tmp):
        mem = mm.cargar_memoria()
        mm.registrar_turno(mem, "usuario", "hola", sesion_id="s1", tema="chat")
        mm.registrar_turno(mem, "asistente", "hola qué tal", sesion_id="s1")
        mm.registrar_turno(mem, "usuario", "otra sesión", sesion_id="s2")
        assert len(mm.obtener_turnos_por_sesion("s1")) == 2
        sesiones = mm.listar_sesiones()
        assert [s["sesion_id"] for s in sesiones] == ["s2", "s1"]
        assert sesiones[0]["preview"] == "otra sesión"
        # Reinicio: el dict RAM nuevo sale del JSON, no del proceso viejo.
        mem2 = mm.cargar_memoria()
        assert len(mem2["conversacion"]) == 3

    def test_comandos_persisten(self, mem_tmp):
        mem = mm.cargar_memoria()
        mm.registrar_comando(mem, "listá archivos", "ls -la", exitoso=True)
        mem2 = mm.cargar_memoria()
        assert mem2["historial_comandos"][0]["cmd"] == "ls -la"

    def test_core_y_resumen(self, mem_tmp):
        mm.actualizar_core("proyecto_activo", "aether")
        assert mm.leer_core_memory()["proyecto_activo"] == "aether"
        mm.guardar_resumen("resumen uno")
        assert mm.obtener_resumen()["texto"] == "resumen uno"
        # guardar sin ultimo_turno_id conserva el cursor de consolidación.
        mm.guardar_resumen("resumen dos", ultimo_turno_id=7)
        mm.guardar_resumen("resumen tres")
        assert mm.obtener_resumen()["ultimo_turno_id"] == 7


class TestRecuperacion:
    def test_json_corrupto_restaura_bak(self, mem_tmp):
        mm.guardar_recuerdo("recuerdo valioso")
        mm.guardar_recuerdo("segundo")  # .bak queda con el estado de 1 recuerdo
        mem_tmp.write_text("{ json roto !!!", encoding="utf-8")
        recs = mm.obtener_recuerdos()
        assert [r["contenido"] for r in recs] == ["recuerdo valioso"]

    def test_json_y_bak_rotos_arranca_vacio_y_sigue_andando(self, mem_tmp):
        mem_tmp.parent.mkdir(parents=True, exist_ok=True)
        mem_tmp.write_text("basura", encoding="utf-8")
        mem_tmp.with_suffix(".json.bak").write_text("también", encoding="utf-8")
        mem = mm.cargar_memoria()
        assert mem["conversacion"] == [] and mm.obtener_recuerdos() == []
        cuarentena = list(mem_tmp.parent.glob("memoria.corrupt-*.json"))
        assert cuarentena, "el JSON corrupto debe apartarse, no perderse"
        # El sistema sigue funcionando inmediatamente después.
        mm.guardar_recuerdo("nuevo")
        assert mm.obtener_recuerdos()[0]["contenido"] == "nuevo"


class TestMigracionLegacy:
    def test_migra_todo_desde_current_db(self, tmp_path, monkeypatch):
        """Crea una current.db sqlite de juguete (formato legacy) y verifica
        que TODO migra al JSON la primera vez que se carga el store."""
        import sqlite3
        db = tmp_path / "db" / "current.db"
        db.parent.mkdir(parents=True)
        con = sqlite3.connect(db)
        con.executescript("""
            CREATE TABLE core_memory (clave TEXT PRIMARY KEY, valor TEXT);
            CREATE TABLE resumen_memoria (id INTEGER PRIMARY KEY, texto TEXT,
                                          ultimo_turno_id INTEGER, actualizado TEXT);
            CREATE TABLE conversaciones (id INTEGER PRIMARY KEY, fecha TEXT,
                rol TEXT, texto TEXT, sesion_id TEXT, tema TEXT, proyecto TEXT);
            CREATE TABLE comandos (id INTEGER PRIMARY KEY, fecha TEXT,
                orden TEXT, cmd TEXT, sesion_id TEXT, exitoso INTEGER);
            CREATE TABLE recuerdos (id INTEGER PRIMARY KEY, fecha TEXT,
                categoria TEXT, contenido TEXT, importancia INTEGER);
        """)
        con.execute("INSERT INTO core_memory VALUES ('k1', 'v1')")
        con.execute("INSERT INTO resumen_memoria VALUES (1, 'res viejo', 1, 'f')")
        con.execute("INSERT INTO conversaciones VALUES (1, 'f1', 'usuario', 'hola', 's1', '', '')")
        con.execute("INSERT INTO comandos VALUES (1, 'f1', 'o', 'ls', 's1', 1)")
        con.execute("INSERT INTO recuerdos VALUES (1, 'f1', 'c', 'recuerdo legacy', 5)")
        con.commit(); con.close()

        ruta = tmp_path / "db" / "memoria.json"  # no existe: dispara migración
        monkeypatch.setattr(store, "RUTA_JSON", ruta)
        monkeypatch.setattr(store, "RUTA_BAK", ruta.with_suffix(".json.bak"))
        monkeypatch.setattr(store, "BASE_AETHER", tmp_path)

        mem = mm.cargar_memoria()
        assert mem["core"]["k1"] == "v1"
        assert mem["resumen"] == "res viejo"
        assert mem["conversacion"][0]["texto"] == "hola"
        assert mem["historial_comandos"][0]["cmd"] == "ls"
        assert mm.obtener_recuerdos()[0]["contenido"] == "recuerdo legacy"
        assert ruta.exists(), "la migración materializa memoria.json"
        # Segunda carga: viene del JSON (aunque la DB legacy desapareciera).
        db.unlink()
        assert mm.cargar_memoria()["conversacion"][0]["texto"] == "hola"


class TestIntegridadIdentidad:
    def test_sin_nombre_hardcodeado(self, monkeypatch):
        """Sin perfil de cuenta, memoria vacía NO trae nombre de usuario."""
        class _CfgFalsa:
            def get(self, *a, **k):
                return {}
        monkeypatch.setattr(mm, "nombre_usuario_default",
                            lambda: "")  # config no disponible en tests
        mem = mm._memoria_vacia()
        assert mem["preferencias"]["nombre_usuario"] == "", (
            "el nombre del usuario jamás debe venir hardcodeado del código")


class TestSystemPromptEndpoints:
    def test_todas_las_rutas_de_system_prompt_existen_y_preview_completo(self):
        """La Web UI mostraba vacío: la pestaña necesita el prompt genérico
        real. El preview debe incluir las 3 personas (backstory, síntesis y
        agent loop — este último es el que usa el agente hoy)."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from backend.api.routes import prompt_routes
        app = FastAPI()
        app.include_router(prompt_routes.router, prefix="/api")
        client = TestClient(app)

        r = client.get("/api/system-prompt")
        assert r.status_code == 200 and r.json()["ok"]
        assert {"override", "extra", "sintesis_extra"} <= set(r.json())

        p = client.get("/api/system-prompt/preview")
        assert p.status_code == 200
        data = p.json()
        assert data.get("backstory"), "el prompt genérico debe mostrarse"
        assert data.get("persona_sintesis")
        assert data.get("agent_loop"), "falta el prompt del agent loop"
        assert "[ADJUNTOS DEL USUARIO]" in data["agent_loop"]
