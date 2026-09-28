"""
Tests del ciclo de vida de inferencia: estados, logs y cancelación.
Sin tocar el backend real: solo prueban las funciones de estado del servicio.
"""
import pytest


@pytest.fixture
def fresh_state():
    from backend.core.aether_service import (
        _CURRENT_STATE, _EVENT_LOG, _GEN_LOCK,
    )
    backup_state = _CURRENT_STATE.copy()
    backup_evs = list(_EVENT_LOG)
    backup_gen = _GEN_LOCK.locked()
    if _GEN_LOCK.locked():
        _GEN_LOCK.release()

    _CURRENT_STATE.update({
        "estado": "IDLE", "inf_id": None, "inicio": 0.0,
        "motivo": "", "error": None,
    })
    yield
    _CURRENT_STATE.update(backup_state)
    _EVENT_LOG[:] = backup_evs
    if _GEN_LOCK.locked() and not backup_gen:
        _GEN_LOCK.release()


class TestLifecycleStates:

    def test_queued_running_completed(self, fresh_state):
        from backend.core.aether_service import (
            _CURRENT_STATE, _EVENT_LOG, _GEN_LOCK,
            _emit_inference_queued, _emit_inference_start, _emit_inference_completed,
        )
        _emit_inference_queued("t1", "en cola")
        assert _CURRENT_STATE["estado"] == "QUEUED"

        _emit_inference_start("t1")
        assert _CURRENT_STATE["estado"] == "RUNNING"

        _emit_inference_completed("t1")
        assert _CURRENT_STATE["estado"] == "COMPLETED"
        assert _CURRENT_STATE["error"] is None

        tipos = [e["tipo"] for e in _EVENT_LOG]
        assert "INFERENCE_QUEUED" in tipos
        assert "INFERENCE_STARTED" in tipos
        assert "INFERENCE_COMPLETED" in tipos

    def test_queued_running_failed(self, fresh_state):
        from backend.core.aether_service import (
            _CURRENT_STATE, _EVENT_LOG, _GEN_LOCK,
            _emit_inference_queued, _emit_inference_start, _emit_inference_failed,
        )
        _emit_inference_queued("t2", "en cola")
        _emit_inference_start("t2")
        _emit_inference_failed("t2", "error de red")
        assert _CURRENT_STATE["estado"] == "FAILED"
        assert _CURRENT_STATE["error"] == "error de red"
        assert not _GEN_LOCK.locked()

    def test_queued_running_no_completed_queued_path(self, fresh_state):
        """Si solo se encola (sin start), el estado queda QUEUED."""
        from backend.core.aether_service import (
            _CURRENT_STATE, _emit_inference_queued,
        )
        _emit_inference_queued("t4", "otra inferencia está corriendo")
        assert _CURRENT_STATE["estado"] == "QUEUED"
        assert _CURRENT_STATE["inf_id"] == "t4"

    def test_lock_released_en_todos_los_caminos(self, fresh_state):
        """_GEN_LOCK siempre se libera después de cualquier ciclo."""
        from backend.core.aether_service import _GEN_LOCK
        assert not _GEN_LOCK.locked()


class TestIterEventosLifecycle:

    def test_iter_eventos_emite_done_o_error(self, fresh_state):
        """iter_eventos siempre termina con DoneEvent o ErrorEvent."""
        from backend.core.aether_service import AetherService, DoneEvent, ErrorEvent
        from core.agent.streaming import reset_cancel
        reset_cancel()
        eventos = list(AetherService.iter_eventos("hola"))
        dones = [e for e in eventos if isinstance(e, DoneEvent)]
        errors = [e for e in eventos if isinstance(e, ErrorEvent)]
        assert len(dones + errors) >= 1

    def test_failed_state_after_init_fail(self, fresh_state):
        """Si falla _asegurar_inicializado, el estado final es FAILED."""
        from backend.core.aether_service import AetherService, _CURRENT_STATE, ErrorEvent
        import backend.core.aether_service as svc
        from core.agent.streaming import reset_cancel
        reset_cancel()
        orig = svc.AetherService._asegurar_inicializado
        try:
            svc.AetherService._asegurar_inicializado = lambda: (
                (_ for _ in ()).throw(RuntimeError("boom-test"))
            )
            eventos = list(AetherService.iter_eventos("hola"))
            errors = [e for e in eventos if isinstance(e, ErrorEvent)]
            assert len(errors) == 1
            assert "boom-test" in errors[0].mensaje
            assert _CURRENT_STATE["estado"] == "FAILED"
        finally:
            svc.AetherService._asegurar_inicializado = orig
