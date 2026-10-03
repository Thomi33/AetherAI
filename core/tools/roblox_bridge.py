"""Lifecycle bridge: arranca/para el runtime autónomo de Roblox desde Aether.

El runtime corre en un subproceso separado. La configuración se pasa como
variables de entorno (snapshot readonly) para evitar la race condition de
config.json: el subproceso NUNCA lee el archivo directamente.

Uso desde Aether:
    runtime = RobloxRuntime()
    ok, msg = runtime.start()   # enfoca Sober y lanza el subproceso
    ok, msg = runtime.stop()    # termina el subproceso limpiamente
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from core.tools.ydotool_wrapper import buscar_ventana, enfocar_ventana

logger = logging.getLogger(__name__)


class RobloxRuntime:
    def __init__(
        self,
        runtime_dir: str | None = None,
        vision_provider: str | None = None,
    ) -> None:
        """
        Args:
            runtime_dir: Ruta al paquete aether-roblox. Por defecto
                usa AETHER_ROBLOX_DIR o ~/aether-roblox.
            vision_provider: "hybrid" (default), "ollama" o "google". Se pasa
                al orquestador via --vision-provider. "hybrid" usa Google
                primero y cae a Ollama si Google no está disponible. El
                proveedor Google usa las credenciales del entorno del proceso.
        """
        vision_provider = vision_provider or os.environ.get(
            "AETHER_ROBLOX_VISION_PROVIDER", "hybrid"
        )
        if vision_provider not in {"ollama", "google", "hybrid"}:
            raise ValueError("vision_provider debe ser 'ollama', 'google' o 'hybrid'.")
        configured = runtime_dir or os.environ.get("AETHER_ROBLOX_DIR", "~/aether-roblox")
        self.runtime_dir = Path(configured).expanduser()
        self.vision_provider = vision_provider
        self.process: subprocess.Popen[str] | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, config: dict | None = None) -> tuple[bool, str]:
        """Enfoca Sober y arranca el subproceso del orquestador.

        Args:
            config: Snapshot readonly del ConfigManager de Aether. Se inyecta
                como variables de entorno para evitar race conditions con
                config.json. Si es None se usan los defaults del orquestador.
        """
        if self.running:
            return False, "El runtime Roblox ya está activo."
        if not self.runtime_dir.is_dir():
            return False, f"No existe el runtime Roblox: {self.runtime_dir}"

        window_id, lookup_error = buscar_ventana("org.vinegarhq.Sober")
        if lookup_error or window_id is None:
            return False, lookup_error or "No se encontró la ventana de Sober."
        focus_message, focus_error = enfocar_ventana(window_id)
        if focus_error:
            return False, f"No se pudo enfocar Sober: {focus_message}"

        env = os.environ.copy()
        # PYTHONPATH: el subproceso hereda acceso a core.tools sin sys.path.insert
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.runtime_dir), "/home/thomi/mi_proyecto_crew", env.get("PYTHONPATH", "")]
        )
        env["AETHER_VISION_PROVIDER"] = self.vision_provider
        # Config snapshot → variables de entorno (solo lectura, sin tocar config.json)
        if config:
            if "MODELO" in config:
                env["AETHER_MODELO"] = str(config["MODELO"])
            if "OLLAMA_HOST" in config:
                env["AETHER_OLLAMA_HOST"] = str(config["OLLAMA_HOST"])
            # No guardamos secretos en config.json, pero permitimos que una
            # integración los inyecte explícitamente en el snapshot.
            for key in ("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_API_KEY"):
                if config.get(key):
                    env[key] = str(config[key])

        self.process = subprocess.Popen(
            [
                sys.executable, "-m", "aether_lab.orchestrator",
                "--log-level", "INFO",
                "--vision-provider", self.vision_provider,
            ],
            cwd=self.runtime_dir,
            env=env,
            text=True,
            start_new_session=True,
        )
        logger.info(
            "Roblox runtime iniciado pid=%s dir=%s provider=%s",
            self.process.pid, self.runtime_dir, self.vision_provider,
        )
        return True, f"Roblox autónomo iniciado (pid {self.process.pid})."

    def stop(self) -> tuple[bool, str]:
        if not self.running:
            self.process = None
            return False, "El runtime Roblox no está activo."
        assert self.process is not None
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        pid = self.process.pid
        self.process = None
        return True, f"Roblox autónomo detenido (pid {pid})."
