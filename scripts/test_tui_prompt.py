"""Test manual del comando /prompt de la TUI (corre headless con Textual run_test)."""
import asyncio
import sys
sys.path.insert(0, "/home/thomi/mi_proyecto_crew")


async def main() -> int:
    from tui.app import AetherApp
    from core.config.config_manager import get_config_manager

    config = get_config_manager()
    # Estado inicial limpio
    for k in ("SYSTEM_PROMPT_OVERRIDE", "SYSTEM_PROMPT_EXTRA", "SYSTEM_PROMPT_SINTESIS_EXTRA"):
        config.set(k, "")

    app = AetherApp()
    async with app.run_test(size=(120, 40)) as pilot:
        chat = app.query_one("#chat_panel")

        # 1) /prompt (estado)
        app._manejar_comando_slash("/prompt")
        await pilot.pause()
        texto = "\n".join(str(x) for x in chat._vivo[-3:])
        assert "SYSTEM_PROMPT_EXTRA" in texto, texto
        print("[OK] /prompt muestra el estado")

        # 2) /prompt extra con texto largo (varias palabras)
        app._manejar_comando_slash("/prompt extra Usa emojis en cada respuesta final.")
        await pilot.pause()
        assert config.get("SYSTEM_PROMPT_EXTRA") == "Usa emojis en cada respuesta final."
        print("[OK] /prompt extra setea SYSTEM_PROMPT_EXTRA")

        # 3) /prompt preview refleja el extra
        largo = len(chat._vivo)
        app._manejar_comando_slash("/prompt preview")
        await pilot.pause()
        preview = "\n".join(str(x) for x in chat._vivo[largo:])
        assert "Usa emojis en cada respuesta final." in preview, preview[:400]
        print("[OK] /prompt preview refleja el extra aplicado")

        # 4) /prompt sintesis
        app._manejar_comando_slash("/prompt sintesis Nunca uses [SHELL] en la síntesis.")
        await pilot.pause()
        assert config.get("SYSTEM_PROMPT_SINTESIS_EXTRA") == "Nunca uses [SHELL] en la síntesis."
        print("[OK] /prompt sintesis setea SYSTEM_PROMPT_SINTESIS_EXTRA")

        # 5) /prompt clear all
        app._manejar_comando_slash("/prompt clear all")
        await pilot.pause()
        assert config.get("SYSTEM_PROMPT_EXTRA") == ""
        assert config.get("SYSTEM_PROMPT_SINTESIS_EXTRA") == ""
        assert config.get("SYSTEM_PROMPT_OVERRIDE") == ""
        print("[OK] /prompt clear all resetea")

        # 6) override vacío rechazado por validación de uso
        app._manejar_comando_slash("/prompt override")
        await pilot.pause()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
