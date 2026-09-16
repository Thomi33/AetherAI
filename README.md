# Aether

> Agente local, terminal-first y en español (por temas de seguridad).
> **LangGraph + Ollama + herramientas reales + memoria persistente.**

Aether es un asistente local que conversa, analiza proyectos, busca información
actualizada, lee y escribe archivos, ejecuta comandos, genera código, observa
la pantalla y encadena varias acciones sin enviar el proyecto a servicios
externos. El modelo corre en tu máquina mediante Ollama.

## Índice

- [Instalación](#instalación)
- [Uso](#uso)
- [Cómo funciona](#cómo-funciona)
- [Herramientas](#herramientas)
- [Configuración y rendimiento](#configuración-y-rendimiento)
- [Memoria y datos](#memoria-y-datos)
- [Arquitectura del proyecto](#arquitectura-del-proyecto)
- [Desarrollo y pruebas](#desarrollo-y-pruebas)
- [Solución de problemas](#solución-de-problemas)

## Instalación

### Descargar Aether

<<<<<<< HEAD
No necesitás instalar manualmente Python, pip, Ollama ni las dependencias de
Aether antes de empezar. El instalador se encarga de preparar el entorno.
=======
- Linux (el instalador está optimizado para Arch Linux).
- Python 3.10 o superior.
- [Ollama](https://ollama.com) ejecutándose en `http://localhost:11434`.
- `zsh` para las herramientas de shell.
- `ydotool` y `ydotoold` para control de mouse/teclado.
- `hyprctl` (Hyprland) o `swaymsg` (Sway) para verificación determinista.
- `grim` solo si vas a usar visión en Hyprland/Wayland.
>>>>>>> 51ef6cd2 (perf: velocidad absoluta)

Podés descargar el repositorio de dos formas:

**Con Git:**

```bash
git clone https://github.com/Thomi33/AetherAI.git
cd AetherAI
```

**Sin Git:** abrí el repositorio en GitHub, elegí **Code → Download ZIP**,
descomprimí el archivo y abrí una terminal dentro de la carpeta `AetherAI`.

### Instalar

Desde la raíz del repositorio:

```bash
chmod +x install.sh
./install.sh
```

**Eso es todo.** El instalador se encarga automáticamente de:

1. Detectar CPU, RAM, VRAM y espacio disponible.
2. Elegir el tier de hardware (`POTATO`, `LOW`, `MID`, `HIGH` o `ULTRA`).
3. Seleccionar el modelo y los parámetros apropiados para ese equipo.
4. Instalar las dependencias Python del core de Aether.
5. Preparar el entorno virtual de Aether.
6. Instalar/verificar Ollama cuando corresponde.
7. Descargar el modelo seleccionado, con tu autorización.
8. Configurar memoria, SQLite y los parámetros de ejecución.
9. Instalar el comando global `aether` en `~/.local/bin` y dejarlo disponible en
   el `PATH`.
10. Ejecutar una validación final del entorno y del modelo.

Durante la instalación solo se te pueden pedir algunas decisiones normales,
como la contraseña de `sudo`, si querés guardar los datos en `~/Aether` y si
querés descargar el modelo seleccionado.

No hace falta ejecutar después comandos de `pip`, crear otro virtualenv ni
instalar las dependencias de Aether a mano.

### Hardware y perfiles

El instalador adapta el modelo y los parámetros al equipo detectado. Por
ejemplo, una máquina muy limitada puede terminar como `POTATO` y usar
`qwen2.5:1.5b`, mientras un equipo más potente recibe un perfil superior.

El contexto recomendado se ajusta automáticamente según el hardware, con un
límite de `65536`.

La visión usa el mismo modelo principal configurado cuando ese modelo soporta
entrada multimodal; no se descarga un segundo modelo de visión.

## Uso

### TUI principal

Después de instalar:

```bash
aether
```

También funciona directamente desde el proyecto:

```bash
python run.py
```

### Lanzador global

El instalador crea el comando `aether` en `~/.local/bin`, por lo que podés
abrir Aether desde cualquier directorio:

```bash
aether
aether task "listá los archivos del proyecto"
aether cli
aether doctor
aether --version
```

También podés ejecutar Aether sobre un directorio específico:

```bash
cd ~/Proyecto
aether
aether task "creá un README para este proyecto"
aether --workdir ~/Proyecto task "ejecutá los tests"
```

Antes de operar en un directorio nuevo, Aether solicita autorización. Las
denegaciones no se guardan: si volvés a iniciar Aether, volverá a preguntar.

### Detener una inferencia

Durante una operación podés usar cualquiera de estas opciones:

- Botón **Detener** en la TUI.
- `Ctrl+C`.
- Comando `/stop`.

La cancelación es cooperativa: detiene el flujo cuando Ollama entrega el
siguiente evento disponible.

## Cómo funciona

Cada pedido se convierte en un estado `AetherState` y atraviesa un grafo
LangGraph:

```text
START
  -> planner
  -> context_manager
  -> plan_executor o agent_loop
  -> plan_synthesizer/finalize
  -> END
```

- **Planner:** resuelve fast-paths simples y prepara planes de uno o varios
  pasos.
- **Agent loop:** usa tool calling nativo de Ollama para tareas abiertas.
- **Executor:** despacha cada paso a la implementación real de la herramienta.
- **Synthesizer:** combina resultados cuando hay varias acciones.
- **Error handler:** diagnostica errores, puede buscar una solución, reintenta
  dentro de límites y deja el error explícito si no se resuelve.
- **Context manager:** poda y compacta contexto para evitar consumo innecesario
  de VRAM.

Si Aether no tiene certeza, el prompt le indica buscar primero en internet
mediante `web` cuando el dato sea actual, versionado o verificable externamente.
Para hechos locales debe preferir archivos, shell y herramientas del sistema.

La TUI muestra estados reales de ejecución, por ejemplo:

```text
[•] Analizando solicitud...
[•] Preparando plan...
[•] Ejecutando herramienta: shell
aptretando contexto pa salvar tu VRAM...
[✓] Operación completada
```

Los mensajes de personalidad son opcionales, breves y secundarios; no
reemplazan estados reales ni inventan acciones.

## Herramientas

| Herramienta | Función |
|---|---|
| `text` | Conversación directa con streaming. |
| `web` | Búsqueda y lectura de información actual. |
| `shell` | Ejecuta comandos `zsh` con barreras y timeout. |
| `launch` | Abre aplicaciones y paquetes Flatpak. |
| `vision` | Captura y analiza la pantalla con el modelo principal multimodal. |
| `codigo` | Genera, guarda y ejecuta Python, Bash o Java. |
| `memory` | Consulta, guarda o elimina recuerdos. |
| `file_write` | Guarda el resultado de un paso en un archivo. |
| `fs_read` / `fs_write` | Lee y escribe archivos con rutas explícitas. |
| `fs_mkdir` / `fs_list` | Crea directorios y lista contenido. |
| `computer_use` | Controla mouse/teclado cuando está disponible. |
| `mcp` | Invoca servidores MCP configurados. |

Las rutas relativas se resuelven en el directorio de trabajo autorizado, no en
la carpeta del repositorio. Por defecto la memoria, notas y logs viven aparte
en `~/Aether`; si durante el onboarding se elige no usar un home separado,
viven en `<proyecto>/.aether-data/`.

`computer_use` consulta directamente a Hyprland (`hyprctl`) o Sway
(`swaymsg`) para resolver monitor, workspace, foco y cursor. Ese contexto se
cachea durante la secuencia y se invalida solo después de acciones que pueden
cambiarlo. Las acciones de mouse se reintentan con backoff corto y se verifican
sin VLM; el diagnóstico visual queda reservado para un fallback explícito.
Cada acción registra tipo, contexto, resultado y reintentos en el logger de
`core.tools.computer_control`.

Las órdenes con coordenadas o workspace explícitos usan un fast-path
determinista y registran `duracion_ms`; no invocan el VLM. Cuando una acción
requiere visión, `ver_pantalla` admite un recorte `x,y,ancho,alto` para evitar
capturas mayores a la región relevante. La rama Sway tiene cobertura con mocks
del formato de `swaymsg`, pero no fue validada en vivo en este entorno.

En una medición local de referencia, mover el cursor a otro monitor y verificarlo
con Hyprland tomó `9.36 ms` end-to-end (incluyendo la consulta posterior).
Como comparación concreta, una captura completa más una inferencia real con
`ornith-1.5:9b` tomó `32794.47 ms`; esa es la latencia que el fast-path evita
para una orden con coordenadas explícitas.

## Configuración y rendimiento

La configuración principal está en [`core/config/config.json`](./core/config/config.json).
Ese JSON es la fuente única de valores configurables. `settings.py` se conserva
como adaptador de compatibilidad para rutas calculadas y módulos antiguos; no
debe contener valores alternativos de configuración.
Los valores más importantes son:

- `MODELO`: modelo principal para texto, tool calling y visión.
- `AETHER_DATA_DIR`: carpeta donde se guardan memoria, notas y logs.
- `NUM_CTX`: contexto efectivo; el instalador lo ajusta según hardware.
- `NUM_PREDICT`: máximo de tokens generados.
- `MAX_AGENT_STEPS`: límite de iteraciones del agente.
- `OLLAMA_KEEP_ALIVE`, threads y batch: parámetros de latencia y memoria.
- `STT_ENABLED`: dictado por voz opcional.

Los valores editados manualmente tienen prioridad sobre los defaults del
programa. Un contexto más grande no siempre es más rápido: depende de la RAM,
VRAM, cuantización y modelo elegido.

## Memoria y datos

Aether mantiene memoria temporal en RAM y persistente mediante el subsistema
controlado de [`core/memory/store`](./core/memory/store):

- Producción: `$AETHER_DATA_DIR/db/current.db`.
- Pruebas: `$AETHER_DATA_DIR/db/staging.db`.
- Snapshots: `$AETHER_DATA_DIR/db/snapshots/`.
- Backups: `$AETHER_DATA_DIR/db/backups/`.

Las escrituras pasan por `MemoryStore`, migraciones versionadas y un guard de
integridad. La base `memoria.db` que estaba en la raíz era un artefacto legacy
sin uso por el runtime actual; fue conservada fuera del árbol principal en
`_legacy/data/` junto con el dump antiguo para no perder datos históricos.

Durante inferencias largas, el resumen consolidado también se guarda en
`notes/contexto_importante.md` antes de compactar el contexto. Así los datos
importantes sobreviven aunque el prompt se reduzca. Durante el onboarding, el
instalador pregunta si Aether debe tener su propia carpeta. Si elegís que no,
usa `<proyecto>/.aether-data/` para la base y sus notas.

### Memoria central compartida

Además del almacenamiento de sesión anterior, Aether tiene una **memoria
central compartida** en JSON (`core/memory/central/`) que vive fuera de
cualquier runtime (default `~/.aether/memory/`, override con
`AETHER_CENTRAL_MEMORY_PATH`) y es la misma para todos: terminal, Web UI,
Roblox Player y futuros runtimes. Tiene tres capas:

- **Conversación**: contexto temporal por sesión, compactable.
- **Usuario**: hechos estables del usuario; `user_set()` nunca sobrescribe
  silenciosamente, las correcciones pasan por `user_update()` con historial.
- **Aprendizaje**: recuerdos con `importance` (1–10) y `strength` (0–1).
  El uso los refuerza, el desuso los debilita (`consolidate()`), y
  `forget()` los hace inaccesibles **sin borrarlos** (solo `hard=True`
  elimina). La personalidad es acumulativa: `personality_signals()` agrega
  patrones, nunca reemplaza nada.

El context builder inyecta los aprendizajes relevantes en el slot
`[APRENDIZAJES]` (desactivable con `AETHER_CENTRAL_MEMORY=0`).

## Arquitectura del proyecto

```text
.
├── run.py                    # Entrada principal
├── install.sh                # Instalador y perfilado de hardware
├── install-core.sh           # Implementación interna del instalador
├── bin/                      # Lanzador global aether
├── core/
│   ├── agent/                # Grafo, planner, loop y tools
│   ├── config/               # Configuración y autorización de directorios
│   ├── memory/               # Memoria, compactación y SQLite
│   ├── services/             # Entrada única al grafo
│   └── tools/                # Shell, web, visión y filesystem
├── tui/                      # Interfaz Textual y estados
├── skills/                   # Instrucciones reutilizables
├── tests/                    # Tests automatizados
└── backend/                  # API FastAPI experimental, no instalada por defecto
```

Las skills operativas viven en `skills/<nombre>/SKILL.md`. Se mantienen como
archivos separados porque Aether los descubre y carga dinámicamente; no son
documentación redundante.

## Backend opcional

El backend FastAPI es experimental y **no forma parte de la instalación normal
de la TUI**. Está separado para evitar que sus dependencias antiguas interfieran
con el runtime actual. La futura WebUI se desarrollará en un repositorio aparte.

Si necesitás trabajar con el backend experimental manualmente:

```bash
cd backend
../crewai-env/bin/python -m pip install -r requirements.txt
../crewai-env/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Desarrollo y pruebas

El instalador normal ya crea y configura el entorno virtual. Para desarrollo:

```bash
source crewai-env/bin/activate
python -m pytest tests
python -m py_compile run.py tui/app.py core/agent/graph_nodes.py
```

### Runtime de Roblox

La automatización específica de Roblox vive en el repositorio separado
`~/aether-roblox`. Aether conserva las primitivas genéricas de control y se
conecta al runtime mediante `core.tools.roblox_bridge`.

Desde la TUI se controla todo con un único comando:

```text
/play-roblox
/play-roblox google
/play-roblox status
/play-roblox stop
```

El arranque busca y enfoca `org.vinegarhq.Sober` antes de lanzar el runtime.
Por defecto usa Ollama; `/play-roblox google` selecciona Google Cloud Vision
para ese arranque. La credencial debe existir en el entorno, sin guardarla en
`config.json`:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/application_default_credentials.json"
# o: export GOOGLE_API_KEY="..."
```

Instalá el extra del runtime una vez:

```bash
cd ~/aether-roblox
venv/bin/pip install -e '.[google]'
```

El runtime coordina percepción, reacción, movimiento WASD y cámara para evitar
que varios loops compitan por el foco o los dispositivos de entrada.

Para comprobar que el grafo compila:

```bash
python -c "from core.agent.graph_builder import build_graph; build_graph(); print('ok')"
```

El script de auditoría está en [`tools/audit_project.py`](./tools/audit_project.py)
y sirve como diagnóstico manual; no forma parte del arranque normal.

## Solución de problemas

### `aether` no aparece después de instalar

El instalador agrega `~/.local/bin` al `PATH`. Si ya había una terminal abierta
durante la instalación, reiniciá esa terminal para que herede el entorno nuevo.

### Ollama no responde

```bash
ollama serve
ollama list
```

En una instalación normal, el instalador ya verifica Ollama y descarga el modelo
seleccionado cuando lo autorizás.

### El modelo responde lento

Reducí `NUM_CTX` o `NUM_PREDICT`, comprobá la VRAM disponible y verificá que
Ollama esté usando GPU. El hardware y el modelo son los límites principales;
Aether evita trabajo redundante, compacta contexto y limita loops, pero no
puede acelerar la generación intrínseca del modelo.

### Directorio rechazado

El rechazo cierra esa ejecución y no queda persistido. Iniciá Aether nuevamente
para autorizarlo cuando quieras.

### Visión no funciona

Verificá `grim`, Wayland/Hyprland y que el modelo configurado soporte imágenes:

```bash
command -v grim
ollama show "$(python -c 'import json; print(json.load(open("core/config/config.json"))["MODELO"])')"
```

## Licencia y estado

Proyecto en desarrollo activo. Revisá los cambios directamente en Git y no
comprometas credenciales, archivos `.env`, bases de datos personales ni
configuraciones locales.
