#!/usr/bin/env bash
# Aether Unified Installer — v5
# Instala el stack necesario de Aether desde un único punto:
# sistema + Docker + SearXNG + FFmpeg/Whisper + Python + Ollama + modelo.
#
# Principios de seguridad:
#   - config.json es versionado y nunca se modifica.
#   - ajustes de hardware/usuario viven en config.local.json (ignorado por Git).
#   - cambios locales de config.json de instalaciones antiguas se migran.
#   - cambios locales de otros archivos versionados bloquean el update.
#   - config.local.json se escribe de forma atómica.
set -euo pipefail

REPO_URL="https://github.com/Thomi33/AetherAI.git"
INSTALL_DIR="${AETHER_DIR:-$HOME/AetherAI}"
BRANCH="${AETHER_BRANCH:-main}"
LAUNCH_DEST="${AETHER_BIN_DIR:-$HOME/.local/bin}"
SEARXNG_DIR="${AETHER_SEARXNG_DIR:-$HOME/.local/share/aether/searxng}"
SEARXNG_PORT="${AETHER_SEARXNG_PORT:-}"
LOW_SPEC=0; MINIMAL=0; NO_SYSTEM=0; NO_OLLAMA=0; ASSUME_YES=0; WANT_MODEL=""; TIER_REQ="${AETHER_TIER:-}"; WANT_SUITE=""
WEB_UI="ask"
PREV=""
MIGRATION_FILE=""
MIGRATION_BASE_FILE=""

for arg in "$@"; do
  if [ "$PREV" = "--model" ]; then WANT_MODEL="$arg"; PREV=""; continue; fi
  if [ "$PREV" = "--tier" ]; then TIER_REQ="$arg"; PREV=""; continue; fi
  case "$arg" in
    --low-spec) LOW_SPEC=1 ;;
    --minimal) MINIMAL=1 ;;
    --no-system) NO_SYSTEM=1 ;;
    --no-ollama) NO_OLLAMA=1 ;;
    --web-ui) WEB_UI=1 ;;
    --no-web-ui) WEB_UI=0 ;;
    --model) PREV="--model" ;;
    --model=*) WANT_MODEL="${arg#--model=}" ;;
    --tier) PREV="--tier" ;;
    --tier=*) TIER_REQ="${arg#--tier=}" ;;
    --suite|--agent-suite) WANT_SUITE="1" ;;
    --no-suite) WANT_SUITE="0" ;;
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# //'; exit 0 ;;
  esac
done

log()  { printf '%s\n' "$*"; }
ok()   { printf '   ✔ %s\n' "$*"; }
warn() { printf '   ⚠ %s\n' "$*" >&2; }
die()  { printf '❌ %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
cleanup() {
  [ -n "${MIGRATION_FILE:-}" ] && [ -f "$MIGRATION_FILE" ] && rm -f -- "$MIGRATION_FILE"
  [ -n "${MIGRATION_BASE_FILE:-}" ] && [ -f "$MIGRATION_BASE_FILE" ] && rm -f -- "$MIGRATION_BASE_FILE"
}
trap cleanup EXIT

ask_yes() {
  if [ "$ASSUME_YES" = "1" ]; then return 0; fi
  read -r -p "$1 [S/n] " r || return 1
  case "$r" in ""|[SsYy]*) return 0 ;; *) return 1 ;; esac
}

suite_models() {
  PROBE_JSON="$PROBE_JSON" python3 -c 'import json,os; print("\n".join((json.loads(os.environ.get("PROBE_JSON") or "{}").get("recomendado") or {}).get("SUITE_MODELOS") or []))'
}
suite_available() {
  PROBE_JSON="$PROBE_JSON" python3 -c 'import json,os; print("1" if (json.loads(os.environ.get("PROBE_JSON") or "{}").get("recomendado") or {}).get("SUITE_AGENTICA_DISPONIBLE") else "0")'
}
choose_suite() {
  [ "$MINIMAL" = "1" ] && { WANT_SUITE=0; return; }
  [ "$NO_OLLAMA" = "1" ] && { WANT_SUITE=0; return; }
  [ "$WANT_SUITE" != "" ] && return
  if [ "$(suite_available)" = "1" ]; then
    echo
    echo "🧠 Suite agentica disponible para $TIER:"
    suite_models | sed 's/^/   • /'
    echo "   Los modelos se descargan, pero Aether no los carga todos simultáneamente."
    if ask_yes "¿Querés instalar la suite completa de modelos agenticos?"; then WANT_SUITE=1; else WANT_SUITE=0; fi
  else
    WANT_SUITE=0
  fi
}

# ---------------------------------------------------------------------------
# Repository safety / config migration
# ---------------------------------------------------------------------------
prepare_git_update() {
  local repo="$1" config="$1/core/config/config.json" status line
  [ -d "$repo/.git" ] || return 0
  cd "$repo"
  status="$(git status --porcelain --untracked-files=all)"
  [ -z "$status" ] && { git fetch origin "$BRANCH" --depth 1 || die "No pude obtener la rama $BRANCH."; git checkout "$BRANCH" || die "No pude cambiar a la rama $BRANCH."; git pull --rebase origin "$BRANCH" || die "git pull/rebase falló."; return; }

  while IFS= read -r line; do
    [ -n "$line" ] || continue
    case "$line" in
      " M core/config/config.json"|"M  core/config/config.json"|"MM core/config/config.json" ) : ;;
      "?? core/config/config.local.json"|"?? core/config/config.local.json.bak"|"?? core/config/config.local.json.tmp" ) : ;;
      *) die "Cambios locales detectados fuera de config.json; no actualizo para no sobrescribir trabajo: $line" ;;
    esac
  done <<< "$status"

  if git status --porcelain -- core/config/config.json | grep -q .; then
    [ -f "$config" ] || die "config.json aparece modificado pero no existe."
    MIGRATION_FILE="$(mktemp "${TMPDIR:-/tmp}/aether-config-migration.XXXXXX.json")"
    MIGRATION_BASE_FILE="$(mktemp "${TMPDIR:-/tmp}/aether-config-base.XXXXXX.json")"
    cp -- "$config" "$MIGRATION_FILE"
    git show HEAD:core/config/config.json > "$MIGRATION_BASE_FILE" || die "No pude recuperar el config.json base anterior."
    git restore --source=HEAD -- core/config/config.json
    ok "Configuración antigua preservada para migración local."
  fi

  git fetch origin "$BRANCH" --depth 1 || die "No pude obtener la rama $BRANCH."
  git checkout "$BRANCH" || die "No pude cambiar a la rama $BRANCH."
  git pull --rebase origin "$BRANCH" || die "git pull/rebase falló; no continúo en estado parcial."
}

migrate_old_config() {
  [ -n "${MIGRATION_FILE:-}" ] || return 0
  local local_cfg="$INSTALL_DIR/core/config/config.local.json"
  mkdir -p "$(dirname "$local_cfg")"
  MIGRATION_FILE="$MIGRATION_FILE" BASE_FILE="$MIGRATION_BASE_FILE" LOCAL_FILE="$local_cfg" python3 - <<'PY'
import json, os
from pathlib import Path
old = json.loads(Path(os.environ["MIGRATION_FILE"]).read_text(encoding="utf-8"))
base = json.loads(Path(os.environ["BASE_FILE"]).read_text(encoding="utf-8"))
local_path = Path(os.environ["LOCAL_FILE"])
local = json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
if not isinstance(local, dict):
    raise SystemExit("config.local.json debe contener un objeto JSON")
for key, value in old.items():
    if base.get(key) != value:
        local[key] = value
tmp = local_path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(local, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
json.loads(tmp.read_text(encoding="utf-8"))
tmp.replace(local_path)
PY
  ok "Configuración local migrada a core/config/config.local.json."
  rm -f -- "$MIGRATION_FILE" "$MIGRATION_BASE_FILE"
  MIGRATION_FILE=""; MIGRATION_BASE_FILE=""
}

write_local_config() {
  local local_cfg="$INSTALL_DIR/core/config/config.local.json"
  mkdir -p "$(dirname "$local_cfg")"
  PROBE_JSON="$PROBE_JSON" WANT_MODEL="$WANT_MODEL" LOCAL_FILE="$local_cfg" "$VENV_DIR/bin/python" - <<'PY'
import json, os
from pathlib import Path
probe = json.loads(os.environ.get("PROBE_JSON") or "{}")
rec = dict(probe.get("recomendado") or {})
if os.environ.get("WANT_MODEL"):
    rec["MODELO"] = os.environ["WANT_MODEL"]
path = Path(os.environ["LOCAL_FILE"])
local = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
if not isinstance(local, dict):
    raise SystemExit("config.local.json debe contener un objeto JSON")
for key in ("MODELO","MODELO_VISION","NUM_CTX","NUM_PREDICT","NUM_PREDICT_PLANNER","MAX_TURNOS_CONTEXTO_CHAT","TIMEOUT_CMD","OLLAMA_KEEP_ALIVE","OLLAMA_NUM_PARALLEL","OLLAMA_MAX_LOADED_MODELS"):
    if rec.get(key) is not None:
        local[key] = rec[key]
gen = dict(local.get("OLLAMA_GEN_OPTIONS") or {})
for key in ("num_batch","num_thread","num_gpu"):
    if (rec.get("OLLAMA_GEN_OPTIONS") or {}).get(key) is not None:
        gen[key] = rec["OLLAMA_GEN_OPTIONS"][key]
if gen:
    local["OLLAMA_GEN_OPTIONS"] = gen
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(local, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
json.loads(tmp.read_text(encoding="utf-8"))
tmp.replace(path)
PY
  ok "Perfil local de hardware escrito de forma atómica."
}

assert_repository_integrity() {
  git diff --exit-code -- core/config/config.json >/dev/null || die "INTEGRITY CHECK: el instalador modificó config.json versionado."
  git check-ignore -q core/config/config.local.json || die "INTEGRITY CHECK: config.local.json no está protegido por .gitignore."
  ok "Repository integrity: config.json intacto y configuración local ignorada."
}

echo "🚀 Aether Unified Installer (v5)"
echo "📦 Target: $INSTALL_DIR"
echo "🌿 Branch: $BRANCH"

# ---------------------------------------------------------------------------
# 0. Hardware probe
# ---------------------------------------------------------------------------
PROBE_JSON=""; TIER="LOW"
run_probe() {
  local script=""
  [ -f "$INSTALL_DIR/tools/hardware_probe.py" ] && script="$INSTALL_DIR/tools/hardware_probe.py"
  [ -z "$script" ] && [ -f "$(dirname "$0")/tools/hardware_probe.py" ] && script="$(dirname "$0")/tools/hardware_probe.py"
  if [ -n "$script" ]; then PROBE_JSON=$(python3 "$script" ${WANT_MODEL:+--model-override "$WANT_MODEL"} 2>/dev/null) || PROBE_JSON=""; fi
  if [ -z "$PROBE_JSON" ]; then
    local mem cpu tier_fb
    mem=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo 2>/dev/null || echo 0) / 1024 / 1024 ))
    cpu=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 || echo "?")
    if { [ "$mem" -ne 0 ] && [ "$mem" -le 10 ]; } || echo "$cpu" | grep -qiE 'celeron|atom|pentium|n4500|n4020'; then tier_fb="LOW"; else tier_fb="HIGH"; fi
    PROBE_JSON=$(python3 -c 'import json,sys; print(json.dumps({"tier":sys.argv[1],"cpu":sys.argv[2],"ram_gb":0,"vram_gb":0,"usable_gb":0,"recomendado":{}}))' "$tier_fb" "$cpu")
  fi
  TIER=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tier","LOW"))' <<<"$PROBE_JSON" 2>/dev/null || echo LOW)
}
run_probe
[ -n "$TIER_REQ" ] && TIER="$(echo "$TIER_REQ" | tr '[:lower:]' '[:upper:]')"
[ "$LOW_SPEC" = "1" ] && [ "$TIER" != "POTATO" ] && TIER="LOW"
echo "🧬 Tier detectado: $TIER"
python3 -c 'import json,sys;r=json.load(sys.stdin);print("   CPU: "+str(r.get("cpu"))+" | RAM: "+str(r.get("ram_gb"))+"GB | VRAM: "+str(r.get("vram_gb"))+"GB")' <<<"$PROBE_JSON" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 1. System dependencies
# ---------------------------------------------------------------------------
OS_ID="unknown"
[ -f /etc/os-release ] && . /etc/os-release && OS_ID="${ID:-unknown}"
IS_ARCH=0; IS_DEBIAN=0; IS_FEDORA=0
case "$OS_ID" in
  arch|endeavouros|manjaro|cachyos) IS_ARCH=1 ;;
  debian|ubuntu|linuxmint|pop) IS_DEBIAN=1 ;;
  fedora|nobara|rhel) IS_FEDORA=1 ;;
esac
SUDO=""; [ "$(id -u)" != "0" ] && SUDO="sudo"
if [ "$NO_SYSTEM" = "0" ]; then
  [ -z "$SUDO" ] || have sudo || die "Necesito sudo para instalar dependencias del sistema."
  echo "📦 Instalando dependencias del sistema..."
  if [ "$IS_ARCH" = "1" ] && have pacman; then
    ARCH_PACKAGES=(git base-devel python python-pip python-virtualenv sqlite curl ffmpeg docker docker-compose wl-clipboard grim slurp ydotool)
    have ollama || ARCH_PACKAGES+=(ollama)
    $SUDO pacman -Sy --needed --noconfirm "${ARCH_PACKAGES[@]}" 2>&1 | tail -8 || die "Falló la instalación de dependencias con pacman."
  elif [ "$IS_DEBIAN" = "1" ] && have apt-get; then
    $SUDO apt-get update -y >/dev/null
    $SUDO apt-get install -y git build-essential python3 python3-venv python3-pip sqlite3 curl ffmpeg docker.io docker-compose-plugin wl-clipboard grim slurp 2>&1 | tail -8 || die "Falló la instalación de dependencias con apt."
  elif [ "$IS_FEDORA" = "1" ] && have dnf; then
    $SUDO dnf install -y git gcc gcc-c++ make python3 python3-pip sqlite curl ffmpeg docker docker-compose-plugin wl-clipboard grim slurp 2>&1 | tail -8 || die "Falló la instalación de dependencias con dnf."
  else
    warn "Distro '$OS_ID' no reconocida."
    have git || die "git no está instalado."; have python3 || die "python3 no está instalado."; have curl || die "curl no está instalado."
  fi
else
  echo "⏭️  --no-system: no instalo paquetes del sistema."
fi
command -v git >/dev/null 2>&1 || die "git requerido"
command -v python3 >/dev/null 2>&1 || die "python3 requerido"
python3 -c 'import sys; assert sys.version_info >= (3,10)' || die "Python >= 3.10 requerido."

# ---------------------------------------------------------------------------
# 2. Docker + SearXNG (local-only)
# ---------------------------------------------------------------------------
setup_searxng() {
  [ "$NO_SYSTEM" = "1" ] && { warn "--no-system: no puedo garantizar Docker/SearXNG."; return 0; }
  have docker || die "Docker no está disponible."
  if have systemctl; then
    $SUDO systemctl enable --now docker.service 2>/dev/null || $SUDO systemctl enable --now docker.socket 2>/dev/null || warn "No pude iniciar Docker automáticamente."
  fi
  local compose_cmd=""
  if docker compose version >/dev/null 2>&1; then compose_cmd="docker compose"; elif have docker-compose && docker-compose version >/dev/null 2>&1; then compose_cmd="docker-compose"; else die "Docker Compose no está disponible."; fi
  if [ -z "$SEARXNG_PORT" ]; then
    if [ "$ASSUME_YES" = "1" ]; then SEARXNG_PORT=8080; else read -r -p "🔎 Puerto para SearXNG [8080]: " INPUT_PORT; SEARXNG_PORT="${INPUT_PORT:-8080}"; fi
  fi
  [[ "$SEARXNG_PORT" =~ ^[0-9]+$ ]] || die "Puerto SearXNG inválido: $SEARXNG_PORT"
  [ "$SEARXNG_PORT" -ge 1 ] && [ "$SEARXNG_PORT" -le 65535 ] || die "Puerto SearXNG fuera de rango: $SEARXNG_PORT"
  mkdir -p "$SEARXNG_DIR/core-config"; cd "$SEARXNG_DIR"
  curl -fsSL -o docker-compose.yml https://raw.githubusercontent.com/searxng/searxng/master/container/docker-compose.yml
  curl -fsSL -o .env.example https://raw.githubusercontent.com/searxng/searxng/master/container/.env.example
  [ -f .env ] || cp .env.example .env
  if command -v ss >/dev/null 2>&1 && ss -ltn 2>/dev/null | grep -qE ":${SEARXNG_PORT}[[:space:]]"; then
    if curl -fsS --max-time 2 "http://127.0.0.1:${SEARXNG_PORT}/" >/dev/null 2>&1; then ok "SearXNG ya responde en http://127.0.0.1:${SEARXNG_PORT}"; cd - >/dev/null; return 0; fi
    die "El puerto $SEARXNG_PORT ya está ocupado. Elegí otro con AETHER_SEARXNG_PORT=XXXX."
  fi
  if grep -q '^SEARXNG_HOST=' .env; then sed -i 's/^SEARXNG_HOST=.*/SEARXNG_HOST=127.0.0.1/' .env; else printf '\nSEARXNG_HOST=127.0.0.1\n' >> .env; fi
  if grep -q '^SEARXNG_PORT=' .env; then sed -i "s/^SEARXNG_PORT=.*/SEARXNG_PORT=$SEARXNG_PORT/" .env; else printf 'SEARXNG_PORT=%s\n' "$SEARXNG_PORT" >> .env; fi
  echo "🔎 Levantando SearXNG en http://127.0.0.1:$SEARXNG_PORT ..."
  $SUDO $compose_cmd up -d
  local i
  for i in $(seq 1 30); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${SEARXNG_PORT}/" >/dev/null 2>&1; then ok "SearXNG listo: http://127.0.0.1:$SEARXNG_PORT"; cd - >/dev/null; return 0; fi
    sleep 2
  done
  $SUDO $compose_cmd logs --tail=50 core || true; cd - >/dev/null; die "SearXNG no respondió a tiempo."
}
setup_searxng

# ---------------------------------------------------------------------------
# 3. Clone / update Aether safely
# ---------------------------------------------------------------------------
if [ ! -d "$INSTALL_DIR" ]; then
  echo "📥 Clonando Aether..."; git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" || die "No pude clonar Aether."
elif git -C "$INSTALL_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  echo "🔄 Actualizando Aether..."; prepare_git_update "$INSTALL_DIR"
else
  warn "$INSTALL_DIR existe pero no es un repo Git; continúo con lo local."
fi
cd "$INSTALL_DIR"
migrate_old_config

# ---------------------------------------------------------------------------
# 4. Python venv + Aether dependencies + Whisper
# ---------------------------------------------------------------------------
echo "🧠 Configurando entorno Python..."
export MAKEFLAGS="${MAKEFLAGS:--j2}"; export PIP_DISABLE_PIP_VERSION_CHECK=1
PYBIN="${AETHER_PYTHON:-python3}"; VENV_DIR="${AETHER_VENV_DIR:-.venv}"
[ -d "$VENV_DIR" ] || "$PYBIN" -m venv "$VENV_DIR" || die "No pude crear el venv."
source "$VENV_DIR/bin/activate"; pip install --upgrade "pip<26" >/dev/null
pip_one() { [ -f "$1" ] || return 0; echo "📚 $1..."; if [ "$TIER" = "POTATO" ] || [ "$TIER" = "LOW" ] || [ "$LOW_SPEC" = "1" ]; then pip install --prefer-binary -r "$1" 2>&1 | tail -5; else pip install -r "$1" 2>&1 | tail -5; fi; }
pip_one requirements.txt
if ! python -c 'import whisper' >/dev/null 2>&1; then echo "🎙️ Instalando OpenAI Whisper..."; pip install --prefer-binary -U openai-whisper 2>&1 | tail -8 || warn "Whisper no pudo instalarse; revisá el error anterior."; fi
have ffmpeg && ok "FFmpeg disponible para Whisper." || warn "FFmpeg no disponible; Whisper no podrá procesar audio."

# ---------------------------------------------------------------------------
# 5. .env + hardware tuning → LOCAL CONFIG ONLY
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then printf 'AETHER_MODE=production\nLOG_LEVEL=info\nAETHER_DATA_DIR=%s\nSEARXNG_URL=http://127.0.0.1:%s\n' "$HOME/Aether" "$SEARXNG_PORT" > .env; else grep -q '^SEARXNG_URL=' .env && sed -i "s|^SEARXNG_URL=.*|SEARXNG_URL=http://127.0.0.1:$SEARXNG_PORT|" .env || printf 'SEARXNG_URL=http://127.0.0.1:%s\n' "$SEARXNG_PORT" >> .env; fi
if [ -f tools/hardware_probe.py ]; then
  if [ -n "$TIER_REQ" ]; then PROBE_JSON=$(python tools/hardware_probe.py --force-tier "$(echo "$TIER_REQ" | tr '[:lower:]' '[:upper:]')" ${WANT_MODEL:+--model-override "$WANT_MODEL"} 2>/dev/null || echo "$PROBE_JSON"); TIER="$(echo "$TIER_REQ" | tr '[:lower:]' '[:upper:]')"; else PROBE_JSON=$(python tools/hardware_probe.py ${WANT_MODEL:+--model-override "$WANT_MODEL"} 2>/dev/null || echo "$PROBE_JSON"); TIER=$(python -c 'import json,sys;print(json.load(sys.stdin).get("tier","LOW"))' <<<"$PROBE_JSON" 2>/dev/null || echo "$TIER"); fi
fi
[ "$LOW_SPEC" = "1" ] && TIER=LOW
[ -z "$WANT_MODEL" ] && WANT_MODEL="$(python -c 'import json,sys;print(json.load(sys.stdin).get("recomendado",{}).get("MODELO",""))' <<<"$PROBE_JSON" 2>/dev/null || true)"
write_local_config
choose_suite

# ---------------------------------------------------------------------------
# 6. Ollama + model(s)
# ---------------------------------------------------------------------------
if [ "$NO_OLLAMA" = "1" ] || [ "$MINIMAL" = "1" ]; then
  echo "⏭️ Ollama/modelos omitidos."
elif have ollama; then
  if ! ollama list >/dev/null 2>&1; then echo "🦙 Levantando Ollama..."; if have systemctl; then systemctl --user enable --now ollama 2>/dev/null || $SUDO systemctl enable --now ollama 2>/dev/null || true; fi; ollama serve >/tmp/ollama-serve.log 2>&1 & sleep 3; fi
  if ollama list >/dev/null 2>&1; then
    if [ -z "$WANT_MODEL" ]; then WANT_MODEL="$(python -c 'import json,sys;print(json.load(sys.stdin).get("recomendado",{}).get("MODELO",""))' <<<"$PROBE_JSON" 2>/dev/null || true)"; fi
    [ -n "$WANT_MODEL" ] || { [ "$TIER" = POTATO ] && WANT_MODEL=qwen3:0.6b || WANT_MODEL=qwen3.5:2b; }
    if ollama list | grep -qiF "$WANT_MODEL"; then ok "Modelo primario $WANT_MODEL presente."; elif ask_yes "¿Descargar modelo primario '$WANT_MODEL'?"; then ollama pull "$WANT_MODEL" || warn "No pude descargar $WANT_MODEL."; fi
    if [ "$WANT_SUITE" = "1" ]; then
      echo "🧠 Descargando suite agentica..."
      suite_models | while IFS= read -r model; do
        [ -n "$model" ] || continue; [ "$model" = "$WANT_MODEL" ] && continue
        if ollama list | grep -qiF "$model"; then ok "Suite: $model ya está presente."; else echo "   ↓ $model"; ollama pull "$model" || warn "No pude descargar $model; continúo con la suite."; fi
      done
    fi
  else warn "Ollama no responde."; fi
else
  warn "Ollama no está instalado."
fi

# ---------------------------------------------------------------------------
# 7. Validation + launcher + repository integrity
# ---------------------------------------------------------------------------
echo "🧪 Validando instalación..."
python -c 'import langgraph, mcp; print("   ✔ Python core OK")' 2>/dev/null || warn "Algunas dependencias Python del core faltan."
python -c 'import whisper; print("   ✔ Whisper OK")' 2>/dev/null || warn "Whisper no está disponible."
mkdir -p "$LAUNCH_DEST"
[ -x "$PWD/bin/aether" ] && { ln -sf "$PWD/bin/aether" "$LAUNCH_DEST/aether"; ok "aether disponible en $LAUNCH_DEST"; }
assert_repository_integrity

# ---------------------------------------------------------------------------
# 8. Web UI (opcional — repo separado, la API la sirve automáticamente)
# ---------------------------------------------------------------------------
WEBUI_DIR="${AETHER_WEBUI_DIR:-$HOME/aether_web_ui}"
WEBUI_REPO="https://github.com/Thomi33/aether_web_ui.git"
case "$WEB_UI" in
  1) : ;;
  0) : ;;
  *) ask_yes "¿Instalar la Web UI de Aether? (interfaz web del agente, repo separado)" && WEB_UI=1 || WEB_UI=0 ;;
esac
if [ "$WEB_UI" = "1" ]; then
  have git || die "git requerido para clonar la Web UI."
  if [ -d "$WEBUI_DIR/.git" ]; then
    git -C "$WEBUI_DIR" pull --ff-only >/dev/null 2>&1 \
      && ok "Web UI actualizada en $WEBUI_DIR" \
      || warn "No pude actualizar la Web UI (existe en $WEBUI_DIR; revisala a mano)."
  elif git clone --depth 1 "$WEBUI_REPO" "$WEBUI_DIR" >/dev/null 2>&1; then
    ok "Web UI instalada en $WEBUI_DIR"
  else
    warn "No pude clonar la Web UI. Podés hacerlo a mano: git clone $WEBUI_REPO ~/aether_web_ui"
    WEB_UI=0
  fi
  [ "$WEB_UI" = "1" ] && log "   → Levantala con: aether web"
else
  log "⏭️  Web UI omitida (instalable después: git clone $WEBUI_REPO ~/aether_web_ui)"
fi
cat <<EOF

✅ Aether instalado correctamente
   Tier: $TIER
   Modelo primario: ${WANT_MODEL:-?}
   Suite agentica: $([ "$WANT_SUITE" = "1" ] && echo "instalada" || echo "solo modelo primario")
   SearXNG: http://127.0.0.1:$SEARXNG_PORT
   Whisper: $(python -c 'import whisper; print("OK")' 2>/dev/null || echo "pendiente")
   Config base: core/config/config.json (versionada, protegida)
   Config local: core/config/config.local.json (hardware/usuario, ignorada por Git)

👉 Ejecutá: aether
EOF
[ "$WEB_UI" = "1" ] && echo "🖥️  Web UI: aether web  (o aether server / aether gateway)"
