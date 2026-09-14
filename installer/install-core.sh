#!/usr/bin/env bash
# Aether Unified Installer — v4
# Instala el stack necesario de Aether desde un único punto:
# sistema + Docker + SearXNG + FFmpeg/Whisper + Python + Ollama + modelo.
set -euo pipefail

REPO_URL="https://github.com/Thomi33/AetherAI.git"
INSTALL_DIR="${AETHER_DIR:-$HOME/AetherAI}"
BRANCH="${AETHER_BRANCH:-main}"
LAUNCH_DEST="${AETHER_BIN_DIR:-$HOME/.local/bin}"
SEARXNG_DIR="${AETHER_SEARXNG_DIR:-$HOME/.local/share/aether/searxng}"
SEARXNG_PORT="${AETHER_SEARXNG_PORT:-}"
LOW_SPEC=0; MINIMAL=0; NO_SYSTEM=0; NO_OLLAMA=0; ASSUME_YES=0; WANT_MODEL=""; TIER_REQ="${AETHER_TIER:-}"; WANT_SUITE=""
PREV=""

for arg in "$@"; do
  if [ "$PREV" = "--model" ]; then WANT_MODEL="$arg"; PREV=""; continue; fi
  if [ "$PREV" = "--tier" ]; then TIER_REQ="$arg"; PREV=""; continue; fi
  case "$arg" in
    --low-spec) LOW_SPEC=1 ;;
    --minimal) MINIMAL=1 ;;
    --no-system) NO_SYSTEM=1 ;;
    --no-ollama) NO_OLLAMA=1 ;;
    --model) PREV="--model" ;;
    --model=*) WANT_MODEL="${arg#--model=}" ;;
    --tier) PREV="--tier" ;;
    --tier=*) TIER_REQ="${arg#--tier=}" ;;
    --suite|--agent-suite) WANT_SUITE="1" ;;
    --no-suite) WANT_SUITE="0" ;;
    -y|--yes) ASSUME_YES=1 ;;
    -h|--help) sed -n '2,8p' "$0" | sed 's/^# //'; exit 0 ;;
  esac
done

log()  { printf '%s\n' "$*"; }
ok()   { printf '   ✔ %s\n' "$*"; }
warn() { printf '   ⚠ %s\n' "$*" >&2; }
die()  { printf '❌ %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
ask_yes() {
  if [ "$ASSUME_YES" = "1" ]; then return 0; fi
  read -r -p "$1 [S/n] " r || return 1
  case "$r" in ""|[SsYy]*) return 0 ;; *) return 1 ;; esac
}

# Reads the optional model suite from the probe without requiring jq.
suite_models() {
  python3 - <<'PY' <<<"$PROBE_JSON"
import json,sys
try:
    data=json.load(sys.stdin)
    models=(data.get("recomendado") or {}).get("SUITE_MODELOS") or []
    print("\n".join(models))
except Exception:
    pass
PY
}
suite_available() {
  python3 - <<'PY' <<<"$PROBE_JSON"
import json,sys
try:
    print("1" if (json.load(sys.stdin).get("recomendado") or {}).get("SUITE_AGENTICA_DISPONIBLE") else "0")
except Exception:
    print("0")
PY
}

# Default to installing only the primary model. For MID/HIGH/ULTRA the
# installer asks whether the user wants the complete local agentic suite.
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

echo "🚀 Aether Unified Installer (v4)"
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
    if have ollama; then
      ok "Ollama ya está instalado ($(ollama --version 2>/dev/null || echo 'versión desconocida'))."
      echo "   → Reutilizando instalación existente; no se instala el paquete de pacman."
    else
      ARCH_PACKAGES+=(ollama)
    fi
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
# 3. Clone / update Aether
# ---------------------------------------------------------------------------
if [ ! -d "$INSTALL_DIR" ]; then
  echo "📥 Clonando Aether..."; git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
else
  echo "🔄 Actualizando Aether..."; cd "$INSTALL_DIR"
  if git rev-parse --git-dir >/dev/null 2>&1; then git fetch origin "$BRANCH" --depth 1 2>/dev/null || true; git checkout "$BRANCH" 2>/dev/null || true; git pull --rebase origin "$BRANCH" || warn "git pull falló; continúo con lo local."; else warn "$INSTALL_DIR no es un repo git; continúo."; fi
fi
cd "$INSTALL_DIR"

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
# 5. .env + hardware tuning
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then printf 'AETHER_MODE=production\nLOG_LEVEL=info\nAETHER_DATA_DIR=%s\nSEARXNG_URL=http://127.0.0.1:%s\n' "$HOME/Aether" "$SEARXNG_PORT" > .env; else grep -q '^SEARXNG_URL=' .env && sed -i "s|^SEARXNG_URL=.*|SEARXNG_URL=http://127.0.0.1:$SEARXNG_PORT|" .env || printf 'SEARXNG_URL=http://127.0.0.1:%s\n' "$SEARXNG_PORT" >> .env; fi
if [ -f tools/hardware_probe.py ]; then
  if [ -n "$TIER_REQ" ]; then PROBE_JSON=$(python tools/hardware_probe.py --force-tier "$(echo "$TIER_REQ" | tr '[:lower:]' '[:upper:]')" ${WANT_MODEL:+--model-override "$WANT_MODEL"} 2>/dev/null || echo "$PROBE_JSON"); TIER="$(echo "$TIER_REQ" | tr '[:lower:]' '[:upper:]')"; else PROBE_JSON=$(python tools/hardware_probe.py ${WANT_MODEL:+--model-override "$WANT_MODEL"} 2>/dev/null || echo "$PROBE_JSON"); TIER=$(python -c 'import json,sys;print(json.load(sys.stdin).get("tier","LOW"))' <<<"$PROBE_JSON" 2>/dev/null || echo "$TIER"); fi
fi
[ "$LOW_SPEC" = "1" ] && TIER=LOW
[ -z "$WANT_MODEL" ] && WANT_MODEL="$(python -c 'import json,sys;print(json.load(sys.stdin).get("recomendado",{}).get("MODELO",""))' <<<"$PROBE_JSON" 2>/dev/null || true)"
if [ -f core/config/config.json ] && [ -n "$WANT_MODEL" ]; then
  cp -n core/config/config.json core/config/config.json.bak 2>/dev/null || true
  PROBE_JSON="$PROBE_JSON" WANT_MODEL="$WANT_MODEL" "$VENV_DIR/bin/python" - core/config/config.json <<'PYEOF'
import json, os, sys
path=sys.argv[1]; probe=json.loads(os.environ.get("PROBE_JSON") or "{}"); rec=probe.get("recomendado") or {}; rec["MODELO"]=os.environ["WANT_MODEL"]; cfg=json.load(open(path))
for k in ("MODELO","MODELO_VISION","NUM_CTX","NUM_PREDICT","NUM_PREDICT_PLANNER","MAX_TURNOS_CONTEXTO_CHAT","TIMEOUT_CMD","OLLAMA_KEEP_ALIVE","OLLAMA_NUM_PARALLEL","OLLAMA_MAX_LOADED_MODELS"):
    if rec.get(k) is not None: cfg[k]=rec[k]
g=cfg.get("OLLAMA_GEN_OPTIONS",{}) or {}
for k in ("num_batch","num_thread","num_gpu"):
    if (rec.get("OLLAMA_GEN_OPTIONS") or {}).get(k) is not None: g[k]=rec["OLLAMA_GEN_OPTIONS"][k]
cfg["OLLAMA_GEN_OPTIONS"]=g; json.dump(cfg,open(path,"w"),indent=2,ensure_ascii=False)
PYEOF
fi
choose_suite

# ---------------------------------------------------------------------------
# 6. Ollama + model(s)
# ---------------------------------------------------------------------------
if [ "$NO_OLLAMA" = "1" ] || [ "$MINIMAL" = "1" ]; then echo "⏭️ Ollama/modelos omitidos."; elif have ollama; then
  if ! ollama list >/dev/null 2>&1; then echo "🦙 Levantando Ollama..."; if have systemctl; then systemctl --user enable --now ollama 2>/dev/null || $SUDO systemctl enable --now ollama 2>/dev/null || true; fi; ollama serve >/tmp/ollama-serve.log 2>&1 & sleep 3; fi
  if ollama list >/dev/null 2>&1; then
    if [ -z "$WANT_MODEL" ]; then WANT_MODEL="$(python -c 'import json,sys;print(json.load(sys.stdin).get("recomendado",{}).get("MODELO",""))' <<<"$PROBE_JSON" 2>/dev/null || true)"; fi
    [ -n "$WANT_MODEL" ] || { [ "$TIER" = POTATO ] && WANT_MODEL=qwen3:0.6b || WANT_MODEL=qwen3.5:2b; }
    if ollama list | grep -qiF "$WANT_MODEL"; then ok "Modelo primario $WANT_MODEL presente."; elif ask_yes "¿Descargar modelo primario '$WANT_MODEL'?"; then ollama pull "$WANT_MODEL" || warn "No pude descargar $WANT_MODEL."; fi

    if [ "$WANT_SUITE" = "1" ]; then
      echo "🧠 Descargando suite agentica..."
      suite_models | while IFS= read -r model; do
        [ -n "$model" ] || continue
        [ "$model" = "$WANT_MODEL" ] && continue
        if ollama list | grep -qiF "$model"; then
          ok "Suite: $model ya está presente."
        else
          echo "   ↓ $model"
          ollama pull "$model" || warn "No pude descargar $model; continúo con la suite."
        fi
      done
    fi
  else warn "Ollama no responde."; fi
else warn "Ollama no está instalado."; fi

# ---------------------------------------------------------------------------
# 7. Validation + launcher
# ---------------------------------------------------------------------------
echo "🧪 Validando instalación..."
python -c 'import langgraph, mcp; print("   ✔ Python core OK")' 2>/dev/null || warn "Algunas dependencias Python del core faltan."
python -c 'import whisper; print("   ✔ Whisper OK")' 2>/dev/null || warn "Whisper no está disponible."
mkdir -p "$LAUNCH_DEST"
[ -x "$PWD/bin/aether" ] && { ln -sf "$PWD/bin/aether" "$LAUNCH_DEST/aether"; ok "aether disponible en $LAUNCH_DEST"; }
cat <<EOF

✅ Aether instalado correctamente
   Tier: $TIER
   Modelo primario: ${WANT_MODEL:-?}
   Suite agentica: $([ "$WANT_SUITE" = "1" ] && echo "instalada" || echo "solo modelo primario")
   SearXNG: http://127.0.0.1:$SEARXNG_PORT
   Whisper: $(python -c 'import whisper; print("OK")' 2>/dev/null || echo "pendiente")

👉 Ejecutá: aether
EOF