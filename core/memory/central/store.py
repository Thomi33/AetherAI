"""
Memoria central compartida de Aether — almacenamiento JSON fuera de los runtimes.

Filosofía:
- La memoria NO pertenece a un runtime. Vive en un directorio propio
  (default ``~/.aether/memory/``, override con ``AETHER_CENTRAL_MEMORY_PATH``)
  y cualquier runtime (Web, Roblox, terminal, futuros) la comparte.
- Cuatro tipos de memoria:
    * conversación  — contexto temporal por sesión (compactable).
    * usuario       — hechos estables del usuario (sin sobrescritura silenciosa).
    * aprendizaje   — "diario" de Aether: recuerdos con importancia y fuerza.
    * intentos      — outcomes operativos: qué tool ya se probó y NO dio
                      resultado ("si no dio resultado, no dio resultado").
                      Efímeros por diseño: TTL corto, decaimiento diario y
                      barrido automático. No son aprendizajes.
- "Olvidar" NO es borrar: los recuerdos se debilitan (``strength`` baja) y
  dejan de aparecer en las búsquedas normales, pero siguen en disco y pueden
  reforzarse si vuelven a utilizarse.
- La personalidad es acumulativa: cada recuerdo es solo una señal pequeña;
  ``personality_signals()`` agrega patrones, nunca reemplaza nada.

Solo stdlib: este módulo lo importan runtimes con entornos distintos
(p. ej. aether-roblox lo carga desde su propio venv).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import date, datetime

_FORMAT_VERSION = 1
_WEAK_THRESHOLD = 0.15      # por debajo de esto un recuerdo no aparece en search()
_STRENGTH_FLOOR = 0.01      # un recuerdo nunca llega a 0: olvidar != borrar
_REINFORCE_RECALL = 0.15    # refuerzo por recall() explícito
_REINFORCE_SEARCH = 0.05    # refuerzo leve por aparecer en un search()
_DECAY_PER_WEEK = 0.85      # factor de decaimiento por semana sin uso
_REINFORCE_DUP = 0.15       # refuerzo por re-aprender lo mismo (dedup de learn)

_ID_PAD = 6                 # mem_000001

# ── Memoria de intentos (outcomes) — "si no dio resultado, no dio resultado" ──
# Los outcomes son memoria OPERATIVA y efímera: registran que una tool
# (una búsqueda web, una lectura de archivo) ya se intentó y NO dio
# resultado, para que el modelo no repita lo que ya falló. No son
# aprendizajes: viven poco, decaen por día y se barren solos.
_OUTCOME_KIND = "outcome"
_OUTCOME_NAMESPACE = "intentos"
_OUTCOME_IMPORTANCE = 2     # importancia fija: no compiten con aprendizajes
_OUTCOME_TTL_HOURS = 48     # vigencia de un intento fallido
_OUTCOME_DECAY_PER_DAY = 0.55   # outcomes decaen por DÍA (no por semana)
_OUTCOME_MAX = 200          # cap: el store de outcomes no crece infinito

_CLASES_OUTCOME_LEGIBLES = {
    "sin_resultados": "SIN RESULTADOS",
    "error": "ERROR",
}


def default_root() -> str:
    """Raíz de la memoria central. Compartida por todos los runtimes."""
    env = os.environ.get("AETHER_CENTRAL_MEMORY_PATH")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.expanduser("~"), ".aether", "memory")


def _today() -> str:
    return date.today().isoformat()


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z0-9_]+", (text or "").lower())
            if len(w) > 2}


def _norm_key(text: str) -> str:
    """Clave de dedup: minúsculas y espacios colapsados."""
    return " ".join((text or "").lower().split())


def _fingerprint(tool: str, args) -> str:
    """Fingerprint estable de (tool + args normalizados) para dedup de
    intentos: reintentar 'lo mismo' debe encontrar el outcome previo."""
    canon: dict[str, str] = {}
    if isinstance(args, dict):
        for k, v in sorted(args.items()):
            canon[str(k).strip().lower()] = str(v).strip().lower()
    raw = (f"{(tool or '').strip().lower()}|"
           f"{json.dumps(canon, sort_keys=True, ensure_ascii=False)}")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _resumen_args(args) -> str:
    """El argumento 'principal' de una tool para el summary del outcome."""
    if not isinstance(args, dict):
        return ""
    for clave in ("instruccion", "query", "command", "app", "path"):
        val = args.get(clave)
        if isinstance(val, str) and val.strip():
            return val.strip()[:120]
    return " ".join(str(v) for v in args.values()
                    if str(v).strip())[:120]


def _outcome_vivo(mem: dict, hoy: date) -> bool:
    """TTL de un outcome: caduca a las _OUTCOME_TTL_HOURS de su último uso."""
    try:
        ref = date.fromisoformat(str(mem.get("last_used") or mem.get("date")))
    except ValueError:
        return True
    return (hoy - ref).days * 24 < _OUTCOME_TTL_HOURS


def _recencia(last_used: str, hoy: date) -> float:
    """Factor de recencia 0-1 (1 = hoy; se halva cada ~2 semanas sin uso)."""
    try:
        days = (hoy - date.fromisoformat(str(last_used))).days
    except ValueError:
        days = 0
    return 1.0 / (1.0 + max(0, days) / 14.0)


class CentralMemory:
    """Memoria central compartida de Aether (JSON, atómica, stdlib-only).

    Operaciones conceptuales:
        memory.search()      — recuperar recuerdos relevantes (los refuerza)
        memory.recall()      — recuperar un recuerdo por id (lo refuerza más)
        memory.learn()       — incorporar un aprendizaje nuevo
        memory.update()      — corrección explícita de un recuerdo
        memory.forget()      — debilitar (NO borrar, salvo hard=True)
        memory.consolidate() — decaimiento por desuso / compactación
    """

    def __init__(self, root: str | None = None) -> None:
        self.root = os.path.abspath(os.path.expanduser(root)) if root else default_root()
        self._learning_path = os.path.join(self.root, "learning.json")
        self._user_path = os.path.join(self.root, "user.json")
        self._conv_dir = os.path.join(self.root, "conversations")
        self._lock = threading.RLock()
        self._learning: dict | None = None
        self._learning_mtime: float = 0.0

    # ── I/O interno ──────────────────────────────────────────────────────

    @staticmethod
    def _write_json_atomic(path: str, data) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _read_json(path: str, default):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _read_json_protegido(path: str, default):
        """Lee JSON; si existe pero está corrupto, lo aparta (renombra a
        *.corrupt-*) en vez de resetear silenciosamente — un save posterior
        NUNCA debe pisar datos posiblemente válidos con un estado vacío."""
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except (OSError, json.JSONDecodeError):
            try:
                ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                os.replace(path, f"{path}.corrupt-{ts}")
            except OSError:
                pass
            return default

    def _load_learning(self) -> dict:
        """Carga learning.json; recarga desde disco si otro runtime lo tocó."""
        with self._lock:
            try:
                mtime = os.path.getmtime(self._learning_path)
            except OSError:
                mtime = 0.0
            if self._learning is None or mtime != self._learning_mtime:
                data = self._read_json_protegido(self._learning_path, None)
                if not isinstance(data, dict) or data.get("version") != _FORMAT_VERSION:
                    data = {"version": _FORMAT_VERSION, "next_id": 1, "memories": {}}
                self._learning = data
                self._learning_mtime = mtime
            return self._learning

    def _save_learning(self) -> None:
        with self._lock:
            assert self._learning is not None
            self._write_json_atomic(self._learning_path, self._learning)
            try:
                self._learning_mtime = os.path.getmtime(self._learning_path)
            except OSError:
                self._learning_mtime = 0.0

    # ── Memoria de aprendizaje ───────────────────────────────────────────

    def learn(self, summary: str, *, detail: str = "", importance: int = 5,
              tags=(), runtime: str = "", namespace: str = "",
              kind: str = "learning", data: dict | None = None,
              memory_id: str | None = None) -> dict:
        """Incorpora un aprendizaje nuevo. No todo merece ser aprendido:
        el llamador decide; aquí solo se registra de forma legible."""
        summary = str(summary).strip()
        if not summary:
            raise ValueError("learn() requiere un summary no vacío")
        importance = max(1, min(10, int(importance)))
        with self._lock:
            store = self._load_learning()
            # Dedup: re-aprender lo MISMO (summary normalizado idéntico en el
            # mismo kind/namespace) refuerza el recuerdo existente en vez de
            # crear un duplicado. Los debilitados quedan afuera (un forget()
            # explícito no se revierte solo por re-aprender) y los outcomes
            # tienen su propio dedup por fingerprint (register_outcome).
            if memory_id is None:
                clave = _norm_key(summary)
                for prev in store["memories"].values():
                    if (prev.get("kind") != _OUTCOME_KIND
                            and prev.get("kind", "") == kind
                            and prev.get("namespace", "") == namespace
                            and not prev.get("weakened")
                            and _norm_key(prev.get("summary", "")) == clave):
                        self._touch(prev, _REINFORCE_DUP)
                        self._save_learning()
                        return dict(prev)
                memory_id = f"mem_{store['next_id']:0{_ID_PAD}d}"
                store["next_id"] += 1
            mem = {
                "id": memory_id,
                "summary": summary[:500],
                "detail": str(detail)[:2000],
                "date": _today(),
                "runtime": str(runtime),
                "namespace": str(namespace),
                "kind": str(kind),
                "tags": [str(t) for t in tags][:12],
                "importance": importance,
                # Fuerza inicial recalibrada: antes 0.3 + 0.07*imp nacía todo
                # casi fuerte (rango 0.37-1.0) y discriminaba poco; ahora la
                # importancia importa de verdad (imp 1 → 0.235, imp 10 → 1.0).
                "strength": min(1.0, 0.15 + 0.085 * importance),
                "uses": 0,
                "last_used": _today(),
                "weakened": False,
                "data": dict(data or {}),
            }
            store["memories"][memory_id] = mem
            self._save_learning()
            return dict(mem)

    def recall(self, memory_id: str) -> dict | None:
        """Recupera un recuerdo por id. Un recuerdo que vuelve a usarse se
        refuerza (y deja de estar debilitado si supera el umbral)."""
        with self._lock:
            store = self._load_learning()
            mem = store["memories"].get(memory_id)
            if mem is None:
                return None
            self._touch(mem, _REINFORCE_RECALL)
            self._save_learning()
            return dict(mem)

    def search(self, query: str = "", *, tags=None, namespace: str | None = None,
               kind: str | None = None, runtime: str | None = None,
               limit: int = 10, include_weak: bool = False,
               include_outcomes: bool = False, touch: bool = True) -> list[dict]:
        """Busca recuerdos relevantes accesibles (fuerza >= umbral).

        include_weak=True permite recuperar también recuerdos debilitados
        (siguen existiendo; solo son menos accesibles). touch=False evita
        el refuerzo: obligatorio para la inyección de contexto — aparecer
        en el prompt NO es "uso" (antes el solo inyectarlos los reforzaba
        en un bucle ricos-más-ricos). include_outcomes=True incluye la
        memoria de intentos (por defecto queda afuera: es operativa, no
        aprendizaje; para eso existe recent_outcomes()).
        """
        q_tokens = _tokens(query)
        tag_set = {str(t).lower() for t in tags} if tags else None
        today_d = date.today()
        scored: list[tuple[float, dict]] = []
        with self._lock:
            store = self._load_learning()
            for mem in store["memories"].values():
                if namespace is not None and mem.get("namespace", "") != namespace:
                    continue
                if kind is not None and mem.get("kind", "") != kind:
                    continue
                if runtime is not None and mem.get("runtime", "") != runtime:
                    continue
                if tag_set and not tag_set <= {t.lower() for t in mem.get("tags", [])}:
                    continue
                strength = float(mem.get("strength", 0.0))
                if not include_weak and (mem.get("weakened") or strength < _WEAK_THRESHOLD):
                    continue
                if not include_outcomes and mem.get("kind") == _OUTCOME_KIND:
                    continue
                if q_tokens:
                    haystack = (_tokens(mem.get("summary", ""))
                                | _tokens(mem.get("detail", ""))
                                | {t.lower() for t in mem.get("tags", [])})
                    overlap = len(q_tokens & haystack)
                    if overlap == 0:
                        continue
                    # Normalizado 0-1: cobertura de la query. Antes era el
                    # overlap crudo, que favorecía a los textos largos.
                    relevance = overlap / len(q_tokens)
                else:
                    relevance = 1.0
                # Ranking consistente: relevancia (0-1) × fuerza (0.6-1.0) ×
                # importancia (0.55-1.0) × recencia (0-1). Todas las señales
                # en escalas comparables, sin una dominando a las demás.
                score = (relevance
                         * (0.6 + 0.4 * strength)
                         * (0.5 + 0.05 * float(mem.get("importance", 5)))
                         * _recencia(mem.get("last_used") or mem.get("date"), today_d))
                scored.append((score, mem))
            scored.sort(key=lambda item: (item[0], item[1].get("last_used", "")),
                        reverse=True)
            results = [mem for _, mem in scored[:max(0, limit)]]
            if touch:
                for mem in results:
                    self._touch(mem, _REINFORCE_SEARCH)
                if results:
                    self._save_learning()
            return [dict(m) for m in results]

    def update(self, memory_id: str, *, summary=None, detail=None,
               importance=None, tags=None, data=None) -> bool:
        """Corrección EXPLÍCITA de un recuerdo existente.

        Es la única vía para corregir: los runtimes no pisan recuerdos
        arbitrariamente; una contradicción pasa por aquí.
        """
        with self._lock:
            store = self._load_learning()
            mem = store["memories"].get(memory_id)
            if mem is None:
                return False
            if summary is not None:
                mem["summary"] = str(summary).strip()[:500]
            if detail is not None:
                mem["detail"] = str(detail)[:2000]
            if importance is not None:
                mem["importance"] = max(1, min(10, int(importance)))
            if tags is not None:
                mem["tags"] = [str(t) for t in tags][:12]
            if data is not None:
                mem["data"] = dict(data)
            mem["updated"] = _today()
            self._save_learning()
            return True

    def forget(self, memory_id: str, *, hard: bool = False) -> bool:
        """Debilita un recuerdo. NO lo borra: queda inaccesible para las
        búsquedas normales pero conservado en disco (include_weak=True lo
        recupera). hard=True es la única forma de borrado físico."""
        with self._lock:
            store = self._load_learning()
            mem = store["memories"].get(memory_id)
            if mem is None:
                return False
            if hard:
                del store["memories"][memory_id]
            else:
                mem["weakened"] = True
                mem["strength"] = min(float(mem.get("strength", 0.0)), 0.05)
            self._save_learning()
            return True

    def consolidate(self, *, today: str | None = None) -> dict:
        """Decaimiento por desuso. Nunca borra APRENDIZAJES.

        Recuerdos no usados pierden fuerza con el tiempo (factor semanal);
        si caen muy bajo quedan marcados como debilitados. Los outcomes
        (memoria de intentos) son la excepción: efímeros por diseño,
        decaen por DÍA y los vencidos por TTL se barren (son la única
        cosa que consolidate elimina físicamente). Devuelve stats.
        """
        today_d = date.fromisoformat(today) if today else date.today()
        decayed = weakened = outcomes_removed = 0
        with self._lock:
            store = self._load_learning()
            for mem in list(store["memories"].values()):
                if mem.get("kind") == _OUTCOME_KIND:
                    if not _outcome_vivo(mem, today_d):
                        store["memories"].pop(mem.get("id"), None)
                        outcomes_removed += 1
                        continue
                    try:
                        last = date.fromisoformat(
                            str(mem.get("last_used") or mem.get("date")))
                    except ValueError:
                        continue
                    days = (today_d - last).days
                    if days > 0:
                        factor = _OUTCOME_DECAY_PER_DAY ** days
                        mem["strength"] = round(
                            max(_STRENGTH_FLOOR,
                                float(mem.get("strength", 0.0)) * factor), 4)
                        if (mem["strength"] < _WEAK_THRESHOLD
                                and not mem.get("weakened")):
                            mem["weakened"] = True
                    continue
                try:
                    last = date.fromisoformat(str(mem.get("last_used") or mem.get("date")))
                except ValueError:
                    continue
                days = (today_d - last).days
                if days <= 0:
                    continue
                factor = _DECAY_PER_WEEK ** (days / 7.0)
                new_strength = max(_STRENGTH_FLOOR,
                                   float(mem.get("strength", 0.0)) * factor)
                if new_strength < float(mem.get("strength", 0.0)):
                    decayed += 1
                mem["strength"] = round(new_strength, 4)
                if new_strength < _WEAK_THRESHOLD / 2 and not mem.get("weakened"):
                    mem["weakened"] = True
                    weakened += 1
            self._save_learning()
            return {"decayed": decayed, "newly_weakened": weakened,
                    "outcomes_removed": outcomes_removed,
                    "total": len(store["memories"])}

    def get(self, memory_id: str) -> dict | None:
        """Lectura sin refuerzo (para inspección/tooling)."""
        with self._lock:
            mem = self._load_learning()["memories"].get(memory_id)
            return dict(mem) if mem else None

    # ── Memoria de intentos (outcomes) ──────────────────────────────────

    def register_outcome(self, tool: str, args, clase: str, *,
                         resultado: str = "", runtime: str = "") -> dict | None:
        """Registra que un intento de tool NO dio resultado.

        "Si no dio resultado, no dio resultado": queda asentado para que el
        model no repita lo mismo en este turno ni en los próximos. Es
        kind='outcome' + namespace='intentos': memoria OPERATIVA y efímera
        (importancia fija baja, excluida de search()/personality_signals,
        decae por día y consolidate() barre los vencidos por TTL).

        Dedup por fingerprint (tool + args normalizados): reintentar lo
        mismo incrementa el contador del outcome existente en vez de
        duplicar la entrada.
        """
        tool = str(tool or "").strip()
        clase = str(clase or "error").strip()
        fp = _fingerprint(tool, args)
        hoy = date.today()
        with self._lock:
            store = self._load_learning()
            for prev in store["memories"].values():
                if prev.get("kind") != _OUTCOME_KIND:
                    continue
                if (prev.get("data") or {}).get("fingerprint") != fp:
                    continue
                if not _outcome_vivo(prev, hoy):
                    continue
                data = prev.setdefault("data", {})
                data["count"] = int(data.get("count", 1)) + 1
                data["clase"] = clase
                prev["last_used"] = _today()
                self._save_learning()
                return dict(prev)

            clase_legible = _CLASES_OUTCOME_LEGIBLES.get(clase, clase.upper())
            summary = f"{tool} '{_resumen_args(args)}' → {clase_legible}".strip()
            nuevo = self.learn(summary,
                              detail=(resultado or "")[:300],
                              importance=_OUTCOME_IMPORTANCE,
                              kind=_OUTCOME_KIND,
                              namespace=_OUTCOME_NAMESPACE,
                              runtime=runtime,
                              data={"tool": tool, "fingerprint": fp,
                                    "clase": clase, "count": 1})
            self._barrer_outcomes()
            return nuevo

    def recent_outcomes(self, query: str = "", *, limit: int = 5) -> list[dict]:
        """Outcomes VIVOS (TTL vigente), relevantes al query primero y
        después por recencia.

        Para el slot [INTENTOS RECIENTES] del context builder. No filtra
        por strength: el TTL y el decaimiento diario ya hacen desaparecer
        los outcomes solos; acá manda la recencia.
        """
        q_tokens = _tokens(query)
        hoy = date.today()
        scored: list[tuple[tuple[int, str, str], dict]] = []
        with self._lock:
            store = self._load_learning()
            for mem in store["memories"].values():
                if mem.get("kind") != _OUTCOME_KIND:
                    continue
                if not _outcome_vivo(mem, hoy):
                    continue
                data = mem.get("data") or {}
                hay = (_tokens(mem.get("summary", ""))
                       | _tokens(str(data.get("tool", ""))))
                overlap = len(q_tokens & hay) if q_tokens else 0
                scored.append(((overlap, str(mem.get("last_used") or ""),
                                str(mem.get("id") or "")), mem))
            scored.sort(key=lambda item: item[0], reverse=True)
            resultados = [mem for _, mem in scored[:max(0, limit)]]
            return [dict(m) for m in resultados]

    def _barrer_outcomes(self) -> None:
        """Cap de crecimiento: sobre _OUTCOME_MAX outcomes se eliminan los
        vencidos por TTL primero, después los más viejos.

        Solo aplica a outcomes (efímeros por diseño): los aprendizajes
        NUNCA se borran acá — olvidar != borrar sigue intacto para ellos.
        Recarga el store fresco: entre el learn() del caller y esta pasada
        pudo haber otra escritura (multi-runtime).
        """
        with self._lock:
            store = self._load_learning()
            outcomes = [m for m in store["memories"].values()
                        if m.get("kind") == _OUTCOME_KIND]
            if len(outcomes) <= _OUTCOME_MAX:
                return
            hoy = date.today()
            ordenados = sorted(
                outcomes,
                key=lambda m: (not _outcome_vivo(m, hoy),   # vencidos primero
                               str(m.get("last_used") or m.get("date") or "")),
            )
            excedente = len(outcomes) - _OUTCOME_MAX
            for m in ordenados[:excedente]:
                store["memories"].pop(m.get("id"), None)
            self._save_learning()

    # ── Memoria del usuario ──────────────────────────────────────────────

    def _load_user(self) -> dict:
        data = self._read_json_protegido(self._user_path, None)
        if not isinstance(data, dict) or data.get("version") != _FORMAT_VERSION:
            data = {"version": _FORMAT_VERSION, "profile": {}}
        return data

    def user_get(self, key: str | None = None):
        """Lee la memoria del usuario. Sin key devuelve el perfil completo
        (valores actuales, sin historial)."""
        profile = self._load_user()["profile"]
        if key is None:
            return {k: v["value"] for k, v in profile.items()}
        entry = profile.get(key)
        return None if entry is None else entry["value"]

    def user_set(self, key: str, value: str, *, runtime: str = "") -> tuple[bool, str]:
        """Establece un dato del usuario SIN sobrescribir silenciosamente.

        Si ya existe un valor distinto, NO se pisa: devuelve (False, conflicto)
        y el llamador debe usar user_update() (corrección explícita).
        """
        key = str(key).strip()
        if not key:
            return False, "clave vacía"
        with self._lock:
            store = self._load_user()
            existing = store["profile"].get(key)
            if existing is not None and existing["value"] != value:
                return False, (f"conflicto en '{key}': existe "
                               f"{existing['value']!r}; usá user_update() para corregir")
            if existing is None:
                store["profile"][key] = {
                    "value": str(value), "updated": _today(),
                    "runtime": str(runtime), "history": [],
                }
                self._write_json_atomic(self._user_path, store)
            return True, "ok"

    def user_update(self, key: str, value: str, *, runtime: str = "") -> bool:
        """Corrección explícita de un dato del usuario. El valor anterior
        queda archivado en el historial (trazabilidad, no borrado)."""
        key = str(key).strip()
        with self._lock:
            store = self._load_user()
            entry = store["profile"].get(key)
            if entry is None:
                store["profile"][key] = {
                    "value": str(value), "updated": _today(),
                    "runtime": str(runtime), "history": [],
                }
            else:
                entry["history"].append({
                    "value": entry["value"], "replaced": _today(),
                    "runtime": entry.get("runtime", ""),
                })
                entry["value"] = str(value)
                entry["updated"] = _today()
                entry["runtime"] = str(runtime)
            self._write_json_atomic(self._user_path, store)
            return True

    def user_forget(self, key: str) -> bool:
        """Olvida un dato del usuario (lo quita del perfil activo)."""
        with self._lock:
            store = self._load_user()
            entry = store["profile"].pop(key, None)
            if entry is None:
                return False
            self._write_json_atomic(self._user_path, store)
            return True

    def _touch(self, mem: dict, amount: float) -> None:
        """Refuerzo asintótico: la fuerza se acerca a 1.0 sin saturarlo.

        Antes el refuerzo era lineal (+amount fijo) y una memoria llegaba a
        1.0 tras ~14 usos, perdiendo todo poder discriminante.
        """
        mem["uses"] = int(mem.get("uses", 0)) + 1
        mem["last_used"] = _today()
        strength = float(mem.get("strength", 0.0))
        mem["strength"] = round(min(1.0, strength + amount * (1.0 - strength)), 4)
        if mem["strength"] >= _WEAK_THRESHOLD:
            mem["weakened"] = False

    # ── Señales de personalidad (acumulativas, nunca destructivas) ───────

    def personality_signals(self, *, limit: int = 10) -> list[str]:
        """Patrones acumulados a partir de los aprendizajes accesibles.

        Cada recuerdo es una señal pequeña; esto solo agrega patrones para
        que puedan influir gradualmente en el comportamiento. NADA aquí
        sobrescribe la personalidad: es una vista de solo lectura.
        """
        with self._lock:
            store = self._load_learning()
            accesibles = [m for m in store["memories"].values()
                          if not m.get("weakened")
                          and m.get("kind") != _OUTCOME_KIND
                          and float(m.get("strength", 0)) >= _WEAK_THRESHOLD]
        accesibles.sort(key=lambda m: float(m.get("strength", 0))
                        * float(m.get("importance", 5)), reverse=True)
        return [m["summary"] for m in accesibles[:max(0, limit)]]


    # ── Memoria de conversación ──────────────────────────────────────────

    def _conv_path(self, session_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(session_id))[:80] or "sin_sesion"
        return os.path.join(self._conv_dir, f"{safe}.json")

    def conversation_append(self, session_id: str, role: str, text: str,
                            *, runtime: str = "", max_turns: int = 200) -> None:
        """Agrega un turno a la memoria de conversación de una sesión."""
        with self._lock:
            path = self._conv_path(session_id)
            conv = self._read_json(path, None)
            if not isinstance(conv, dict) or conv.get("version") != _FORMAT_VERSION:
                conv = {"version": _FORMAT_VERSION, "session_id": str(session_id),
                        "runtime": str(runtime), "summary": "", "turns": []}
            conv["turns"].append({
                "role": str(role), "text": str(text)[:2000],
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
            })
            del conv["turns"][:-max_turns]
            self._write_json_atomic(path, conv)

    def conversation_get(self, session_id: str) -> dict:
        """Devuelve {session_id, summary, turns} de una sesión (vacía si no existe)."""
        conv = self._read_json(self._conv_path(session_id), None)
        if not isinstance(conv, dict) or conv.get("version") != _FORMAT_VERSION:
            return {"session_id": str(session_id), "summary": "", "turns": []}
        return {"session_id": conv["session_id"], "summary": conv.get("summary", ""),
                "turns": list(conv.get("turns", []))}

    def conversation_compact(self, session_id: str, summary: str,
                             *, keep_last: int = 6) -> bool:
        """Compacta una conversación: los turnos viejos se reemplazan por un
        resumen (contexto temporal, no aprendizaje permanente)."""
        with self._lock:
            path = self._conv_path(session_id)
            conv = self._read_json(path, None)
            if not isinstance(conv, dict) or conv.get("version") != _FORMAT_VERSION:
                return False
            conv["summary"] = str(summary)[:4000]
            conv["turns"] = conv.get("turns", [])[-keep_last:]
            self._write_json_atomic(path, conv)
            return True

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._lock:
            memories = list(self._load_learning()["memories"].values())
        weak = sum(1 for m in memories
                   if m.get("weakened") or float(m.get("strength", 0)) < _WEAK_THRESHOLD)
        by_kind: dict[str, int] = {}
        for m in memories:
            k = m.get("kind", "learning")
            by_kind[k] = by_kind.get(k, 0) + 1
        return {
            "root": self.root,
            "learnings": len(memories),
            "accessible": len(memories) - weak,
            "weak": weak,
            "by_kind": by_kind,
            "outcomes": by_kind.get(_OUTCOME_KIND, 0),
            "user_facts": len(self._load_user()["profile"]),
        }


_default: CentralMemory | None = None
_default_lock = threading.Lock()


def get_central_memory(root: str | None = None) -> CentralMemory:
    """Singleton por proceso para la raíz por defecto (o una instancia nueva
    si se pasa una raíz explícita)."""
    global _default
    if root is not None:
        return CentralMemory(root)
    with _default_lock:
        if _default is None:
            _default = CentralMemory()
        return _default

