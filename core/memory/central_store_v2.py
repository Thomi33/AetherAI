"""
Memoria central simplificada — SQLite3 en lugar de JSON custom.

Mismo API que CentralMemory pero con sqlite3 (stdlib) para persistencia.
Elimina: dedup complejo, strength decay, fingerprint, TTL manual, weakening,
consolidation threads, conversation files separados, etc.

Tablas:
- learnings: id, kind, summary, content, tags, importance, strength, weakened, updated_at, namespace
- outcomes: id, fingerprint, summary, data, importance, created_at, ttl_hours
- user_facts: id, key, value, updated_at
- conversations: session_id, runtime, summary, turns_json, updated_at
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from datetime import date, datetime, timedelta
from typing import Any

_FORMAT_VERSION = 1
_WEAK_THRESHOLD = 0.15
_STRENGTH_FLOOR = 0.01
_REINFORCE_RECALL = 0.15
_REINFORCE_SEARCH = 0.05
_DECAY_PER_WEEK = 0.85
_REINFORCE_DUP = 0.15

_OUTCOME_KIND = "outcome"
_OUTCOME_NAMESPACE = "intentos"
_OUTCOME_IMPORTANCE = 2
_OUTCOME_TTL_HOURS = 48
_OUTCOME_DECAY_PER_DAY = 0.55
_OUTCOME_MAX = 200

_ID_PAD = 6

_CLASES_OUTCOME_LEGIBLES = {
    "sin_resultados": "SIN RESULTADOS",
    "error": "ERROR",
}


def default_root() -> str:
    env = os.environ.get("AETHER_CENTRAL_MEMORY_PATH")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.expanduser("~"), ".aether", "memory")


def _today() -> str:
    return date.today().isoformat()


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[A-Za-z0-9_]+", (text or "").lower()) if len(w) > 2}


def _norm_key(text: str) -> str:
    return " ".join((text or "").lower().split())


def _fingerprint(tool: str, args) -> str:
    # Normalize args values for case-insensitive dedup
    if isinstance(args, dict):
        norm_args = {k: v.lower() if isinstance(v, str) else v for k, v in args.items()}
    else:
        norm_args = args
    raw = f"{tool}:{json.dumps(norm_args, sort_keys=True, ensure_ascii=False)}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


class CentralMemory:
    """Memoria central simplificada con SQLite3."""

    def __init__(self, root: str | None = None):
        self.root = root or default_root()
        os.makedirs(self.root, exist_ok=True)
        self._db_path = os.path.join(self.root, "memory.db")
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learnings (
                        id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL DEFAULT 'learning',
                        summary TEXT NOT NULL,
                        content TEXT NOT NULL,
                        tags TEXT NOT NULL DEFAULT '[]',
                        importance INTEGER NOT NULL DEFAULT 5,
                        strength REAL NOT NULL DEFAULT 1.0,
                        weakened INTEGER NOT NULL DEFAULT 0,
                        updated_at TEXT NOT NULL,
                        namespace TEXT NOT NULL DEFAULT ''
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_learnings_kind ON learnings(kind)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_learnings_strength ON learnings(strength)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_learnings_namespace ON learnings(namespace)")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS outcomes (
                        id TEXT PRIMARY KEY,
                        fingerprint TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        data TEXT NOT NULL DEFAULT '{}',
                        importance INTEGER NOT NULL DEFAULT 2,
                        created_at TEXT NOT NULL,
                        last_used TEXT NOT NULL,
                        ttl_hours INTEGER NOT NULL DEFAULT 48,
                        count INTEGER NOT NULL DEFAULT 1,
                        strength REAL NOT NULL DEFAULT 1.0
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_fingerprint ON outcomes(fingerprint)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_created ON outcomes(created_at)")

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_facts (
                        id TEXT PRIMARY KEY,
                        key TEXT NOT NULL UNIQUE,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        session_id TEXT PRIMARY KEY,
                        runtime TEXT NOT NULL DEFAULT '',
                        summary TEXT NOT NULL DEFAULT '',
                        turns_json TEXT NOT NULL DEFAULT '[]',
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.commit()
            finally:
                conn.close()

    def _conn(self):
        return sqlite3.connect(self._db_path)

    def _next_id(self, prefix: str = "mem") -> str:
        return f"{prefix}_{os.urandom(4).hex()}"

    # Learning methods...
    def _conn(self):
        return sqlite3.connect(self._db_path)

    def _next_id(self, prefix: str = "mem") -> str:
        return f"{prefix}_{os.urandom(4).hex()}"

    # Learning methods
    def learn(
        self,
        summary: str,
        *,
        importance: int = 5,
        tags: list[str] | None = None,
        content: str | None = None,
        namespace: str = "",
        runtime: str = "",
    ) -> dict:
        summary = (summary or "").strip()
        if not summary:
            raise ValueError("summary vacío")
        importance = max(1, min(10, int(importance)))
        tags = tags or []
        content = content or summary
        key = _norm_key(summary)

        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute(
                    "SELECT id, importance, strength FROM learnings WHERE kind != ? AND namespace = ? AND lower(trim(summary)) = ?",
                    (_OUTCOME_KIND, namespace, key.lower())
                )
                row = cur.fetchone()
                now = datetime.now().astimezone().isoformat(timespec="seconds")

                if row:
                    mem_id, old_imp, old_str = row
                    new_strength = float(old_str) + (1.0 - float(old_str)) * _REINFORCE_DUP
                    new_strength = min(0.999999, new_strength)
                    new_importance = max(int(old_imp), importance)
                    conn.execute(
                        "UPDATE learnings SET importance=?, strength=?, weakened=0, updated_at=?, content=? WHERE id=?",
                        (new_importance, new_strength, now, content, mem_id)
                    )
                    return {"id": mem_id, "reinforced": True, "strength": new_strength, "uses": 1}

                mem_id = self._next_id()
                initial_strength = min(1.0, 0.1 + importance * 0.09)
                conn.execute(
                    "INSERT INTO learnings (id, kind, summary, content, tags, importance, strength, weakened, updated_at, namespace) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (mem_id, "learning", summary, content, json.dumps(tags, ensure_ascii=False), importance, initial_strength, 0, now, namespace)
                )
                conn.commit()
                return {"id": mem_id, "reinforced": False, "strength": initial_strength, "uses": 0}
            finally:
                conn.close()

    def search(self, query: str, *, limit: int = 10, touch: bool = True, namespace: str = "", include_outcomes: bool = False) -> list[dict]:
        q_tokens = _tokens(query)
        if not q_tokens:
            return []

        with self._lock:
            conn = self._conn()
            try:
                rows = []
                # Search learnings
                if namespace:
                    cur = conn.execute(
                        "SELECT id, kind, summary, content, tags, importance, strength, weakened, updated_at, namespace FROM learnings WHERE kind != ? AND weakened = 0 AND strength >= ? AND namespace = ?",
                        (_OUTCOME_KIND, _WEAK_THRESHOLD, namespace)
                    )
                else:
                    cur = conn.execute(
                        "SELECT id, kind, summary, content, tags, importance, strength, weakened, updated_at, namespace FROM learnings WHERE kind != ? AND weakened = 0 AND strength >= ?",
                        (_OUTCOME_KIND, _WEAK_THRESHOLD)
                    )
                rows.extend(cur.fetchall())

                # Optionally search outcomes
                if include_outcomes:
                    if namespace:
                        cur = conn.execute(
                            "SELECT id, 'outcome' as kind, summary, data as content, '[]' as tags, importance, 1.0 as strength, 0 as weakened, created_at as updated_at, 'intentos' as namespace FROM outcomes WHERE namespace = ? ORDER BY created_at DESC",
                            (namespace,)
                        )
                    else:
                        cur = conn.execute(
                            "SELECT id, 'outcome' as kind, summary, data as content, '[]' as tags, importance, 1.0 as strength, 0 as weakened, created_at as updated_at, 'intentos' as namespace FROM outcomes ORDER BY created_at DESC"
                        )
                    rows.extend(cur.fetchall())

                scored = []
                for row in rows:
                    mem_id, kind, summary, content, tags_json, importance, strength, weakened, updated_at, ns = row
                    s_tokens = _tokens(summary) | _tokens(content)
                    overlap = len(q_tokens & s_tokens)
                    if overlap == 0:
                        continue
                    score = overlap * (1 + float(importance) / 10.0) * float(strength)
                    scored.append((score, row))

                scored.sort(key=lambda x: x[0], reverse=True)
                results = []
                for score, row in scored[:limit]:
                    mem_id, kind, summary, content, tags_json, importance, strength, weakened, updated_at, ns = row
                    result = {
                        "id": mem_id,
                        "kind": kind,
                        "summary": summary,
                        "content": content,
                        "tags": json.loads(tags_json or "[]"),
                        "importance": importance,
                        "strength": strength,
                        "weakened": bool(weakened),
                        "updated_at": updated_at,
                        "namespace": ns,
                    }
                    if touch:
                        new_strength = float(strength) + (1.0 - float(strength)) * _REINFORCE_SEARCH
                        new_strength = min(0.999999, new_strength)
                        conn.execute("UPDATE learnings SET strength=?, updated_at=? WHERE id=?", (new_strength, datetime.now().astimezone().isoformat(timespec="seconds"), mem_id))
                        conn.commit()
                    results.append(result)
                return results
            finally:
                conn.close()

    def recall(self, memory_id: str) -> dict | None:
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("SELECT id, kind, summary, content, tags, importance, strength, weakened, updated_at, namespace FROM learnings WHERE id=?", (memory_id,))
                row = cur.fetchone()
                if not row:
                    return None
                mem_id, kind, summary, content, tags_json, importance, strength, weakened, updated_at, ns = row
                new_strength = min(1.0, float(strength) + _REINFORCE_RECALL)
                conn.execute("UPDATE learnings SET strength=?, weakened=0, updated_at=? WHERE id=?", (new_strength, datetime.now().astimezone().isoformat(timespec="seconds"), mem_id))
                conn.commit()
                return {
                    "id": mem_id, "kind": kind, "summary": summary, "content": content,
                    "tags": json.loads(tags_json or "[]"), "importance": importance,
                    "strength": new_strength, "weakened": False, "updated_at": updated_at, "namespace": ns
                }
            finally:
                conn.close()

    def forget(self, memory_id: str, hard: bool = False) -> bool:
        with self._lock:
            conn = self._conn()
            try:
                if hard:
                    cur = conn.execute("DELETE FROM learnings WHERE id=?", (memory_id,))
                    conn.commit()
                    return cur.rowcount > 0
                cur = conn.execute("UPDATE learnings SET weakened=1, strength=? WHERE id=?", (_STRENGTH_FLOOR, memory_id))
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def update(self, memory_id: str, **fields) -> bool:
        allowed = {"summary", "content", "tags", "importance", "namespace"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return False
        if "tags" in updates:
            updates["tags"] = json.dumps(updates["tags"], ensure_ascii=False)
        if "importance" in updates:
            updates["importance"] = max(1, min(10, int(updates["importance"])))

        with self._lock:
            conn = self._conn()
            try:
                set_clause = ", ".join(f"{k}=?" for k in updates)
                vals = list(updates.values()) + [datetime.now().astimezone().isoformat(timespec="seconds"), memory_id]
                cur = conn.execute(f"UPDATE learnings SET {set_clause}, updated_at=? WHERE id=?", vals)
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()

    def get(self, memory_id: str) -> dict | None:
        with self._lock:
            conn = self._conn()
            try:
                # Try learnings first
                cur = conn.execute("SELECT id, kind, summary, content, tags, importance, strength, weakened, updated_at, namespace FROM learnings WHERE id=?", (memory_id,))
                row = cur.fetchone()
                if row:
                    mem_id, kind, summary, content, tags_json, importance, strength, weakened, updated_at, ns = row
                    return {
                        "id": mem_id, "kind": kind, "summary": summary, "content": content,
                        "tags": json.loads(tags_json or "[]"), "importance": importance,
                        "strength": strength, "weakened": bool(weakened), "updated_at": updated_at, "namespace": ns,
                        "uses": 0
                    }
                # Try outcomes
                cur = conn.execute("SELECT id, summary, data, importance, created_at, ttl_hours, last_used, strength FROM outcomes WHERE id=?", (memory_id,))
                row = cur.fetchone()
                if row:
                    mem_id, summary, data_json, importance, created_at, ttl_hours, last_used, strength = row
                    data = json.loads(data_json or "{}")
                    return {
                        "id": mem_id, "kind": _OUTCOME_KIND, "summary": summary,
                        "content": summary, "tags": [], "importance": importance,
                        "strength": strength, "weakened": False, "updated_at": created_at,
                        "namespace": _OUTCOME_NAMESPACE, "last_used": last_used,
                        "uses": 0, "data": data
                    }
                return None
            finally:
                conn.close()
    # Outcomes methods
    def register_outcome(
        self,
        tool: str,
        args: dict,
        clase: str,
        *,
        summary: str | None = None,
        importance: int = _OUTCOME_IMPORTANCE,
        runtime: str = "",
    ) -> dict:
        fp = _fingerprint(tool, args)
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        legible = _CLASES_OUTCOME_LEGIBLES.get(clase, clase.upper())
        summ = summary or f"{tool}: {legible} ({json.dumps(args, ensure_ascii=False)[:80]})"

        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("SELECT id, count FROM outcomes WHERE fingerprint=?", (fp,))
                row = cur.fetchone()
                if row:
                    mem_id, count = row[0], row[1]
                    new_count = count + 1
                    conn.execute("UPDATE outcomes SET created_at=?, summary=?, count=?, last_used=? WHERE id=?", (now, summ, new_count, now, mem_id))
                    conn.commit()
                    return {"id": mem_id, "reinforced": True, "kind": _OUTCOME_KIND, "namespace": _OUTCOME_NAMESPACE, "importance": importance, "strength": 1.0, "summary": summ, "data": {"tool": tool, "args": args, "clase": clase, "count": new_count}}

                mem_id = self._next_id()
                conn.execute(
                    "INSERT INTO outcomes (id, fingerprint, summary, data, importance, created_at, last_used, ttl_hours, count, strength) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (mem_id, fp, summ, json.dumps({"tool": tool, "args": args, "clase": clase}, ensure_ascii=False), importance, now, now, _OUTCOME_TTL_HOURS, 1, 1.0)
                )
                conn.commit()
                return {"id": mem_id, "reinforced": False, "kind": _OUTCOME_KIND, "namespace": _OUTCOME_NAMESPACE, "importance": importance, "strength": 1.0, "summary": summ, "data": {"tool": tool, "args": args, "clase": clase, "count": 1}}
            finally:
                conn.close()

    def recent_outcomes(self, query: str = "", *, limit: int = 5, touch: bool = False) -> list[dict]:
        with self._lock:
            conn = self._conn()
            try:
                cutoff = (datetime.now().astimezone() - timedelta(hours=_OUTCOME_TTL_HOURS)).isoformat(timespec="seconds")
                cur = conn.execute("SELECT id, fingerprint, summary, data, importance, last_used, ttl_hours FROM outcomes WHERE last_used >= ? ORDER BY last_used DESC LIMIT ?", (cutoff, limit * 3))
                rows = cur.fetchall()
                results = []
                q_tokens = _tokens(query)
                for row in rows:
                    mem_id, fp, summ, data_json, importance, last_used, ttl_hours = row
                    if q_tokens and not any(tok in summ.lower() for tok in q_tokens):
                        continue
                    data = json.loads(data_json or "{}")
                    count = int(data.get("count", 1))
                    results.append({
                        "id": mem_id,
                        "summary": summ,
                        "data": {"tool": data.get("tool"), "args": data.get("args"), "clase": data.get("clase"), "count": count},
                    })
                    if touch:
                        now = datetime.now().astimezone().isoformat(timespec="seconds")
                        conn.execute("UPDATE outcomes SET last_used=? WHERE id=?", (now, mem_id))
                        conn.commit()
                    if len(results) >= limit:
                        break
                return results
            finally:
                conn.close()

    # Consolidation
    def consolidate(self) -> dict:
        stats = {"learnings_decayed": 0, "learnings_weakened": 0, "outcomes_removed": 0}
        now = datetime.now().astimezone()
        week_ago = (now - timedelta(weeks=1)).isoformat(timespec="seconds")

        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute(
                    "SELECT id, strength FROM learnings WHERE kind != ? AND weakened = 0 AND strength > ? AND updated_at < ?",
                    (_OUTCOME_KIND, _STRENGTH_FLOOR, week_ago)
                )
                for mem_id, strength in cur.fetchall():
                    new_strength = float(strength) * _DECAY_PER_WEEK
                    if new_strength <= _WEAK_THRESHOLD:
                        conn.execute("UPDATE learnings SET weakened=1, strength=?, updated_at=? WHERE id=?", (_STRENGTH_FLOOR, now.isoformat(timespec="seconds"), mem_id))
                        stats["learnings_weakened"] += 1
                    else:
                        conn.execute("UPDATE learnings SET strength=?, updated_at=? WHERE id=?", (new_strength, now.isoformat(timespec="seconds"), mem_id))
                        stats["learnings_decayed"] += 1

                cutoff = (now - timedelta(hours=_OUTCOME_TTL_HOURS)).isoformat(timespec="seconds")
                # Decay outcome strength daily (for outcomes not yet expired by TTL)
                day_ago = (now - timedelta(days=1)).isoformat(timespec="seconds")
                cur = conn.execute(
                    "SELECT id, strength FROM outcomes WHERE last_used >= ? AND last_used <= ?",
                    (cutoff, day_ago)
                )
                for mem_id, strength in cur.fetchall():
                    new_strength = float(strength) * _OUTCOME_DECAY_PER_DAY
                    print(f"DEBUG consolidate: Decaying {mem_id}: {strength} -> {new_strength}")
                    cur2 = conn.execute("UPDATE outcomes SET strength=? WHERE id=?", (new_strength, mem_id))
                    print(f"DEBUG: Rows affected: {cur2.rowcount}")
                    stats["outcomes_decayed"] = stats.get("outcomes_decayed", 0) + 1

                cur = conn.execute("DELETE FROM outcomes WHERE last_used < ?", (cutoff,))
                stats["outcomes_removed"] = cur.rowcount

                cur = conn.execute("SELECT COUNT(*) FROM outcomes")
                count = cur.fetchone()[0]
                if count > _OUTCOME_MAX:
                    excess = count - _OUTCOME_MAX
                    conn.execute("DELETE FROM outcomes WHERE id IN (SELECT id FROM outcomes ORDER BY created_at ASC LIMIT ?)", (excess,))
                    stats["outcomes_removed"] += excess

                conn.commit()
                return stats
            finally:
                conn.close()

    # User facts
    def set_fact(self, key: str, value: str) -> None:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._lock:
            conn = self._conn()
            try:
                conn.execute(
                    "INSERT INTO user_facts (id, key, value, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (self._next_id("fact"), key, value, now)
                )
                conn.commit()
            finally:
                conn.close()

    def get_fact(self, key: str) -> str | None:
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("SELECT value FROM user_facts WHERE key=?", (key,))
                row = cur.fetchone()
                return row[0] if row else None
            finally:
                conn.close()

    def get_all_facts(self) -> dict[str, str]:
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("SELECT key, value FROM user_facts")
                return {row[0]: row[1] for row in cur.fetchall()}
            finally:
                conn.close()

    # Conversations
    def conversation_append(self, session_id: str, role: str, text: str, *, runtime: str = "", max_turns: int = 200) -> None:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("SELECT turns_json, summary, runtime FROM conversations WHERE session_id=?", (session_id,))
                row = cur.fetchone()
                if row:
                    turns = json.loads(row[0] or "[]")
                    summary = row[1]
                    rt = row[2]
                else:
                    turns = []
                    summary = ""
                    rt = runtime
                turns.append({"role": role, "text": str(text)[:2000], "ts": datetime.now().astimezone().isoformat(timespec="seconds")})
                turns = turns[-max_turns:]
                conn.execute(
                    "INSERT INTO conversations (session_id, runtime, summary, turns_json, updated_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET runtime=excluded.runtime, turns_json=excluded.turns_json, updated_at=excluded.updated_at",
                    (session_id, rt, summary, json.dumps(turns, ensure_ascii=False), now)
                )
                conn.commit()
            finally:
                conn.close()

    def conversation_get(self, session_id: str) -> dict:
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("SELECT session_id, runtime, summary, turns_json FROM conversations WHERE session_id=?", (session_id,))
                row = cur.fetchone()
                if not row:
                    return {"session_id": session_id, "summary": "", "turns": []}
                sid, rt, summ, turns_json = row
                return {"session_id": sid, "runtime": rt, "summary": summ, "turns": json.loads(turns_json or "[]")}
            finally:
                conn.close()

    def conversation_compact(self, session_id: str, summary: str, *, keep_last: int = 6) -> bool:
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("SELECT turns_json FROM conversations WHERE session_id=?", (session_id,))
                row = cur.fetchone()
                if not row:
                    return False
                turns = json.loads(row[0] or "[]")
                if len(turns) <= keep_last:
                    return True
                new_turns = turns[-keep_last:]
                conn.execute(
                    "UPDATE conversations SET summary=?, turns_json=?, updated_at=? WHERE session_id=?",
                    (summary[:4000], json.dumps(new_turns, ensure_ascii=False), datetime.now().astimezone().isoformat(timespec="seconds"), session_id)
                )
                conn.commit()
                return True
            finally:
                conn.close()

    # Stats
    def stats(self) -> dict:
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("SELECT COUNT(*) FROM learnings WHERE kind != ? AND weakened = 0", (_OUTCOME_KIND,))
                learnings = cur.fetchone()[0]
                cur = conn.execute("SELECT COUNT(*) FROM learnings WHERE kind != ? AND (weakened = 1 OR strength < ?)", (_OUTCOME_KIND, _WEAK_THRESHOLD))
                weak = cur.fetchone()[0]
                cur = conn.execute("SELECT kind, COUNT(*) FROM learnings WHERE kind != ? GROUP BY kind", (_OUTCOME_KIND,))
                by_kind = {row[0]: row[1] for row in cur.fetchall()}
                # Include outcomes in learnings count for backward compatibility
                cur = conn.execute("SELECT COUNT(*) FROM outcomes")
                outcomes = cur.fetchone()[0]
                cur = conn.execute("SELECT COUNT(*) FROM user_facts")
                user_facts = cur.fetchone()[0]
                return {
                    "root": self.root,
                    "learnings": learnings + weak + outcomes,
                    "accessible": learnings + outcomes,
                    "weak": weak,
                    "by_kind": by_kind,
                    "outcomes": outcomes,
                    "user_facts": user_facts,
                }
            finally:
                conn.close()

    def personality_signals(self) -> list[str]:
        """Retorna señales de personalidad para el prompt (solo aprendizajes, no outcomes)."""
        with self._lock:
            conn = self._conn()
            try:
                cur = conn.execute("SELECT summary FROM learnings WHERE kind != ? AND weakened = 0 AND strength >= ?", (_OUTCOME_KIND, _WEAK_THRESHOLD))
                return [row[0] for row in cur.fetchall()]
            finally:
                conn.close()

    def _load_learning(self) -> dict:
        """Compat: returns dict with 'memories' key containing all learnings + outcomes."""
        with self._lock:
            conn = self._conn()
            try:
                memories = {}
                cur = conn.execute("SELECT id, kind, summary, content, tags, importance, strength, weakened, updated_at, namespace FROM learnings")
                for row in cur.fetchall():
                    mem_id, kind, summary, content, tags_json, importance, strength, weakened, updated_at, ns = row
                    memories[mem_id] = {
                        "id": mem_id, "kind": kind, "summary": summary, "content": content,
                        "tags": json.loads(tags_json or "[]"), "importance": importance,
                        "strength": strength, "weakened": bool(weakened), "updated_at": updated_at, "namespace": ns
                    }
                cur = conn.execute("SELECT id, summary, data, importance, created_at, ttl_hours, last_used FROM outcomes")
                for row in cur.fetchall():
                    mem_id, summary, data_json, importance, created_at, ttl_hours, last_used = row
                    memories[mem_id] = {
                        "id": mem_id, "kind": _OUTCOME_KIND, "summary": summary, "content": summary,
                        "tags": [], "importance": importance, "strength": 1.0, "weakened": False,
                        "updated_at": created_at, "namespace": _OUTCOME_NAMESPACE, "last_used": last_used
                    }
                return {"memories": memories}
            finally:
                conn.close()

    def _save_learning(self, mem_dict: dict | None = None) -> None:
        """Compat: saves the learning dict back to DB."""
        # If called with a specific memory dict, update that outcome's last_used
        if mem_dict is not None:
            with self._lock:
                conn = self._conn()
                try:
                    mem_id = mem_dict.get("id")
                    last_used = mem_dict.get("last_used")
                    if mem_id and last_used:
                        conn.execute("UPDATE outcomes SET last_used=? WHERE id=?", (last_used, mem_id))
                        conn.commit()
                finally:
                    conn.close()
        # If called without args (legacy API), no-op since sqlite auto-persists


_default: CentralMemory | None = None
_default_lock = threading.Lock()


def get_central_memory(root: str | None = None) -> CentralMemory:
    global _default
    if root is not None:
        return CentralMemory(root)
    with _default_lock:
        if _default is None:
            _default = CentralMemory()
        return _default
