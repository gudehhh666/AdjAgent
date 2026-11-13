"""Atomic turns, fenced checkpoints and host-backed public working state.

Runs.state is the runtime checkpoint, not a second research ledger. Extra tables
own only leases, editable plans and actual-user clarification obligations.
"""

from datetime import datetime, timezone
import hashlib
import json
import math
import platform
import re
import sqlite3
import subprocess
import time

from ..contracts import Budget, new_id
from ..research.acceptance import requirements
from ..storage import RUN_STATUSES, TERMINAL_STATUSES
from ..tools.catalog import canonical_json


class RunBusy(RuntimeError):
    pass


class LeaseLost(RuntimeError):
    pass


class LeaseClockUnavailable(LeaseLost):
    """A persisted monotonic lease cannot be compared safely in this process."""


def now():
    return datetime.now(timezone.utc).isoformat()


def task_digest(task):
    return hashlib.sha256(canonical_json({"goal": task["goal"], "acceptance": task["acceptance"]}).encode()).hexdigest()


def text_value(value, label, maximum=100000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} must be nonempty text of at most {maximum} characters")
    canonical_json(value)
    return value


def _system_boot_id():
    """Return a stable boot identity, or None when the host cannot provide one."""
    try:
        if platform.system() == "Darwin":
            result = subprocess.run(["sysctl", "-n", "kern.bootsessionuuid"],
                                    check=True, capture_output=True, text=True, timeout=0.5)
            value = result.stdout.strip()
        elif platform.system() == "Linux":
            with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as stream:
                value = stream.read(128).strip()
        else:
            return None
    except (OSError, subprocess.SubprocessError):
        return None
    return value if re.fullmatch(r"[A-Za-z0-9-]{8,128}", value) else None


class RunJournal:
    def __init__(self, store, *, wall_clock=time.time, monotonic_clock=time.monotonic,
                 boot_id_provider=_system_boot_id):
        self.store = store
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._boot_id_provider = boot_id_provider
        with store.transaction(write=True) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS runtime_leases (
                session_id TEXT PRIMARY KEY REFERENCES sessions(id),
                run_id TEXT NOT NULL REFERENCES runs(id), owner TEXT NOT NULL, expires REAL NOT NULL)""")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(runtime_leases)")}
            if "clock_id" not in columns:
                conn.execute("ALTER TABLE runtime_leases ADD COLUMN clock_id TEXT")
            if "expires_monotonic" not in columns:
                conn.execute("ALTER TABLE runtime_leases ADD COLUMN expires_monotonic REAL")
            conn.execute("""CREATE TABLE IF NOT EXISTS runtime_lease_diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, owner TEXT NOT NULL,
                category TEXT NOT NULL, exception_type TEXT NOT NULL, wall_time REAL NOT NULL,
                monotonic_time REAL NOT NULL, clock_id TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS runtime_inputs (
                id TEXT UNIQUE NOT NULL, run_id TEXT NOT NULL REFERENCES runs(id),
                call_id TEXT NOT NULL, question TEXT NOT NULL, answer TEXT,
                PRIMARY KEY(run_id,call_id))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS runtime_plans (
                run_id TEXT PRIMARY KEY REFERENCES runs(id), steps TEXT NOT NULL)""")

    def encode(self, value):
        return canonical_json(self.store.redact(value))

    def _event(self, conn, sid, rid, kind, payload):
        conn.execute("INSERT INTO events(session_id,run_id,kind,payload,created_at) VALUES(?,?,?,?,?)",
                     (sid, rid, kind, self.encode(payload), now()))

    def _run(self, conn, rid):
        row = conn.execute("SELECT * FROM runs WHERE id=?", (rid,)).fetchone()
        if row is None: raise KeyError("Unknown run")
        return dict(row)

    def _open_run(self, conn, rid):
        run = self._run(conn, rid)
        if run["status"] in TERMINAL_STATUSES:
            raise ValueError("Run is terminal")
        return run

    def _boot_id(self):
        try:
            value = self._boot_id_provider()
        except Exception:
            return None
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9-]{8,128}", value):
            return None
        return value

    def _lease_live(self, row):
        """Compare expiry only in the clock domain recorded by the lease."""
        clock_id = row["clock_id"]
        if clock_id is None or clock_id == "wall":  # legacy rows use their original wall deadline
            deadline = row["expires"]
            if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
                raise LeaseClockUnavailable("Persisted wall lease deadline is invalid; ownership is fenced")
            return deadline > self._wall_clock()
        if not isinstance(clock_id, str) or not re.fullmatch(r"[A-Za-z0-9-]{8,128}", clock_id):
            raise LeaseClockUnavailable("Persisted lease clock identity is invalid; ownership is fenced")
        current_boot = self._boot_id()
        if current_boot is None:
            raise LeaseClockUnavailable("Lease clock identity is unavailable; ownership is fenced")
        if current_boot != clock_id:
            return False
        deadline = row["expires_monotonic"]
        if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
            raise LeaseClockUnavailable("Persisted monotonic lease deadline is invalid; ownership is fenced")
        return deadline > self._monotonic_clock()

    def _lease_values(self, ttl):
        wall_deadline = self._wall_clock() + ttl
        boot_id = self._boot_id()
        if boot_id is None:
            return wall_deadline, "wall", None
        return wall_deadline, boot_id, self._monotonic_clock() + ttl

    def start(self, session_id, goal, acceptance, budget):
        text_value(goal, "Goal")
        if acceptance: requirements(acceptance)
        elif not isinstance(acceptance, dict): raise ValueError("Acceptance must be an object")
        budget = Budget(**budget).to_dict()
        tid, rid, stamp = new_id("task"), new_id("run"), now()
        state = {"schema_version": 1, "phase": "ready", "steps": 0, "elapsed_active_seconds": 0.0}
        with self.store.transaction(write=True) as conn:
            if conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone() is None:
                raise KeyError("Unknown session")
            active = conn.execute("SELECT status FROM runs WHERE session_id=?", (session_id,)).fetchall()
            lease = conn.execute("SELECT * FROM runtime_leases WHERE session_id=?", (session_id,)).fetchone()
            if any(row[0] not in TERMINAL_STATUSES for row in active) or (lease and self._lease_live(lease)):
                raise RunBusy("Continue/cancel the outstanding run before submitting another turn")
            conn.execute("INSERT INTO tasks VALUES(?,?,?,?,?,0,?)",
                         (tid, session_id, self.store.redact(goal), self.encode(acceptance), "open", stamp))
            conn.execute("INSERT INTO runs VALUES(?,?,?,?,0,?,?,?,?)",
                         (rid, session_id, tid, "created", self.encode(budget), "{}", self.encode(state), stamp))
            message = {"role": "user", "content": goal}
            conn.execute("INSERT INTO messages(session_id,run_id,message,created_at) VALUES(?,?,?,?)",
                         (session_id, rid, self.encode(message), stamp))
            self._event(conn, session_id, rid, "runtime.turn_created", {"task_id": tid, "goal": goal, "acceptance": acceptance})
        return self.store.get_run(rid)

    @staticmethod
    def _ttl(ttl):
        if isinstance(ttl, bool) or not isinstance(ttl, (int, float)) or not math.isfinite(ttl) or not 0 < ttl <= 3600:
            raise ValueError("Lease ttl must be finite, positive and at most3600s")

    def acquire(self, run_id, ttl=15):
        self._ttl(ttl)
        owner = new_id("driver")
        with self.store.transaction(write=True) as conn:
            run = self._run(conn, run_id)
            current = conn.execute("SELECT * FROM runtime_leases WHERE session_id=?", (run["session_id"],)).fetchone()
            if current and self._lease_live(current):
                raise RunBusy("Another driver owns this session; reconnect after its lease expires")
            expires, clock_id, expires_monotonic = self._lease_values(ttl)
            conn.execute("""INSERT OR REPLACE INTO runtime_leases
                (session_id,run_id,owner,expires,clock_id,expires_monotonic) VALUES(?,?,?,?,?,?)""",
                (run["session_id"], run_id, owner, expires, clock_id, expires_monotonic))
        return owner

    def _fence(self, conn, rid, owner):
        row = conn.execute("SELECT * FROM runtime_leases WHERE run_id=? AND owner=?", (rid, owner)).fetchone()
        if row is None or not self._lease_live(row):
            raise LeaseLost("Runtime ownership expired; no further requests/effects may be initiated")
        return row

    def renew(self, run_id, owner, ttl=15):
        self._ttl(ttl)
        with self.store.transaction(write=True) as conn:
            lease = self._fence(conn, run_id, owner)
            # Keep the lease in its established time domain. In particular,
            # never turn an existing monotonic lease into a wall lease if the
            # boot-ID probe changes availability between renewal operations.
            clock_id = lease["clock_id"]
            if clock_id is not None and clock_id != "wall":
                expires, expires_monotonic = self._wall_clock() + ttl, self._monotonic_clock() + ttl
            else:
                expires, expires_monotonic = self._wall_clock() + ttl, None
            conn.execute("UPDATE runtime_leases SET expires=?,clock_id=?,expires_monotonic=? WHERE run_id=? AND owner=?",
                         (expires, clock_id if clock_id is not None else None,
                          expires_monotonic, run_id, owner))

    @staticmethod
    def _diagnostic_type(exc):
        name = type(exc).__name__
        safe_name = name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) else "Exception"
        if isinstance(exc, LeaseLost):
            category = "lease_fence_failure"
        elif isinstance(exc, sqlite3.Error):
            category = "sqlite_renewal_error"
        else:
            category = "heartbeat_renewal_error"
        return category, safe_name

    @classmethod
    def heartbeat_failure_error(cls, exc):
        category, exception_type = cls._diagnostic_type(exc)
        return LeaseLost(f"Runtime lease heartbeat failed ({category}, {exception_type})")

    def record_heartbeat_failure(self, run_id, owner, exc):
        """Append bounded diagnostic evidence without changing lease or run state."""
        category, exception_type = self._diagnostic_type(exc)
        wall_time, monotonic_time, clock_id = self._wall_clock(), self._monotonic_clock(), self._boot_id()
        try:
            with self.store.transaction(write=True) as conn:
                conn.execute("""INSERT INTO runtime_lease_diagnostics
                    (run_id,owner,category,exception_type,wall_time,monotonic_time,clock_id)
                    VALUES(?,?,?,?,?,?,?)""",
                    (run_id, owner, category, exception_type, wall_time, monotonic_time, clock_id))
        except Exception:
            # Diagnostics are observational; persistence failure must not mask the
            # renewal failure or alter the owner-fencing outcome.
            pass
        return self.heartbeat_failure_error(exc)

    def diagnostics(self, run_id):
        self.store.get_run(run_id)
        with self.store.transaction() as conn:
            rows = conn.execute("SELECT * FROM runtime_lease_diagnostics WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        return [dict(row) for row in rows]

    def release(self, run_id, owner):
        with self.store.transaction(write=True) as conn:
            conn.execute("DELETE FROM runtime_leases WHERE run_id=? AND owner=?", (run_id, owner))

    def checkpoint(self, run_id, owner, *, state=None, status=None, usage=None, messages=(), event="runtime.checkpoint",
                   expected_task_binding=None, active_deadline=None):
        encoded_state = None if state is None else self.encode(state)
        encoded_usage = None if usage is None else self.encode(usage)
        encoded_messages = [self.encode(message) for message in messages]
        if status is not None and status not in RUN_STATUSES: raise ValueError("Invalid run status")
        with self.store.transaction(write=True) as conn:
            self._fence(conn, run_id, owner)
            run = self._run(conn, run_id)
            if run["status"] in TERMINAL_STATUSES and (status not in {None, run["status"]} or encoded_messages):
                raise ValueError("Terminal run cannot be restarted or publish new conversation messages")
            if status == "succeeded":
                task_row = conn.execute("SELECT goal,acceptance FROM tasks WHERE id=?", (run["task_id"],)).fetchone()
                current_binding = task_digest({"goal": task_row["goal"], "acceptance": json.loads(task_row["acceptance"])})
                reason = None
                if expected_task_binding is not None and expected_task_binding != current_binding:
                    status, reason = "failed", "task_changed"
                if active_deadline is not None and time.monotonic() >= active_deadline:
                    status, reason = "budget_exhausted", "budget_exhausted"
                if reason:
                    updated = json.loads(encoded_state or run["state"])
                    updated.update(phase="finished", stop_reason=reason)
                    updated.pop("delivery", None)
                    encoded_state, encoded_messages = self.encode(updated), []
            conn.execute("UPDATE runs SET state=?,usage=?,status=?,revision=revision+1 WHERE id=?",
                         (run["state"] if state is None else encoded_state,
                          run["usage"] if usage is None else encoded_usage, status or run["status"], run_id))
            for message in encoded_messages:
                conn.execute("INSERT INTO messages(session_id,run_id,message,created_at) VALUES(?,?,?,?)",
                             (run["session_id"], run_id, message, now()))
            self._event(conn, run["session_id"], run_id, event, {
                "phase": json.loads(encoded_state or run["state"]).get("phase"),
                "status": status or run["status"], "message_count": len(encoded_messages)})
            if status in TERMINAL_STATUSES:
                conn.execute("UPDATE tasks SET status=?,revision=revision+1 WHERE id=?", (status, run["task_id"]))
        return self.store.get_run(run_id)

    def abandon(self, run_id, *, idle_seconds):
        """Close a run left 'running' by a driver that died: no live lease and no event for idle_seconds.

        The run becomes failed with stop_reason 'abandoned'. A request that was in flight is recorded
        as having unknown completion and cost; nothing is replayed. Returns None when the run is not
        abandoned (terminal, leased, or recently active).
        """
        with self.store.transaction(write=True) as conn:
            run = self._run(conn, run_id)
            if run["status"] != "running":
                return None
            lease = conn.execute("SELECT * FROM runtime_leases WHERE session_id=?", (run["session_id"],)).fetchone()
            if lease and lease["run_id"] == run_id and self._lease_live(lease):
                return None
            last = conn.execute("SELECT MAX(created_at) FROM events WHERE run_id=?", (run_id,)).fetchone()[0] or run["created_at"]
            if (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds() < idle_seconds:
                return None
            state = json.loads(run["state"]) if isinstance(run["state"], str) else dict(run["state"] or {})
            in_flight = state.get("phase") == "requesting"
            state.update(phase="finished", stop_reason="abandoned", error={
                "code": "abandoned", "message": "Driver exited without finishing; closed by host reconciliation"
                + ("; an in-flight model request has unknown completion and cost" if in_flight else "")})
            usage = json.loads(run["usage"]) if isinstance(run.get("usage"), str) else dict(run.get("usage") or {})
            request = state.pop("pending_request", None) if in_flight else None
            if request:
                from .engine import AgentEngine
                usage = AgentEngine._usage(usage, {}, request)
            conn.execute("UPDATE runs SET status='failed',state=?,usage=?,revision=revision+1 WHERE id=?",
                         (canonical_json(self.store.redact(state)), canonical_json(usage), run_id))
            conn.execute("UPDATE tasks SET status='failed',revision=revision+1 WHERE id=?", (run["task_id"],))
            self._event(conn, run["session_id"], run_id, "runtime.abandoned",
                        {"last_activity": last, "in_flight_request": in_flight})
        return self.store.get_run(run_id)

    def cancel(self, run_id):
        with self.store.transaction(write=True) as conn:
            run = self._run(conn, run_id)
            if run["status"] not in TERMINAL_STATUSES:
                conn.execute("UPDATE runs SET status='cancelled',revision=revision+1 WHERE id=?", (run_id,))
                conn.execute("UPDATE tasks SET status='cancelled',revision=revision+1 WHERE id=?", (run["task_id"],))
                self._event(conn, run["session_id"], run_id, "runtime.cancel_requested", {
                    "remote_effects": "Local cancellation does not prove remote requests stopped or were not billed"})
        return self.store.get_run(run_id)

    def ask(self, run_id, call_id, question):
        text_value(question, "Question", 4000)
        if not isinstance(call_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", call_id):
            raise ValueError("Invalid clarification call ID")
        with self.store.transaction(write=True) as conn:
            run = self._open_run(conn, run_id)
            row = conn.execute("SELECT * FROM runtime_inputs WHERE run_id=? AND call_id=?", (run_id, call_id)).fetchone()
            if row:
                if row["question"] != self.store.redact(question): raise ValueError("Conflicting clarification call")
                return self.store.redact(dict(row))
            identifier = new_id("input")
            conn.execute("INSERT INTO runtime_inputs VALUES(?,?,?,?,NULL)", (identifier, run_id, call_id, self.store.redact(question)))
            self._event(conn, run["session_id"], run_id, "runtime.input_requested", {"id": identifier, "call_id": call_id, "question": question})
        return {"id": identifier, "run_id": run_id, "call_id": call_id, "question": self.store.redact(question), "answer": None}

    def pending_inputs(self, run_id):
        self.store.get_run(run_id)
        with self.store.transaction() as conn:
            return self.store.redact([dict(row) for row in conn.execute(
                "SELECT * FROM runtime_inputs WHERE run_id=? AND answer IS NULL ORDER BY rowid", (run_id,))])

    def input_result(self, run_id, call_id):
        self.store.get_run(run_id)
        with self.store.transaction() as conn:
            row = conn.execute("SELECT * FROM runtime_inputs WHERE run_id=? AND call_id=?", (run_id, call_id)).fetchone()
        if row is None or row["answer"] is None: return None
        return self.store.redact({"answer": row["answer"], "source": "host_user", "request_id": row["id"]})

    def answer(self, run_id, request_id, answer):
        text_value(answer, "User answer", 20000)
        with self.store.transaction(write=True) as conn:
            run = self._open_run(conn, run_id)
            changed = conn.execute("UPDATE runtime_inputs SET answer=? WHERE run_id=? AND id=? AND answer IS NULL",
                                   (self.store.redact(answer), run_id, request_id)).rowcount
            if not changed: raise ValueError("Clarification is missing or already resolved")
            self._event(conn, run["session_id"], run_id, "runtime.input_answered", {"id": request_id, "answer": answer, "source": "host_user"})

    def set_plan(self, run_id, steps):
        if not isinstance(steps, list) or len(steps) > 30: raise ValueError("Plan must have at most30 steps")
        for item in steps:
            if (not isinstance(item, dict) or set(item) != {"step", "status"}
                    or item["status"] not in {"pending", "in_progress", "completed"}): raise ValueError("Invalid public plan step")
            text_value(item["step"], "Step", 1000)
        if sum(item["status"] == "in_progress" for item in steps) > 1: raise ValueError("Only one plan step may be in progress")
        with self.store.transaction(write=True) as conn:
            run = self._open_run(conn, run_id)
            conn.execute("INSERT OR REPLACE INTO runtime_plans VALUES(?,?)", (run_id, self.encode(steps)))
            self._event(conn, run["session_id"], run_id, "runtime.plan_updated", {"steps": steps})
        return self.get_plan(run_id)

    def get_plan(self, run_id):
        self.store.get_run(run_id)
        with self.store.transaction() as conn:
            row = conn.execute("SELECT steps FROM runtime_plans WHERE run_id=?", (run_id,)).fetchone()
        return self.store.redact(json.loads(row[0]) if row else [])
