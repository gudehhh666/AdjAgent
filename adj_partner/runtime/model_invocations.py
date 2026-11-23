"""Durable auxiliary model requests sharing runtime leases and bounded accounting.

Responses are immutable receipts. A pending or uncertain request is never replayed.
The main engine retains its existing request journal; totals include both paths.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from threading import Event, Thread
import time

from ..contracts import ModelReply, ToolCall
from ..models.base import ProviderFailure, supports_tool_choice
from ..models.requests import public_request
from ..storage import TERMINAL_STATUSES
from ..tools.catalog import canonical_json
from .journal import RunJournal, LeaseLost


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def fenced_guard(guard, journal, run_id, owner):
    """Attach a transaction-local lease fence to an existing boundary guard."""
    def fence(conn):
        journal._fence(conn, run_id, owner)
        journal._open_run(conn, run_id)
    guard.fence, guard.run_id = fence, run_id
    return guard


def inherit_guard(guard, parent):
    """Preserve transaction fencing when a controller adds boundary checks."""
    for key in ('fence', 'run_id', 'remaining_seconds'):
        if hasattr(parent, key): setattr(guard, key, getattr(parent, key))
    return guard


@contextmanager
def leased_run(store, run_id):
    journal = RunJournal(store)
    owner = journal.acquire(run_id)
    stop, lost = Event(), Event()
    def heartbeat():
        while not stop.wait(1):
            try: journal.renew(run_id, owner)
            except Exception:
                lost.set()
                return
    def guard():
        if lost.is_set(): raise LeaseLost('Runtime lease lost')
        journal.renew(run_id, owner)
        if store.get_run(run_id)['status'] in TERMINAL_STATUSES:
            raise RuntimeError('Terminal run cannot initiate model operations')
    fenced_guard(guard, journal, run_id, owner)
    thread = Thread(target=heartbeat, daemon=True, name='adj-stage-lease')
    thread.start()
    try:
        guard()
        yield guard
    finally:
        stop.set()
        thread.join(timeout=2)
        journal.release(run_id, owner)


class InvocationLedger:
    def __init__(self, store):
        self.store = store
        with store.transaction(write=True) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS model_invocations_v1 (
                run_id TEXT NOT NULL REFERENCES runs(id), request_key TEXT NOT NULL,
                role TEXT NOT NULL, identity TEXT NOT NULL, status TEXT NOT NULL,
                reservation TEXT NOT NULL, receipt TEXT, usage TEXT NOT NULL,
                PRIMARY KEY(run_id, request_key))''')
            conn.execute('''CREATE TABLE IF NOT EXISTS model_budget_clocks_v1 (
                run_id TEXT PRIMARY KEY REFERENCES runs(id), started_at REAL NOT NULL,
                elapsed_base REAL NOT NULL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS workflow_checkpoints_v1 (
                run_id TEXT PRIMARY KEY REFERENCES runs(id), stage TEXT NOT NULL,
                payload TEXT NOT NULL, started_at REAL NOT NULL)''')

    @staticmethod
    def _fence(conn, rid, guard):
        if getattr(guard, 'run_id', None) != rid or not callable(getattr(guard, 'fence', None)):
            raise ValueError('Model operations require a run-bound transaction fence')
        guard.fence(conn)

    def _remaining(self, conn, run):
        # The budget epoch belongs to the run, independently of workflow state.
        # Old checkpoints seed migrated runs without forgiving elapsed time.
        clock = conn.execute('SELECT * FROM model_budget_clocks_v1 WHERE run_id=?', (run['id'],)).fetchone()
        if clock is None:
            old = conn.execute('SELECT started_at FROM workflow_checkpoints_v1 WHERE run_id=?', (run['id'],)).fetchone()
            started = old[0] if old else time.time()
            base = run['state'].get('elapsed_active_seconds', 0.)
            conn.execute('INSERT INTO model_budget_clocks_v1 VALUES(?,?,?)', (run['id'], started, base))
        else: started, base = clock['started_at'], clock['elapsed_base']
        return run['budget']['wall_time_seconds'] - base - max(0., time.time() - started)

    def finish(self, rid, *, status, state=None, guard, on_commit=None):
        """Commit outcomes atomically; on_commit(conn) must only perform local SQL.

        The callback runs inside this transaction, before its actual commit. If
        it fails, the run, task, events and callback writes all roll back.
        """
        if status not in TERMINAL_STATUSES: raise ValueError('finish requires a terminal outcome')
        if on_commit is not None and not callable(on_commit): raise ValueError('on_commit must be callable')
        guard()
        with self.store.transaction(write=True) as conn:
            self._fence(conn, rid, guard)
            run = self.store._get(conn, 'runs', rid)
            if status == 'succeeded':
                uses = [json.loads(r[0]) for r in conn.execute('SELECT usage FROM model_invocations_v1 WHERE run_id=?', (rid,))]
                if (self._remaining(conn, run) <= 0
                    or len(uses) + run['state'].get('steps', 0) > run['budget']['max_steps']
                    or any(run['usage'].get(k, 0) + sum(u.get(k, 0) for u in uses) > run['budget'][limit]
                           for k, limit in [('accounted_input_tokens', 'max_input_tokens'), ('accounted_output_tokens', 'max_output_tokens')])):
                    raise RuntimeError('Model request budget exhausted before delivery')
            encoded = canonical_json(self.store.redact(run['state'] if state is None else state))
            conn.execute('UPDATE runs SET status=?,state=?,revision=revision+1 WHERE id=?', (status, encoded, rid))
            conn.execute('UPDATE tasks SET status=?,revision=revision+1 WHERE id=?', (status, run['task_id']))
            self.store._event(conn, run['session_id'], 'runtime.auxiliary_finished', {'status': status}, rid)
            if on_commit is not None: on_commit(conn)
        return self.store.get_run(rid)

    def state(self, rid):
        with self.store.transaction() as conn:
            row = conn.execute('SELECT * FROM workflow_checkpoints_v1 WHERE run_id=?', (rid,)).fetchone()
        return {'stage': row['stage'], 'payload': json.loads(row['payload'])} if row else None

    def checkpoint(self, rid, stage, payload, *, guard):
        guard()
        encoded = canonical_json(self.store.redact(payload))
        with self.store.transaction(write=True) as conn:
            self._fence(conn, rid, guard)
            self._remaining(conn, self.store._get(conn, 'runs', rid))
            conn.execute('''INSERT INTO workflow_checkpoints_v1 VALUES(?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET stage=excluded.stage,payload=excluded.payload''',
                         (rid, stage, encoded, time.time()))

    def usage(self, rid):
        result = dict(requests=0, input_tokens=0, output_tokens=0,
                      accounted_input_tokens=0, accounted_output_tokens=0, unknown_requests=0)
        with self.store.transaction() as conn:
            rows = conn.execute('SELECT usage FROM model_invocations_v1 WHERE run_id=?', (rid,)).fetchall()
        for row in rows:
            result['requests'] += 1
            value = json.loads(row['usage'])
            for k in result:
                if k != 'requests': result[k] += value.get(k, 0)
        return result

    def _account(self, actual, estimated, maximum):
        actual = actual if isinstance(actual, dict) else {}
        result = {}
        known = True
        for k, fallback in [('input_tokens', estimated), ('output_tokens', maximum)]:
            v = actual.get(k)
            valid = type(v) is int and v >= 0
            result[k] = v if valid else 0
            result['accounted_' + k] = v if valid else fallback
            known = known and valid
        result['unknown_requests'] = int(not known)
        return result

    def _required_action_error(self, reply, tool_choice):
        if (tool_choice == 'required_or_refuse' and not reply.tool_calls
                and not (isinstance(reply.refusal, str) and len(reply.refusal) >= 1)):
            actual = reply.usage if isinstance(reply.usage, dict) else {}
            return ProviderFailure('required_action_missing', 'Required decision must contain an action or nonempty refusal',
                                   usage=actual, cost_unknown=bool(self._account(actual, 0, 0)['unknown_requests']))
        return None

    def complete(self, sid, rid, port, messages, tools, *, role, key,
                 max_output_tokens, timeout_seconds, guard, tool_choice='auto'):
        guard()
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('Invalid output budget')
        if not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError('Invalid timeout')
        self.store.register_secrets(getattr(port, 'secrets', ()))
        request = self.store.redact({'messages': messages, 'tools': tools, 'model': port.metadata,
                                    'role': role, 'max_output_tokens': max_output_tokens})
        if tool_choice != 'auto':
            public_request(messages, tools, max_output_tokens, timeout_seconds, tool_choice=tool_choice)
            request['tool_choice'] = tool_choice
        identity = digest(request)
        run = self.store.get_run(rid)
        if run['session_id'] != sid: raise ValueError('Run scope mismatch')
        estimated = max(1, len(canonical_json(request).encode()) // 3 + 1)
        cached_failure = None
        with self.store.transaction(write=True) as conn:
            self._fence(conn, rid, guard)
            run = self.store._get(conn, 'runs', rid)
            old = conn.execute('SELECT * FROM model_invocations_v1 WHERE run_id=? AND request_key=?', (rid, key)).fetchone()
            if old:
                if old['identity'] != identity: raise ValueError('Request identity changed')
                if old['status'] != 'completed': raise RuntimeError('Unknown or failed request cannot be replayed')
                raw = json.loads(old['receipt'])
                cached = ModelReply(**{**raw, 'tool_calls': [ToolCall(**c) for c in raw['tool_calls']]})
                cached_failure = self._required_action_error(cached, tool_choice)
                if cached_failure is None: return cached
                # Commit the failed state before raising; preserve the original receipt and usage.
                conn.execute("UPDATE model_invocations_v1 SET status='failed' WHERE run_id=? AND request_key=?",
                             (rid, key))
            else:
                if not supports_tool_choice(port, tool_choice):
                    raise ProviderFailure('unsupported_tool_choice', 'Model port does not support the requested tool choice',
                                          cost_unknown=False)
                # Include engine requests already accounted in the same run.
                rows = conn.execute('SELECT usage FROM model_invocations_v1 WHERE run_id=?', (rid,)).fetchall()
                usages = [json.loads(r['usage']) for r in rows]
                used_in = run['usage'].get('accounted_input_tokens', 0) + sum(u['accounted_input_tokens'] for u in usages)
                used_out = run['usage'].get('accounted_output_tokens', 0) + sum(u['accounted_output_tokens'] for u in usages)
                remaining = self._remaining(conn, run)
                if (len(rows) + run['state'].get('steps', 0) >= run['budget']['max_steps']
                    or estimated + used_in > run['budget']['max_input_tokens']
                    or max_output_tokens + used_out > run['budget']['max_output_tokens'] or remaining <= 0):
                    raise RuntimeError('Model request budget exhausted')
                remaining_guard = getattr(guard, 'remaining_seconds', None)
                if remaining_guard is not None:
                    remaining = min(remaining, remaining_guard())
                    if remaining <= 0: raise RuntimeError('Model request budget exhausted')
                timeout_seconds = min(timeout_seconds, remaining)
                reserve = {'estimated_input_tokens': estimated, 'max_output_tokens': max_output_tokens,
                           'timeout_seconds': timeout_seconds, 'issued_at': time.time()}
                if tool_choice != 'auto':
                    reserve['tool_choice'] = tool_choice
                usage = self._account({}, estimated, max_output_tokens)
                conn.execute('INSERT INTO model_invocations_v1 VALUES(?,?,?,?,?,?,?,?)',
                             (rid, key, role, identity, 'pending', canonical_json(reserve), None, canonical_json(usage)))
        if cached_failure is not None:
            raise cached_failure
        # Intent was committed before the external effect. SIGKILL leaves pending.
        try:
            guard()
            mode = {'tool_choice': tool_choice} if tool_choice != 'auto' else {}
            reply = port.complete(request['messages'], request['tools'], max_output_tokens=max_output_tokens,
                                  timeout_seconds=timeout_seconds, **mode)
            from .engine import AgentEngine
            raw = AgentEngine._valid_reply(reply, set())
            usage = self._account(reply.usage, estimated, max_output_tokens)
            action_failure = self._required_action_error(reply, tool_choice)
            if action_failure is not None:
                raise action_failure
            with self.store.transaction(write=True) as conn:
                conn.execute('UPDATE model_invocations_v1 SET status=?,receipt=?,usage=? WHERE run_id=? AND request_key=?',
                             ('completed', canonical_json(self.store.redact(raw)), canonical_json(usage), rid, key))
            guard()
            return reply
        except Exception as exc:
            with self.store.transaction(write=True) as conn:
                conn.execute("UPDATE model_invocations_v1 SET status='failed',usage=? WHERE run_id=? AND request_key=? AND status='pending'",
                             (canonical_json(self._account(getattr(exc, 'usage', {}), estimated, max_output_tokens)), rid, key))
            raise
