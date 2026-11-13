"""Workflow extension tables referencing the existing session/task/run identities."""
from contextlib import contextmanager
import json
from threading import Event,Thread
import time
from ..contracts import new_id
from ..tools.catalog import canonical_json


class WorkflowRepository:
    def __init__(self, store):
        self.store=store
        with store.transaction(write=True) as c:
            c.execute('''CREATE TABLE IF NOT EXISTS workflow_tasks_v1 (
                task_id TEXT PRIMARY KEY REFERENCES tasks(id), fingerprint TEXT NOT NULL UNIQUE,
                session_id TEXT NOT NULL REFERENCES sessions(id), initial_run_id TEXT NOT NULL REFERENCES runs(id),
                spec TEXT NOT NULL, state TEXT NOT NULL, result TEXT, owner TEXT, lease_until REAL NOT NULL DEFAULT 0)''')

    def find(self, fingerprint):
        with self.store.transaction() as c:
            row=c.execute('SELECT * FROM workflow_tasks_v1 WHERE fingerprint=?',(fingerprint,)).fetchone()
        if row is None:return None
        value=dict(row)
        for k in ('spec','state','result'):value[k]=json.loads(value[k]) if value[k] else None
        return self.store.redact(value)

    def get(self, fingerprint):
        value=self.find(fingerprint)
        if value is None:raise KeyError('Unknown workflow task')
        return value

    def bind(self,spec,sid,tid,rid):
        state={'phase':'discovery' if spec.task_type=='discovery_design' else 'retrieval','retrieval_run':rid,
               'attempts':[rid],'round':0,'started_at':None}
        with self.store.transaction(write=True) as c:
            run=c.execute('SELECT session_id,task_id FROM runs WHERE id=?',(rid,)).fetchone()
            if not run or tuple(run)!=(sid,tid):raise ValueError('Workflow binding scope mismatch')
            c.execute('INSERT OR IGNORE INTO workflow_tasks_v1(task_id,fingerprint,session_id,initial_run_id,spec,state) VALUES(?,?,?,?,?,?)',
                      (tid,spec.fingerprint,sid,rid,canonical_json(spec.to_dict()),canonical_json(state)))
        return self.get(spec.fingerprint)

    def list(self):
        with self.store.transaction() as c:keys=[r[0] for r in c.execute('SELECT fingerprint FROM workflow_tasks_v1 ORDER BY rowid')]
        return [self.get(k) for k in keys]

    @contextmanager
    def owned(self, fingerprint):
        owner=new_id('workflow_owner');stop,lost=Event(),Event()
        with self.store.transaction(write=True) as c:
            row=c.execute('SELECT lease_until FROM workflow_tasks_v1 WHERE fingerprint=?',(fingerprint,)).fetchone()
            if row is None:raise KeyError('Unknown workflow task')
            if row[0]>time.time():raise RuntimeError('Workflow already owned')
            c.execute('UPDATE workflow_tasks_v1 SET owner=?,lease_until=? WHERE fingerprint=?',(owner,time.time()+15,fingerprint))
        def guard():
            if lost.is_set():raise RuntimeError('Workflow ownership lost')
            with self.store.transaction(write=True) as c:
                n=c.execute('UPDATE workflow_tasks_v1 SET lease_until=? WHERE fingerprint=? AND owner=? AND lease_until>?',
                            (time.time()+15,fingerprint,owner,time.time())).rowcount
                if n!=1:raise RuntimeError('Workflow ownership lost')
        guard.owner=owner
        def heartbeat():
            while not stop.wait(1):
                try:guard()
                except Exception:lost.set();return
        thread=Thread(target=heartbeat,daemon=True);thread.start()
        try:yield guard
        finally:
            stop.set();thread.join(timeout=2)
            with self.store.transaction(write=True) as c:c.execute('UPDATE workflow_tasks_v1 SET owner=NULL,lease_until=0 WHERE fingerprint=? AND owner=?',(fingerprint,owner))

    def save(self,fingerprint,state,*,guard,result=None):
        guard()
        with self.store.transaction(write=True) as c:
            n=c.execute('UPDATE workflow_tasks_v1 SET state=?,result=? WHERE fingerprint=? AND owner=? AND lease_until>?',
                        (canonical_json(self.store.redact(state)),canonical_json(self.store.redact(result)) if result else None,
                         fingerprint,guard.owner,time.time())).rowcount
            if n!=1:raise RuntimeError('Workflow ownership lost')
        return self.get(fingerprint)

    def create_attempt(self,fingerprint,state,field,budget,*,guard,message=None):
        """Create a run and bind its checkpoint atomically; a crash cannot orphan it."""
        from datetime import datetime,timezone
        from copy import deepcopy
        guard();updated=deepcopy(state);rid=new_id('run');stamp=datetime.now(timezone.utc).isoformat()
        updated[field]=rid;updated['attempts'].append(rid)
        initial={'schema_version':1,'phase':'ready','steps':0,'elapsed_active_seconds':0.}
        with self.store.transaction(write=True) as conn:
            row=conn.execute('SELECT * FROM workflow_tasks_v1 WHERE fingerprint=? AND owner=? AND lease_until>?',
                             (fingerprint,guard.owner,time.time())).fetchone()
            if row is None:raise RuntimeError('Workflow ownership lost')
            persisted=json.loads(row['state'])
            if persisted.get(field):
                state.clear();state.update(persisted)
                return state[field]
            conn.execute('INSERT INTO runs VALUES(?,?,?,?,0,?,?,?,?)',
                (rid,row['session_id'],row['task_id'],'created',canonical_json(budget),'{}',canonical_json(initial),stamp))
            conn.execute('UPDATE tasks SET status=?,revision=revision+1 WHERE id=?',('open',row['task_id']))
            self.store._event(conn,row['session_id'],'run.created',self.store._get(conn,'runs',rid),rid)
            if message is not None:
                message=self.store.redact(message)
                conn.execute('INSERT INTO messages(session_id,run_id,message,created_at) VALUES(?,?,?,?)',
                             (row['session_id'],rid,canonical_json(message),stamp))
                self.store._event(conn,row['session_id'],'message.appended',{'message':message},rid)
            conn.execute('UPDATE workflow_tasks_v1 SET state=?,result=NULL WHERE fingerprint=?',
                         (canonical_json(self.store.redact(updated)),fingerprint))
        state.clear();state.update(updated)
        return rid

    def save_in_transaction(self,conn,fingerprint,state,result,*,guard):
        """Commit workflow delivery in the same fenced transaction as its run."""
        n=conn.execute('UPDATE workflow_tasks_v1 SET state=?,result=? WHERE fingerprint=? AND owner=? AND lease_until>?',
            (canonical_json(self.store.redact(state)),canonical_json(self.store.redact(result)),fingerprint,guard.owner,time.time())).rowcount
        if n!=1:raise RuntimeError('Workflow ownership lost')
