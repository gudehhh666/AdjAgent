"""Immutable discovery and design records in the host database.

Bindings freeze research revisions and material hashes. Feedback resolution is a
structural change check, not a claim that the change is scientifically adequate.
Artifact reads happen outside SQLite writer transactions; referenced heads and
artifact metadata are checked again in the committing transaction.
"""
from copy import deepcopy
import json

from .design import validate_design
from .validation import bounded_json, digest, identifier, ids, strings, text

KINDS = frozenset({'finding', 'design_brief', 'design_revision', 'design_feedback', 'experiment_result'})
STAGES = frozenset({'discovery', 'analysis', 'evidence', 'hypothesis', 'decision', 'report'})


def _field(value, path):
    if path == '$':
        def semantic(item):
            if isinstance(item, dict): return {k: semantic(v) for k, v in item.items() if k != 'claim_ids'}
            if isinstance(item, list): return [semantic(v) for v in item]
            return item
        return semantic(value)
    try:
        for key in path.split('.'):
            value = value[int(key)] if isinstance(value, list) and key.isdecimal() else value[key]
        return value
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError('Unknown affected design field: ' + path) from exc


class DesignRecords:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self.store.transaction(write=True) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS design_records_v1 (
                seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL, session_id TEXT NOT NULL REFERENCES sessions(id),
                run_id TEXT NOT NULL REFERENCES runs(id), task_id TEXT NOT NULL REFERENCES tasks(id),
                record TEXT NOT NULL)''')
            conn.execute('CREATE INDEX IF NOT EXISTS design_scope_v1 ON design_records_v1(session_id,task_id,kind)')
            conn.execute('CREATE INDEX IF NOT EXISTS design_run_kind_v1 ON design_records_v1(session_id,task_id,run_id,kind)')

    def _design_get(self, conn, sid, tid, record_id, kind=None):
        row = conn.execute('SELECT record FROM design_records_v1 WHERE id=? AND session_id=? AND task_id=?',
                           (identifier(record_id), sid, tid)).fetchone()
        if row is None:
            raise KeyError('Unknown or out-of-scope design record: ' + record_id)
        rec = json.loads(row[0])
        if kind and rec['kind'] != kind:
            raise ValueError('Expected ' + kind)
        if rec['hash'] != digest({k: v for k, v in rec.items() if k != 'hash'}):
            raise ValueError('Design record hash integrity mismatch')
        return rec

    def _design_binding_record(self, conn, sid, tid, binding):
        if binding['entity'] in KINDS:
            return self._design_get(conn, sid, tid, binding['id'])
        rec = self._get(conn, binding['entity'], sid, tid, binding['id'])
        if rec['task_id'] != tid:
            raise PermissionError('Research reference must belong to the same task')
        return rec

    def _design_bind(self, conn, sid, tid, entity, record_id, bindings, *, inherit=True):
        rec = self._design_binding_record(conn, sid, tid, {'entity': entity, 'id': record_id})
        if entity in KINDS and rec['kind'] != entity:
            raise ValueError('Expected ' + entity)
        binding = {'entity': entity, 'id': record_id,
                   'revision': rec.get('revision'), 'hash': digest(rec)}
        if (entity, record_id) in bindings and bindings[(entity, record_id)] != binding:
            raise ValueError('Conflicting stale upstream bindings: ' + record_id)
        bindings[(entity, record_id)] = binding
        if entity in KINDS and inherit:
            for b in rec['payload']['bindings']:
                key = (b['entity'], b['id'])
                if key in bindings and bindings[key] != b:
                    raise ValueError('Conflicting stale upstream bindings')
                bindings[key] = b
        elif entity == 'claim':
            for eid in rec['evidence_ids']:
                self._design_bind(conn, sid, tid, 'evidence', eid, bindings)
        elif entity == 'evidence':
            src = self._design_bind(conn, sid, tid, 'source', rec['source_id'], bindings)
            if rec['source_hash'] != src['artifact_hash']:
                raise ValueError('Evidence source hash mismatch')
        return rec

    def _design_check_bindings(self, conn, sid, tid, bindings):
        sources = []
        for binding in bindings:
            rec = self._design_binding_record(conn, sid, tid, binding)
            if digest(rec) != binding['hash'] or rec.get('revision') != binding['revision']:
                raise ValueError('Stale upstream binding: ' + binding['id'])
            if binding['entity'] == 'source':
                self._source_metadata(conn, rec)
                sources.append(rec)
        return sources

    def _create_design_record(self, sid, rid, kind, build, *, identity_fields=()):
        with self.store.transaction() as conn:
            task = self._scope(conn, sid, rid)
            bindings = {}
            payload = build(conn, task['id'], bindings)
            payload['bindings'] = [bindings[key] for key in sorted(bindings)]
            sources = self._design_check_bindings(conn, sid, task['id'], payload['bindings'])
        for src in sources:
            self._verify_source(src)
        record = self._record(kind, sid, rid, task['id'], kind=kind, payload=deepcopy(payload))
        record.pop('revision')
        record = self.store.redact(record)
        record['hash'] = digest(record)
        encoded = bounded_json(record)
        reused = False
        with self.store.transaction(write=True) as conn:
            self._scope(conn, sid, rid)
            self._design_check_bindings(conn, sid, task['id'], payload['bindings'])
            # A controller may crash after this commit and before its checkpoint.
            # Serialize lookup + insert so replay (including concurrent replay)
            # preserves the original immutable record and emits no second event.
            payload_hash = digest(record['payload'])
            query='SELECT id,record FROM design_records_v1 WHERE session_id=? AND task_id=? AND kind=?'
            params=[sid,task['id'],kind]
            if not identity_fields:query+=' AND run_id=?';params.append(rid)
            rows = conn.execute(query+' ORDER BY seq',params).fetchall()
            for row in rows:
                existing = json.loads(row[1])
                if identity_fields and all(existing['payload'].get(k)==record['payload'].get(k) for k in identity_fields):
                    if digest(existing['payload'])!=payload_hash:
                        raise ValueError('Conflicting immutable result identity; use a new result key for a distinct measurement')
                if digest(existing['payload']) == payload_hash:
                    record = self._design_get(conn, sid, task['id'], row[0], kind)
                    reused = True
                    break
            else:
                conn.execute('INSERT INTO design_records_v1(id,kind,session_id,run_id,task_id,record) VALUES(?,?,?,?,?,?)',
                             (record['id'], kind, sid, rid, task['id'], encoded))
                conn.execute('INSERT INTO research_events(session_id,run_id,kind,payload,created_at) VALUES(?,?,?,?,?)',
                             (sid, rid, 'research.' + kind + '.recorded', encoded, record['created_at']))
        return self.store.redact(record) if reused else self._mirror(kind, record)

    def create_finding(self, session_id, run_id, *, observation, evidence_ids, limitations, counterevidence_ids=None):
        text(observation, 'observation'); strings(limitations, 'limitations')
        ids(evidence_ids); counterevidence_ids = [] if counterevidence_ids is None else counterevidence_ids
        ids(counterevidence_ids)
        if not evidence_ids:
            raise ValueError('Finding requires evidence')
        def build(conn, tid, bindings):
            for eid in evidence_ids + counterevidence_ids:
                self._design_bind(conn, session_id, tid, 'evidence', eid, bindings)
            return dict(observation=observation, evidence_ids=evidence_ids, limitations=limitations,
                        counterevidence_ids=counterevidence_ids)
        return self._create_design_record(session_id, run_id, 'finding', build)

    def create_design_brief(self, session_id, run_id, *, goal, requirements, finding_ids=None, evidence_ids=None):
        text(goal, 'goal'); strings(requirements, 'requirements')
        finding_ids = [] if finding_ids is None else finding_ids
        evidence_ids = [] if evidence_ids is None else evidence_ids
        ids(finding_ids); ids(evidence_ids)
        def build(conn, tid, bindings):
            for fid in finding_ids:
                self._design_bind(conn, session_id, tid, 'finding', fid, bindings)
            for eid in evidence_ids:
                self._design_bind(conn, session_id, tid, 'evidence', eid, bindings)
            return dict(goal=goal, requirements=requirements, finding_ids=finding_ids, evidence_ids=evidence_ids)
        return self._create_design_record(session_id, run_id, 'design_brief', build)

    def create_design_revision(self, session_id, run_id, *, brief_id, design, parent_id=None, feedback_ids=None, result_ids=None):
        feedback_ids = [] if feedback_ids is None else feedback_ids
        ids(feedback_ids)
        result_ids=[] if result_ids is None else result_ids
        ids(result_ids)
        def build(conn, tid, bindings):
            self._design_bind(conn, session_id, tid, 'design_brief', brief_id, bindings)
            claims = self._list(conn, 'claim', session_id, tid)
            checked = validate_design(design, [c['id'] for c in claims])
            by_id = {c['id']: c for c in claims}
            for card in checked['verification'].get('checkpoints', []):
                if not any(by_id[cid]['evidence_ids'] for cid in card['claim_ids']):
                    raise ValueError('checkpoint requires at least one claim with bound evidence')
            linked = [*checked['requirements'], *checked['candidates'], checked['mechanism'], checked['verification']]
            for cid in {cid for item in linked for cid in item['claim_ids']}:
                self._design_bind(conn, session_id, tid, 'claim', cid, bindings)
            ancestors = set()
            parent = None
            cursor = parent_id
            while cursor:
                old = self._design_get(conn, session_id, tid, cursor, 'design_revision')
                if old['payload']['brief_id'] != brief_id:
                    raise ValueError('Parent must belong to the same brief')
                if cursor in ancestors:
                    raise ValueError('Cycle in design lineage')
                ancestors.add(cursor)
                if parent is None:
                    parent = old
                    self._design_bind(conn, session_id, tid, 'design_revision', cursor, bindings, inherit=False)
                cursor = old['payload']['parent_id']
            for fid in feedback_ids:
                fb = self._design_bind(conn, session_id, tid, 'design_feedback', fid, bindings, inherit=False)
                if fb['payload']['design_id'] not in ancestors:
                    raise ValueError('Feedback must target a parent or ancestor design')
                target = self._design_get(conn, session_id, tid, fb['payload']['design_id'], 'design_revision')
                for field in fb['payload']['affected_fields']:
                    if (_field(checked, field) == _field(target['payload']['design'], field)
                            or _field(checked, field) == _field(parent['payload']['design'], field)):
                        raise ValueError('Every affected field must have changed in the new revision')
                for eid in fb['payload']['evidence_ids']:
                    self._design_bind(conn, session_id, tid, 'evidence', eid, bindings)
            payload=dict(brief_id=brief_id, design=checked, parent_id=parent_id, feedback_ids=feedback_ids)
            if result_ids:
                self._bind_experiment_results(conn,session_id,tid,checked,parent,ancestors,result_ids,bindings)
                payload['result_ids']=result_ids
            return payload
        return self._create_design_record(session_id, run_id, 'design_revision', build)

    def record_design_feedback(self, session_id, run_id, *, design_id, target_stage, issue, affected_fields, evidence_ids=None):
        text(issue, 'issue'); strings(affected_fields, 'affected_fields', unique=True)
        if not isinstance(target_stage, str) or target_stage not in STAGES:
            raise ValueError('Unknown feedback target stage')
        if not affected_fields:
            raise ValueError('Feedback requires affected fields')
        evidence_ids = [] if evidence_ids is None else evidence_ids
        ids(evidence_ids)
        def build(conn, tid, bindings):
            target = self._design_bind(conn, session_id, tid, 'design_revision', design_id, bindings, inherit=False)
            for field in affected_fields:
                _field(target['payload']['design'], field)
            for eid in evidence_ids:
                self._design_bind(conn, session_id, tid, 'evidence', eid, bindings)
            return dict(design_id=design_id, target_stage=target_stage, issue=issue,
                        affected_fields=affected_fields, evidence_ids=evidence_ids)
        return self._create_design_record(session_id, run_id, 'design_feedback', build)

    def get_design_record(self, session_id, run_id, record_id):
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            record = self._design_get(conn, session_id, task['id'], record_id)
        return self.store.redact(record)

    def list_design_records(self, session_id, run_id, kind=None):
        if kind is not None and kind not in KINDS:
            raise ValueError('Unknown design record kind')
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            rows = conn.execute('SELECT id FROM design_records_v1 WHERE session_id=? AND task_id=? ORDER BY seq',
                                (session_id, task['id'])).fetchall()
            records = [self._design_get(conn, session_id, task['id'], row[0]) for row in rows]
        return self.store.redact([r for r in records if kind is None or r['kind'] == kind])

    def validate_design_record(self, session_id, run_id, record_id):
        issues, sources = [], []
        with self.store.transaction() as conn:
            task = self._scope(conn, session_id, run_id)
            rec = self._design_get(conn, session_id, task['id'], record_id)
            for b in rec['payload']['bindings']:
                try:
                    sources.extend(self._design_check_bindings(conn, session_id, task['id'], [b]))
                except (KeyError, ValueError, PermissionError) as exc:
                    issues.append(str(exc))
        for src in sources:
            try:
                self._verify_source(src)
            except (OSError, ValueError, KeyError) as exc:
                issues.append(src['id'] + ': ' + str(exc))
        return self.store.redact(issues)

    def design_lineage(self, session_id, run_id):
        records = self.list_design_records(session_id, run_id)
        edges, feedback = [], []
        for rec in records:
            p = rec['payload']
            for key in ('brief_id', 'parent_id', 'design_id', 'supersedes'):
                if p.get(key):
                    edges.append({'from': p[key], 'to': rec['id'], 'relation': key})
            for key in ('finding_ids', 'feedback_ids', 'result_ids'):
                for ref in p.get(key, []):
                    edges.append({'from': ref, 'to': rec['id'], 'relation': key})
            if rec['kind'] == 'design_feedback':
                resolved_by = [d['id'] for d in records if d['kind'] == 'design_revision'
                               and rec['id'] in d['payload']['feedback_ids']]
                feedback.append({'feedback_id': rec['id'], 'design_id': p['design_id'],
                                 'resolved': bool(resolved_by), 'resolved_by': resolved_by})
        return {'records': records, 'edges': edges, 'feedback': feedback}
