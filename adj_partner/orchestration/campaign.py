"""Persistent, serial campaign queue over public dataset projections only."""
import hashlib
import json
import os
from pathlib import Path
import socket
import time
from .contracts import TaskSpec, DEFAULT_BUDGET, implementation_hash
from .exports import verified_result
from .identity import provider_identity
from ..tools.catalog import canonical_json


def dataset_rows(path):
    raw = Path(path).expanduser().read_bytes()
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError('Dataset exceeds 64 MiB')
    material = raw.decode('utf-8-sig')
    if material.lstrip().startswith('['):
        rows = json.loads(material)
    else:
        rows = [json.loads(line) for line in material.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError('Dataset must be a JSON array or JSONL rows')
    return hashlib.sha256(raw).hexdigest(), rows


MAX_ITEM_RETRIES = 2
STALE_CLAIM_SECONDS = 6 * 3600
TRANSIENT_MARKERS = ('provider_connection', 'connection failed', 'connecterror', 'connecttimeout', 'http_429',
                     '(429)', '(503)', '(529)', 'rate limit')


def _transient(error):
    text = json.dumps(error or {}, sort_keys=True).lower()
    return any(marker in text for marker in TRANSIENT_MARKERS)


def _claim_alive(claim):
    """A claim is live if its process still runs on this host, or if it is recent on another host."""
    if claim.get('host') == socket.gethostname() and isinstance(claim.get('pid'), int):
        if claim['pid'] == os.getpid():
            return True
        try:
            os.kill(claim['pid'], 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    at = claim.get('at')
    return isinstance(at, (int, float)) and time.time() - at < STALE_CLAIM_SECONDS


def resume_issue(framework, item):
    """Conservative resume guard; completed receipts remain reusable."""
    if item['state'].get('phase') == 'done':
        verified_result(framework, item['fingerprint'])
        return None
    if item['spec']['implementation_sha256'] != implementation_hash():
        return 'Implementation changed; create a new versioned task to execute current code'
    for rid in item['state'].get('attempts', []):
        run = framework.store.get_run(rid)
        if run['usage'].get('unknown_requests', 0):
            return 'An earlier runtime request has unknown completion or usage; inspect its trace before resuming'
        with framework.store.transaction() as conn:
            rows = conn.execute('SELECT status FROM model_invocations_v1 WHERE run_id=?', (rid,)).fetchall()
        if any(row[0] != 'completed' for row in rows):
            return 'An earlier auxiliary request is pending or failed and cannot be replayed'
    return None


class Campaign:
    def __init__(self, framework):
        self.framework, self.store = framework, framework.store
        with self.store.transaction(write=True) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS workflow_campaigns_v1 (
                id TEXT PRIMARY KEY, metadata TEXT NOT NULL)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS workflow_campaign_items_v1 (
                campaign_id TEXT NOT NULL REFERENCES workflow_campaigns_v1(id), row_index INTEGER NOT NULL,
                record TEXT NOT NULL, PRIMARY KEY(campaign_id,row_index))''')

    def get(self, campaign_id):
        with self.store.transaction() as conn:
            row = conn.execute('SELECT metadata FROM workflow_campaigns_v1 WHERE id=?', (campaign_id,)).fetchone()
            if row is None:
                raise KeyError('Unknown campaign')
            result = json.loads(row[0])
            result['items'] = [json.loads(r[0]) for r in conn.execute(
                'SELECT record FROM workflow_campaign_items_v1 WHERE campaign_id=? ORDER BY row_index', (campaign_id,))]
        return self.store.redact(result)

    def list(self):
        with self.store.transaction() as conn:
            keys = [row[0] for row in conn.execute('SELECT id FROM workflow_campaigns_v1 ORDER BY rowid')]
        return [self.get(key) for key in keys]

    def prepare(self, dataset, *, task_type, limit=None, offset=0, model='gpt-5.6-luna', budget=None):
        if task_type not in {'qa', 'design'}:
            raise ValueError('Campaign task type must be qa or design')
        if type(offset) is not int or offset < 0 or (limit is not None and (type(limit) is not int or limit <= 0)):
            raise ValueError('Offset must be nonnegative; limit must be positive')
        effective_budget = {**DEFAULT_BUDGET, **(budget or {})}
        # Validate shared configuration even for an empty selection.
        prototype = TaskSpec(task_type, 'Campaign configuration', model=model, budget=effective_budget)
        sha, rows = dataset_rows(dataset)
        metadata = {'schema_version': 1, 'dataset_sha256': sha, 'task_type': task_type,
                    'offset': offset, 'limit': limit, 'model': model, 'budget': effective_budget,
                    'concurrency': 1, 'workflow_version': prototype.workflow_version,
                    'implementation_sha256': prototype.implementation_sha256,
                    'provider_sha256': provider_identity(self.framework.service)}
        campaign_id = hashlib.sha256(canonical_json(metadata).encode()).hexdigest()
        metadata['id'] = campaign_id
        with self.store.transaction(write=True) as conn:
            conn.execute('INSERT OR IGNORE INTO workflow_campaigns_v1 VALUES(?,?)',
                         (campaign_id, canonical_json(metadata)))
        existing = {i['row_index'] for i in self.get(campaign_id)['items']}
        stop = len(rows) if limit is None else min(len(rows), offset + limit)
        for index in range(offset, stop):
            if index in existing:
                continue
            row = rows[index]
            item = {'row_index': index, 'row_id': str(index), 'fingerprint': None, 'status': 'invalid', 'error': None}
            try:
                if not isinstance(row, dict):
                    raise ValueError('Row requires a public question string')
                question = next((row[key] for key in ('question', 'Question:')
                                 if isinstance(row.get(key), str) and row[key].strip()), None)
                if question is None:
                    raise ValueError('Row requires a public question string')
                identifier = row.get('id', str(index))
                if not isinstance(identifier, (str, int)) or isinstance(identifier, bool):
                    raise ValueError('Row id must be text or integer')
                # Exact byte hash plus position disambiguates repeated source IDs.
                item['row_id'] = str(index) + ':' + str(identifier)
                spec = TaskSpec(task_type, question, dataset_sha256=sha,
                                row_id=item['row_id'], model=model, budget=effective_budget,
                                implementation_sha256=prototype.implementation_sha256,
                                workflow_version=prototype.workflow_version)
                submitted = self.framework.submit(spec)
                item.update(question=spec.question, fingerprint=submitted['fingerprint'], status='queued')
            except Exception as exc:
                # Never include raw row contents or hidden labels in an error record.
                item['error'] = {'code': type(exc).__name__, 'message': 'Row validation or submission failed'}
            with self.store.transaction(write=True) as conn:
                conn.execute('INSERT OR IGNORE INTO workflow_campaign_items_v1 VALUES(?,?,?)',
                             (campaign_id, index, canonical_json(self.store.redact(item))))
        return self.get(campaign_id)

    def _save_item(self, campaign_id, item):
        with self.store.transaction(write=True) as conn:
            conn.execute('UPDATE workflow_campaign_items_v1 SET record=? WHERE campaign_id=? AND row_index=?',
                         (canonical_json(self.store.redact(item)), campaign_id, item['row_index']))

    def _requeue(self, campaign_id, *, retry_failed):
        """Reclaim items left 'running' by a dead process and, on request, re-queue transient failures.

        A re-queued item still passes resume_issue before execution, so a request with unknown
        completion or cost is never replayed.
        """
        for item in self.get(campaign_id)['items']:
            claim = item.get('claimed_by') or {}
            if item['status'] == 'running' and not _claim_alive(claim):
                item.update(status='queued', reclaimed=item.get('reclaimed', 0) + 1)
                self._save_item(campaign_id, item)
            elif (retry_failed and item['status'] == 'failed' and _transient(item.get('error'))
                    and item.get('retries', 0) < MAX_ITEM_RETRIES):
                item.update(status='queued', retries=item.get('retries', 0) + 1, previous_error=item.get('error'))
                self._save_item(campaign_id, item)

    def execute(self, campaign_id, *, retry_failed=False):
        self._requeue(campaign_id, retry_failed=retry_failed)
        for item in self.get(campaign_id)['items']:
            if item['status'] == 'succeeded':
                try:
                    _, current, _ = verified_result(self.framework, item['fingerprint'])
                    if current['execution_status'] != 'succeeded':
                        raise ValueError('Workflow task no longer has a successful result')
                except Exception as exc:
                    item.update(status='needs_attention', error={'code': 'cached_integrity_failure', 'message': self.store.redact(str(exc))})
                    with self.store.transaction(write=True) as conn:
                        conn.execute('UPDATE workflow_campaign_items_v1 SET record=? WHERE campaign_id=? AND row_index=?',
                                     (canonical_json(self.store.redact(item)), campaign_id, item['row_index']))
            if item['status'] != 'queued':
                continue
            before = canonical_json(item)
            item['status'] = 'running'
            item['claimed_by'] = {'host': socket.gethostname(), 'pid': os.getpid(), 'at': time.time()}
            with self.store.transaction(write=True) as conn:
                claimed = conn.execute('''UPDATE workflow_campaign_items_v1 SET record=?
                    WHERE campaign_id=? AND row_index=? AND record=?''',
                    (canonical_json(item), campaign_id, item['row_index'], before)).rowcount
            if claimed != 1:
                continue
            try:
                task = self.framework.repository.get(item['fingerprint'])
                issue = resume_issue(self.framework, task)
                if issue:
                    item.update(status='needs_attention', error={'code': 'uncertain_previous_request', 'message': issue})
                else:
                    result = self.framework.run(item['fingerprint'])
                    item.update(status=result['execution_status'], result=result, error=result.get('error'))
            except Exception as exc:
                item.update(status='failed', error={'code': type(exc).__name__, 'message': self.store.redact(str(exc))})
            with self.store.transaction(write=True) as conn:
                conn.execute('UPDATE workflow_campaign_items_v1 SET record=? WHERE campaign_id=? AND row_index=?',
                             (canonical_json(self.store.redact(item)), campaign_id, item['row_index']))
        return self.get(campaign_id)
