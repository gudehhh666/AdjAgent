"""Auditable exports: independent statuses and verified host artifact bytes."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from .contracts import TaskSpec, result_record


def _atomic(path, content):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verified_result(framework, fingerprint):
    item = framework.repository.get(fingerprint)
    result = dict(item['result'] or result_record(TaskSpec.from_dict(item['spec']), 'queued'))
    if result.get('task_fingerprint') != fingerprint:
        raise ValueError('Result fingerprint differs from task identity')
    delivery = result.get('delivery') or {}
    if result.get('execution_status') == 'succeeded':
        rid = item['state'].get('stage_run') or item['initial_run_id']
        run = framework.store.get_run(rid)
        if run['status'] != 'succeeded':
            raise ValueError('Workflow result is pending run finalization; inspect or resume before export')
    artifact_id = delivery.get('artifact_id')
    if delivery.get('report_id'):
        rid = item['state'].get('stage_run') or item['initial_run_id']
        research = framework.service.research
        accepted = research.accepted_report(item['session_id'], rid, delivery['report_id'])
        report = accepted['report']
        design_id = delivery.get('design_revision_id')
        if not design_id:
            raise ValueError('Report delivery is missing its design revision binding')
        design = research.get_design_record(item['session_id'], rid, design_id)
        issues = research.validate_design_record(item['session_id'], rid, design_id)
        if issues:
            raise ValueError('Stale design revision bindings: ' + '; '.join(issues))
        if design['kind'] != 'design_revision' or design['payload']['design'] != report.get('research_design'):
            raise ValueError('Report does not match its bound design revision')
        artifact_id = report['artifact_id']
        if report['artifact_hash'] != delivery.get('sha256'):
            raise ValueError('Report artifact hash differs from delivery hash')
    content = None
    if artifact_id:
        artifacts = {a['id']: a for a in framework.store.artifacts(session_id=item['session_id'])}
        metadata = artifacts.get(artifact_id)
        if metadata is None or metadata['run_id'] not in item['state']['attempts']:
            raise ValueError('Delivery artifact is outside workflow scope')
        content = framework.store.read_artifact(artifact_id)
        actual = hashlib.sha256(content).hexdigest()
        if actual != metadata['sha256'] or actual != delivery.get('sha256'):
            raise ValueError('Delivery artifact hash integrity mismatch')
        if delivery.get('content') is not None and content != delivery['content'].encode('utf-8'):
            raise ValueError('Delivery content differs from verified artifact bytes')
        result['artifact_verification'] = {'status': 'verified', 'artifact_id': artifact_id,
                                           'sha256': actual, 'bytes': len(content)}
    elif delivery.get('content'):
        raise ValueError('Delivery content has no verifiable artifact')
    else:
        result['artifact_verification'] = {'status': 'not_available'}
    return item, result, content


def export_task(framework, fingerprint, output):
    item, result, content = verified_result(framework, fingerprint)
    delivery = result.get('delivery') or {}
    # Verify everything before publishing any output.
    target = Path(output).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        previous = target/'task.json'
        if not previous.is_file() or json.loads(previous.read_text()).get('fingerprint') != fingerprint:
            raise ValueError('Export directory contains another task or unrelated files')
        if content is None and any((target/name).exists() for name in ('report.md', 'answer.md')):
            raise ValueError('Export directory contains an earlier delivery; choose a fresh directory')
    target.mkdir(parents=True, exist_ok=True)
    payloads = {'result.json': result, 'task.json': item}
    payloads['feedback.json']=result.get('feedback_history',[])
    payloads['claim-coverage.json']=result.get('claim_coverage',{'scientific_support_assessed':False,'claims':[]})
    payloads['selection-audit.json']=result.get('selection_audit',{})
    if result.get('lineage') is not None:
        payloads['lineage.json'] = result['lineage']
    for name, value in payloads.items():
        _atomic(target/name, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())
    if content is not None:
        _atomic(target/('report.md' if result.get('delivery_status') == 'report' else 'answer.md'), content)
    summary = '# Workflow export\n\n' + '\n'.join(
        '- ' + key + ': ' + str(result.get(key, 'not_assessed')) for key in
        ('execution_status', 'delivery_status', 'content_status', 'review_status', 'empirical_validation_status'))
    summary += '\n\nArtifact verification: ' + result['artifact_verification']['status'] + '\n'
    _atomic(target/'README.md', summary.encode())
    return {'fingerprint': fingerprint, 'output': str(target), 'artifact_verification': result['artifact_verification']}
