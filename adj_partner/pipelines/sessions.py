"""Session plumbing shared by recipe pipelines."""
from datetime import datetime, timezone
import json

from ..tools.catalog import canonical_json


def enable_host_tools(service, session_id, names):
    """Add host-registered tools to a session's enabled set (same write path as profiles.apply_profile)."""
    store = service.store
    with store.transaction(write=True) as conn:
        row = conn.execute('SELECT * FROM sessions WHERE id=?', (session_id,)).fetchone()
        cfg = json.loads(row['config'])
        cfg['enabled_tools'] = sorted(set(cfg['enabled_tools']) | (set(names) & set(cfg['allowed_tools'])))
        conn.execute('UPDATE sessions SET config=?,revision=revision+1 WHERE id=?',
                     (canonical_json(store.redact(cfg)), session_id))
        conn.execute('INSERT INTO events(session_id,run_id,kind,payload,created_at) VALUES(?,NULL,?,?,?)',
                     (session_id, 'profile.changed',
                      canonical_json({'host_tools_enabled': sorted(names), 'revision': row['revision'] + 1}),
                      datetime.now(timezone.utc).isoformat()))


def open_session(service, *, profile, instructions, allowed_roots, allow_network, budget, allowed_tools=None,
                 host_tools=()):
    """Create the session; ``host_tools`` must already be registered on ``service.catalog``."""
    options = dict(profile=profile, instructions=instructions, allowed_roots=list(allowed_roots),
                   allow_network=allow_network, budget=dict(budget))
    if allowed_tools is not None:
        options['allowed_tools'] = list(allowed_tools) + list(host_tools)
    session = service.create_session(**options)
    if host_tools:
        enable_host_tools(service, session['id'], host_tools)
        session = service.store.get_session(session['id'])
    return session


def delivered(run):
    return run.get('state', {}).get('delivery', {}).get('content', '') or ''


def drive(service, session, goal):
    """Submit ``goal`` and drive the run; failures keep whatever the run recorded."""
    out = {'session_id': session['id'], 'run_id': None, 'status': 'failed', 'content': ''}
    try:
        out['run_id'] = service.submit(session['id'], goal)['id']
        run = service.drive(out['run_id'])
        out.update(status=run['status'], content=delivered(run), stop_reason=run.get('state', {}).get('stop_reason'),
                   runtime=run)
    except Exception as exc:
        out.update(error_type=type(exc).__name__, error_code=str(getattr(exc, 'code', 'unknown')), error=str(exc)[:500])
    if out['run_id']:
        try:
            out['actions'] = service.actions(out['run_id'])
        except Exception as exc:
            out['export_error'] = type(exc).__name__
    return out
