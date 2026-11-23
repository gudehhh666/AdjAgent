"""Health checks for the local installation and a failure census over persisted runs.

``adj doctor`` checks what a run needs (Python, packages, model configuration names, resource pack,
DEG cache, literature library, disk, frozen evaluation protocols). ``adj doctor runs`` classifies
every persisted run by outcome so recurring failure modes are visible. Credential values are never
read into output; only whether a name is set.
"""

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

from .application_config import HOST_ENV_NAME, PROJECT_ROOT, host_env_file

RUN_ROOTS = ('workspace', '.runtime', 'outputs')
MODEL_NAMES = ('ADJ_LLM_API_KEY', 'ADJ_LLM_BASE_URL', 'ADJ_LLM_MODEL')
OPTIONAL_NAMES = ('ADJ_NCBI_API_KEY', 'ADJ_NCBI_EMAIL', 'ADJ_UNPAYWALL_EMAIL', 'ADJ_WEB_SEARCH_PROVIDER',
                  'ADJ_WEB_SEARCH_API_KEY')
PACKAGES = ('httpx', 'jsonschema', 'lxml', 'pandas', 'pyarrow', 'textual', 'pypdf', 'openpyxl')
STALE_AFTER = timedelta(hours=6)


def _check(name, ok, detail, *, level='error'):
    return {'check': name, 'status': 'ok' if ok else level, 'detail': detail}


def _key_file(root, env_file):
    selected = Path(env_file) if env_file else host_env_file(root)
    if selected is None:
        return _check('key_file', False, f'missing {root / HOST_ENV_NAME}; create it (mode 600) or set ADJ_ENV_FILE', level='warn')
    mode = selected.stat().st_mode & 0o777
    problems = [f'mode {oct(mode)}, run chmod 600 {selected.name}'] if mode & 0o077 else []
    legacy = root / '.env'
    if legacy.is_file() and not legacy.is_symlink() and selected.name == HOST_ENV_NAME and not os.path.samefile(legacy, selected):
        problems.append(f'.env is a separate copy; run: ln -f {HOST_ENV_NAME} .env')
    return _check('key_file', not problems, '; '.join(problems) or f'{selected.name} (mode 600, .env linked)', level='warn')


def environment(root=PROJECT_ROOT, env_file=None):
    root = Path(root)
    results = [_check('python', sys.version_info >= (3, 11), sys.version.split()[0])]
    missing = []
    for package in PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            missing.append(package)
    results.append(_check('packages', not missing, 'all importable' if not missing else f'missing: {", ".join(missing)}'))

    import os
    from .native.scholarly import _explicit_values
    results.append(_key_file(root, env_file))
    try:
        values = _explicit_values(root, env_file or host_env_file(root) or (root / '.env' if (root / '.env').exists() else None))
    except Exception as exc:
        values = {}
        results.append(_check('env_file', False, f'unreadable: {type(exc).__name__}'))
    present = lambda name: bool(os.environ.get(name) or values.get(name))
    absent = [n for n in MODEL_NAMES if not present(n)]
    results.append(_check('model_configuration', not absent,
                          'set: ' + ', '.join(MODEL_NAMES) if not absent else f'not set: {", ".join(absent)}'))
    optional = [n for n in OPTIONAL_NAMES if not present(n)]
    results.append(_check('optional_integrations', not optional,
                          'all set' if not optional else f'not set: {", ".join(optional)}', level='warn'))

    pack = root / 'data' / 'adjuvant_resources_v4'
    try:
        manifest = json.loads((pack / 'manifest.json').read_text())
        results.append(_check('resource_pack', (pack / 'records.jsonl').exists(),
                              f"v4: {len(manifest.get('datasets', []))} datasets, {sum(manifest.get('counts', {}).values())} records"))
    except (OSError, ValueError):
        results.append(_check('resource_pack', False, f'{pack} missing or unreadable'))

    deg = json.loads((root / 'adj_partner' / 'native' / 'deg-cache-manifest.json').read_text())
    deg_root = root / 'data' / 'gene_diff'
    wrong = [item['name'] for item in deg
             if not any(p.stat().st_size == item['size'] for p in deg_root.rglob(item['name']))] if deg_root.exists() else ['<missing>']
    results.append(_check('deg_cache', not wrong, f'{len(deg)} files match manifest sizes' if not wrong
                          else f'{len(wrong)} mismatched/missing, e.g. {wrong[:3]}', level='warn'))

    from .library import shared
    try:
        library = shared()
        if library is None:
            results.append(_check('literature_library', True, 'disabled by ADJ_LITERATURE_LIBRARY', level='warn'))
        else:
            stats = library.stats()
            results.append(_check('literature_library', stats['papers'] > 0,
                                  f"{stats['papers']} papers, {stats['passages']} passages, {stats['failures']} ingest failures",
                                  level='warn'))
    except Exception as exc:
        results.append(_check('literature_library', False, f'{type(exc).__name__}: {exc}', level='warn'))

    free = shutil.disk_usage(root).free / 2 ** 30
    results.append(_check('disk', free > 20, f'{free:.0f} GiB free', level='warn'))
    results.append(frozen_protocols(root))
    return results


def frozen_protocols(root=PROJECT_ROOT):
    """Report frozen evaluation protocols whose pinned files changed without a recorded snapshot."""
    root = Path(root)
    drifted = {}
    for protocol in sorted([*root.glob('outputs/*/protocol.json'), *root.glob('outputs/*/freeze.json')]):
        try:
            data = json.loads(protocol.read_text())
        except ValueError:
            continue
        files = data.get('files') or data.get('hashes') or {}
        if not isinstance(files, dict):
            continue
        snapshot = protocol.parent / 'frozen_snapshot'
        changed = []
        for name, pinned in files.items():
            pinned = pinned if isinstance(pinned, str) else (pinned or {}).get('sha256')
            path = root / name
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != pinned:
                kept = snapshot / name
                if not (kept.exists() and hashlib.sha256(kept.read_bytes()).hexdigest() == pinned):
                    changed.append(name)
        if changed:
            drifted[protocol.parent.name] = changed
    detail = 'all pinned files intact or snapshotted' if not drifted else '; '.join(
        f'{k}: {len(v)} changed' for k, v in drifted.items())
    return _check('frozen_protocols', not drifted, detail, level='warn')


def classify(status, state, created_at=None, *, now=None):
    state = state or {}
    error = state.get('error') or ''
    if isinstance(error, dict):
        error = ' '.join(str(error.get(k) or '') for k in ('code', 'message', 'type')).strip()
    error = str(error)
    stop = str(state.get('stop_reason') or '')
    if status == 'succeeded':
        return 'succeeded'
    if status in ('created', 'running'):
        try:
            age = (now or datetime.now(timezone.utc)) - datetime.fromisoformat(created_at)
        except (TypeError, ValueError):
            age = None
        stale = age is not None and age > STALE_AFTER
        return ('never_started' if status == 'created' else 'orphaned_running') if stale else f'{status}_recent'
    if status == 'cancelled':
        return 'cancelled'
    if status == 'budget_exhausted':
        return 'context_budget' if 'context' in error else 'budget_exhausted'
    lowered = error.lower()
    if '429' in lowered or 'rate limit' in lowered:
        return 'provider_rate_limited'
    if 'http status' in lowered or 'http_' in lowered:
        return 'provider_http_error'
    if 'connection failed' in lowered or 'connecterror' in lowered or 'connecttimeout' in lowered:
        return 'provider_connection'
    if 'timed out' in lowered or 'readtimeout' in lowered:
        return 'provider_timeout'
    if 'malformed' in lowered:
        return 'malformed_reply'
    if stop == 'model_length':
        return 'output_truncated'
    if stop:
        return f'stopped:{stop}'
    return f'error:{error[:60]}' if error else 'unexplained'


def run_databases(root=PROJECT_ROOT, roots=RUN_ROOTS):
    root = Path(root)
    return sorted(p for base in roots if (root / base).exists() for p in (root / base).rglob('db.sqlite3')
                  if 'literature_library' not in p.parts)


def runs(root=PROJECT_ROOT, *, since=None, roots=RUN_ROOTS):
    totals, by_place, examples = Counter(), {}, {}
    now = datetime.now(timezone.utc)
    for database in run_databases(root, roots):
        try:
            conn = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
            rows = conn.execute('SELECT id, status, state, created_at FROM runs').fetchall()
            conn.close()
        except sqlite3.Error:
            continue
        place = '/'.join(database.relative_to(root).parts[:2])
        for run_id, status, state, created_at in rows:
            if since and (created_at or '') < since:
                continue
            try:
                parsed = json.loads(state or '{}')
            except ValueError:
                parsed = {}
            kind = classify(status, parsed, created_at, now=now)
            totals[kind] += 1
            by_place.setdefault(kind, Counter())[place] += 1
            examples.setdefault(kind, f'{database.relative_to(root)}#{run_id}')
    finished = sum(v for k, v in totals.items() if not k.endswith('_recent') and k not in ('never_started', 'orphaned_running'))
    return {'since': since, 'totals': dict(totals.most_common()),
            'success_rate_of_finished': round(totals['succeeded'] / finished, 4) if finished else None,
            'top_places': {k: dict(v.most_common(3)) for k, v in by_place.items() if k != 'succeeded'},
            'examples': {k: v for k, v in examples.items() if k != 'succeeded'}}


SEALED_ROOTS = ('outputs',)


def reconcile(state_dir, *, idle_hours=6.0, apply=False, root=PROJECT_ROOT):
    """Close runs left 'running' by dead drivers in one state directory; dry run unless apply.

    Sealed evaluation evidence under outputs/ is refused: those databases are hashed by reports.
    """
    from .runtime.journal import RunJournal
    from .storage import Store
    state_dir = Path(state_dir).resolve()
    try:
        relative = state_dir.relative_to(Path(root).resolve())
    except ValueError:
        relative = None
    if relative is not None and relative.parts[:1] and relative.parts[0] in SEALED_ROOTS:
        raise ValueError(f'{relative} is sealed evaluation evidence; reconcile a copy instead')
    if not (state_dir / 'db.sqlite3').exists():
        raise FileNotFoundError(f'no db.sqlite3 in {state_dir}')
    conn = sqlite3.connect(f'file:{state_dir / "db.sqlite3"}?mode=ro', uri=True)
    candidates = [row[0] for row in conn.execute("SELECT id FROM runs WHERE status='running'")]
    conn.close()
    store = Store(state_dir)
    journal = RunJournal(store)
    closed, kept = [], []
    for run_id in candidates:
        if apply:
            run = journal.abandon(run_id, idle_seconds=idle_hours * 3600)
            (closed if run else kept).append(run_id)
        else:
            closed.append(run_id)
    return {'state_dir': str(state_dir), 'applied': apply, 'idle_hours': idle_hours,
            'abandoned' if apply else 'running_candidates': closed, 'still_active': kept}


def _print_checks(results):
    marks = {'ok': 'OK  ', 'warn': 'WARN', 'error': 'FAIL'}
    for item in results:
        print(f"[{marks[item['status']]}] {item['check']}: {item['detail']}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog='adj doctor', description=__doc__.splitlines()[0])
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--json', action='store_true')
    sub = parser.add_subparsers(dest='command')
    census = sub.add_parser('runs', help='classify persisted runs by outcome')
    census.add_argument('--since', help='ISO date, e.g. 2026-10-15')
    census.add_argument('--root', action='append', help='run roots under the project (default: workspace .runtime outputs)')
    census.add_argument('--reconcile', type=Path, metavar='STATE_DIR',
                        help='close runs left running by dead drivers in this state directory (dry run without --apply)')
    census.add_argument('--apply', action='store_true')
    census.add_argument('--idle-hours', type=float, default=6.0)
    args = parser.parse_args(argv)
    if args.command == 'runs' and args.reconcile:
        result = reconcile(args.reconcile, idle_hours=args.idle_hours, apply=args.apply)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == 'runs':
        result = runs(since=args.since, roots=tuple(args.root) if args.root else RUN_ROOTS)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    results = environment(env_file=args.env_file)
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        _print_checks(results)
    return 1 if any(r['status'] == 'error' for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
