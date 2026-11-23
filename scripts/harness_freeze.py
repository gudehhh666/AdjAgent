"""Pin and check the harness fingerprint recorded in harness_freeze.json.

The code fingerprint covers every .py file under adj_partner/ and environment/; host_files pins the non-Python
files the harness reads (skills, the resource index, provider configuration). A public checkout may lack the
host files listed as held_back.

    .venv/bin/python scripts/harness_freeze.py snapshot   # record current code and host-file hashes
    .venv/bin/python scripts/harness_freeze.py verify     # list files changed since the snapshot; exit 1 on drift
    .venv/bin/python scripts/harness_freeze.py slots      # print the task slots
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / 'harness_freeze.json'
CODE_ROOTS = ('adj_partner', 'environment')
HOST_FILES = ('config/skills', 'config/research-resource-index.json', 'config/llm.yaml', 'config/settings.yaml',
              'config/sources.yaml')


def fingerprint():
    files = {}
    for top in CODE_ROOTS:
        for path in sorted((ROOT / top).rglob('*.py')):
            if '__pycache__' not in path.parts:
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return digest, files


def host_files():
    files = {}
    for name in HOST_FILES:
        path = ROOT / name
        paths = sorted(path.rglob('*')) if path.is_dir() else [path]
        for item in paths:
            if item.is_file() and not item.name.startswith('.'):
                files[str(item.relative_to(ROOT))] = hashlib.sha256(item.read_bytes()).hexdigest()
    return files


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main(command):
    freeze = json.loads(FREEZE.read_text())
    digest, files = fingerprint()
    if command == 'snapshot':
        dirty = [line[3:] for line in git('status', '--porcelain', '--', *CODE_ROOTS).splitlines()]
        freeze['code'] = {'git_head': git('rev-parse', '--short', 'HEAD'), 'uncommitted': dirty,
                          'tree_sha256': digest, 'file_count': len(files), 'files': files}
        freeze['host_files'] = {'roots': list(HOST_FILES), 'files': host_files()}
        FREEZE.write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + '\n')
        print(f'snapshot {digest[:12]} over {len(files)} files; {len(dirty)} uncommitted; '
              f"{len(freeze['host_files']['files'])} host files")
    elif command == 'verify':
        pinned = freeze.get('code', {}).get('files', {})
        changed = sorted(p for p in files.keys() & pinned.keys() if files[p] != pinned[p])
        added, removed = sorted(files.keys() - pinned.keys()), sorted(pinned.keys() - files.keys())
        host, current = freeze.get('host_files', {}), host_files()
        held_back = set(host.get('held_back', []))
        pinned_host = host.get('files', {})
        changed += sorted(p for p in current.keys() & pinned_host.keys() if current[p] != pinned_host[p])
        added += sorted(current.keys() - pinned_host.keys())
        removed += sorted(pinned_host.keys() - current.keys() - held_back)
        for label, paths in (('changed', changed), ('added', added), ('removed', removed)):
            for path in paths:
                print(f'{label}: {path}')
        ok = digest == freeze.get('code', {}).get('tree_sha256') and not (changed or added or removed)
        print('ok' if ok else 'DRIFT')
        return 0 if ok else 1
    elif command == 'slots':
        for slot in freeze['slots']:
            candidate = slot['candidate']['status'] if slot.get('candidate') else '-'
            incumbent = slot['incumbent'] or {'form': '-', 'result': 'no incumbent yet'}
            pipeline = incumbent.get('pipeline') or 'unregistered'
            note = incumbent.get('result') or incumbent.get('note') or slot.get('note')
            print(f"{slot['task']:<28}{incumbent['form'][:40]:<42}pipeline: {pipeline}")
            if note:
                print(f"{'':<28}{note}")
            print(f"{'':<28}candidate: {candidate}")
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ''))
