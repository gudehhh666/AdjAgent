"""Bind explicitly named task files to authorized identities, without directory scans.

Cards are discovery metadata, not evidence or input snapshots. Execution must
revalidate permissions and hash the actual staged bytes through native tools.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .tools.policy import _credential_path


INPUT_GUIDANCE = '''When task input cards are present, use their exact regular-file paths in code_interpreter.inputs, never the containing directory. Read the staged files at each card's snapshot path and compute from those bytes; do not invent or transcribe intermediate counts as if they were file computations. Inspect actual headers before selecting fields. Keep joins, inclusion rules, missing values and counting units explicit. Files with the same basename require separate calls because sandbox names would collide. Write outputs/analysis_result.json with a compact summary object, optionally metrics (object) and limitations (array of strings); save long tables separately. Report failed file reads or calculations as incomplete. A matching input receipt proves staged identity but does not prove that the code read it or computed correctly. Cards and hashes are metadata, not scientific observations or reference answers. No repeated review rounds are required.'''

_MAX_FILE_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_BYTES = 128 * 1024 * 1024
_MAX_CARDS = 8


def _public_strings(value, depth=0):
    if depth > 8:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _public_strings(item, depth + 1)]
    if isinstance(value, dict):
        return [text for key, item in value.items()
                if 'private' not in str(key).casefold()
                and str(key).casefold() not in {'gold', 'answer', 'answers', 'reference',
                                              'references', 'expected', 'rubric'}
                for text in _public_strings(item, depth + 1)]
    return []


def _goal_segments(goal):
    # Decode JSON envelopes before reading their literal Markdown/plain text.
    # Otherwise a quoted input_text consumes its nested backtick file paths.
    # Never consult adjacent manifests, reference files or directory contents.
    decoder, cursor, segments = json.JSONDecoder(), 0, []
    for match in re.finditer(r'[\[{]', goal):
        if match.start() < cursor:
            continue
        try:
            value, end = decoder.raw_decode(goal, match.start())
        except ValueError:
            continue
        segments.append(goal[cursor:match.start()])
        segments.extend(_public_strings(value))
        cursor = end
    segments.append(goal[cursor:])
    return segments


def _literal_paths(goal):
    # Quoted / Markdown paths support spaces. Do not discover a basename by
    # walking roots: only the literal project-relative path is considered.
    marked = re.compile(r'''`([^`\n]+)`|"([^"\n]+)"|'([^'\n]+)'|\]\(([^)\n]+)\)''')
    values = []
    def collect(match):
        values.append(next(value for value in match.groups() if value is not None))
        return ' '
    remainder = marked.sub(collect, goal)
    tokens = re.findall(r'''[^\s`"'<>\[\](){},;，；。]+''', remainder)
    values.extend(token for token in tokens
                  if '/' in token or re.fullmatch(r'[\w.-]+\.[A-Za-z0-9]{1,8}[.]?', token))
    return [value.rstrip('.,;') for value in values if '://' not in value]


def _named_paths(goal):
    return [path for segment in _goal_segments(goal) for path in _literal_paths(segment)]


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _identity(path):
    """Hash stable regular bytes via no-follow descriptors, never FIFOs or links."""
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    parent = os.open(path.anchor, flags | os.O_DIRECTORY)
    fd = None
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, flags | os.O_DIRECTORY, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(path.name, flags, dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError('Task inputs must be regular, non-linked files')
        if before.st_size > _MAX_FILE_BYTES:
            raise ValueError('Task input exceeds the analysis per-file limit')
        digest, size = hashlib.sha256(), 0
        while chunk := os.read(fd, min(65536, _MAX_FILE_BYTES + 1 - size)):
            size += len(chunk)
            if size > _MAX_FILE_BYTES:
                raise ValueError('Task input grew beyond the analysis limit')
            digest.update(chunk)
        after = os.fstat(fd)
        if (_signature(before) != _signature(after)
                or _signature(after) != _signature(os.stat(path, follow_symlinks=False))
                or path.resolve(strict=True) != path):
            raise ValueError('Task input changed during identity read')
        return digest.hexdigest(), size
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent)


def task_input_cards(goal, config, project_root, *, explicit_files=(), store_root=None):
    """Return up to eight authorized, named file identities; skip invalid paths.

    ``explicit_files`` is host-supplied input metadata, not a new permission.
    Pass the runtime Store root when called by a service to exclude its private
    files. Roots and tool permissions remain unchanged. No input contents, schema
    guesses or answer fields are emitted. Missing/denied inputs yield no card;
    callers must not interpret an empty list as proof of data absence.
    """
    if 'code_interpreter' not in config.get('allowed_tools', []):
        return []
    project = Path(project_root).expanduser().absolute()
    roots = [Path(root).expanduser() for root in config.get('allowed_roots', [])]
    roots = [(root if root.is_absolute() else project / root).resolve() for root in roots]
    private = Path(store_root).resolve() if store_root is not None else None
    cards, seen, total = [], set(), 0
    for value in [*explicit_files, *_named_paths(goal)]:
        try:
            candidate = Path(value).expanduser()
            candidate = candidate if candidate.is_absolute() else project / candidate
            if '..' in candidate.parts or _credential_path(candidate):
                continue
            if (not any(candidate.is_relative_to(root) for root in roots)
                    or private is not None and candidate.is_relative_to(private)
                    or candidate in seen):
                continue
            digest, size = _identity(candidate)
        except (OSError, TypeError, ValueError):
            continue
        if total + size > _MAX_TOTAL_BYTES:
            continue
        seen.add(candidate)
        total += size
        cards.append({'path': str(candidate), 'snapshot': 'inputs/' + candidate.name,
                      'sha256': digest, 'bytes': size, 'tools': ['code_interpreter']})
        if len(cards) == _MAX_CARDS:
            break
    counts = Counter(card['snapshot'] for card in cards)
    for card in cards:
        if counts[card['snapshot']] > 1:
            card['separate_call_required'] = True
    return cards
