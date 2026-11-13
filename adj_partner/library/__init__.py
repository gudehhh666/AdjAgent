"""Persistent, project-wide literature library."""

import os
from pathlib import Path

from .store import LiteratureLibrary, normalize_ids

ENV = 'ADJ_LITERATURE_LIBRARY'
_SHARED = {}


def default_root():
    from ..application_config import PROJECT_ROOT
    return PROJECT_ROOT / 'data' / 'literature_library'


def shared():
    """The library live runs write to: ``$ADJ_LITERATURE_LIBRARY`` (``off`` disables), else the project default."""
    setting = os.environ.get(ENV, '').strip()
    if setting.lower() in ('off', '0', 'false', 'none'):
        return None
    root = Path(setting) if setting else default_root()
    key = str(root.resolve())
    if key not in _SHARED:
        _SHARED[key] = LiteratureLibrary(root)
    return _SHARED[key]


__all__ = ['LiteratureLibrary', 'normalize_ids', 'shared', 'default_root', 'ENV']
