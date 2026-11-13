"""Thread-safe minimum-interval rate limiter, keyed by API name.

Specialists run retrieval in parallel threads, so per-call sleeps aren't enough
to respect NCBI/E-utilities limits (3 req/s without key). A shared limiter
serializes spacing across all threads for a given key."""
from __future__ import annotations

import threading
import time

_locks: dict[str, threading.Lock] = {}
_last: dict[str, float] = {}
_registry_lock = threading.Lock()


def _lock_for(key: str) -> threading.Lock:
    with _registry_lock:
        if key not in _locks:
            _locks[key] = threading.Lock()
            _last[key] = 0.0
        return _locks[key]


def throttle(key: str, min_interval: float) -> None:
    """Block until at least `min_interval` seconds have passed since the last
    throttle() call for this key (across all threads)."""
    lock = _lock_for(key)
    with lock:
        now = time.monotonic()
        wait = _last[key] + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _last[key] = time.monotonic()
