"""Process-wide NCBI E-utilities pacing shared by every client that calls eutils or the ID converter."""
from threading import Lock
import time

# NCBI allows 3 requests/s without an API key and 10/s with one; stay just under both.
INTERVAL_SECONDS = {False: 0.36, True: 0.11}

_lock = Lock()
_last = 0.0


def wait(api_key=None):
    global _last
    with _lock:
        delay = _last + INTERVAL_SECONDS[bool(api_key)] - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _last = time.monotonic()
