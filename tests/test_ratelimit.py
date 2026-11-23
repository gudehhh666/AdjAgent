import threading
import time

from core.ratelimit import throttle


def test_throttle_spaces_calls():
    key = "test_key_serial"
    throttle(key, 0.0)  # prime
    t0 = time.monotonic()
    for _ in range(3):
        throttle(key, 0.05)
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.10  # at least 2 gaps of 0.05


def test_throttle_thread_safe_serialization():
    key = "test_key_threads"
    throttle(key, 0.0)
    n = 5
    interval = 0.02

    def worker():
        throttle(key, interval)

    t0 = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - t0
    assert elapsed >= interval * (n - 1)
