"""Detached supervisor and its directly owned trusted-handler child process."""

import importlib
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import tempfile
import time

from ..contracts import ToolResult
from ..storage import Store
from ..tools.catalog import canonical_json
from ..tools.policy import ToolCancelled
from .context import JobContext
from .manager import (ACTIVE, MAX_LOG_BYTES, MAX_RESULT_BYTES, JobManager, bounded_text,
                      failure, validate_result, worker_command, worker_environment)


_ASSIGNMENT_KEY = re.compile(
    r'''(?<![A-Za-z0-9_-])(?P<key>[A-Za-z_][A-Za-z0-9_-]*)(?:\\*["'])?\s*[=:]''',
)
_JSON_RECORD_START = re.compile(r"(?m)^[ \t]*(?P<start>\{|\[)")


def _scrub_truncated_capture(store, text):
    """Fail closed on parser loss, using only the already bounded capture.

    Once a record is incomplete, neither quote termination nor JSON escape depth
    is trustworthy. Keep preceding ordinary records, but discard an incomplete
    JSON-looking record and everything after a recognized sensitive assignment.
    """
    end, decoded_end = len(text), 0
    decoder = json.JSONDecoder()
    for match in _JSON_RECORD_START.finditer(text):
        start = match.start("start")
        if start < decoded_end:
            continue
        try:
            _, decoded_end = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            end = match.start()
            break
    for match in _ASSIGNMENT_KEY.finditer(text, 0, end):
        # Ask the existing Store policy, including token/cookie/camelCase keys,
        # instead of maintaining a second list of credential field names.
        probe = {match.group("key"): ""}
        if store.redact(probe) != probe:
            end = match.start()
            break
    return text[:end] + ("[REDACTED]" if end < len(text) else "")


def _outcome_path(job):
    return Path(job["workspace"]).parent / "control" / "outcome.json"


def invoke(root, job_id, owner):
    manager = JobManager(Store(root))
    job = manager._read(job_id)
    if job["owner_id"] != owner or job["status"] != "running":
        return
    try:
        context = JobContext(manager, job)
        context.check_cancelled()
        module, name = job["handler_ref"].split(":")
        handler = getattr(importlib.import_module(module), name)
        result = handler(context, **job["arguments"])
    except ToolCancelled as exc:
        result = failure("cancelled", bounded_text(manager.store, exc, 2000), "cancelled")
    except PermissionError as exc:
        result = failure("permission_denied", bounded_text(manager.store, exc, 2000), "denied")
    except BaseException as exc:
        result = failure("handler_error", bounded_text(manager.store, f"{type(exc).__name__}: {exc}", 2000))
    result = validate_result(manager.store, job, result)
    content = canonical_json(result.to_dict()).encode()
    target = _outcome_path(job)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".outcome-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


class _Capture:
    def __init__(self, process):
        self.selector = selectors.DefaultSelector()
        self.buffers = {"stdout": bytearray(), "stderr": bytearray()}
        self.truncated = {"stdout": False, "stderr": False}
        for name in self.buffers:
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, name)

    def pump(self, timeout=0.05):
        for key, _ in self.selector.select(timeout):
            try:
                chunk = os.read(key.fileobj.fileno(), 65536)
            except BlockingIOError:
                continue
            if not chunk:
                self.selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            buffer = self.buffers[key.data]
            available = MAX_LOG_BYTES - len(buffer)
            buffer.extend(chunk[:available])
            self.truncated[key.data] |= len(chunk) > available

    def finish(self, store):
        # A trusted handler could leave a descendant holding a pipe open. Do not
        # wait forever or pretend this local supervisor owns external daemons.
        end = time.monotonic() + 0.3
        while self.selector.get_map() and time.monotonic() < end:
            self.pump(0.02)
        for key in list(self.selector.get_map().values()):
            self.selector.unregister(key.fileobj)
            key.fileobj.close()
            self.truncated[key.data] = True
        self.selector.close()
        result = {}
        for name, content in self.buffers.items():
            text = content.decode("utf-8", errors="replace")
            if self.truncated[name]:
                text = _scrub_truncated_capture(store, text)
            result[name] = bounded_text(store, text)
        result.update({name + "_truncated": value for name, value in self.truncated.items()})
        return result


def _stop_owned_child(child, capture):
    """Only this process's unreaped Popen child may be signalled, never DB PIDs."""
    if child.poll() is not None:
        return True
    try:
        os.killpg(child.pid, signal.SIGTERM)
        end = time.monotonic() + 0.25
        while child.poll() is None and time.monotonic() < end:
            capture.pump(0.02)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=1)
        return True
    except ProcessLookupError:
        try:
            child.wait(timeout=1)
            return True
        except subprocess.TimeoutExpired:
            return False
    except (OSError, subprocess.TimeoutExpired):
        return False


def supervise(root, job_id):
    manager = JobManager(Store(root))
    job = manager._claim(job_id)
    if job is None:
        return
    child, capture = None, None
    try:
        run_status = manager.store.get_run(job["run_id"])["status"]
        if job["status"] == "cancel_requested" or run_status == "cancelled":
            manager._finish(job, "cancelled", failure("cancelled", "Cancellation acknowledged before child launch", "cancelled"))
            return
        if time.time() >= job["deadline"]:
            manager._finish(job, "timed_out", failure("timed_out", "Job deadline elapsed before child launch"))
            return
        if run_status in {"succeeded", "failed", "budget_exhausted"}:
            manager._finish(job, "failed", failure("run_terminal", "Run became terminal before child launch"))
            return
        workspace = manager._prepare_workspace(job)
        child = subprocess.Popen(worker_command("invoke", root, job_id, job["owner_id"]),
            cwd=workspace, env=worker_environment(workspace), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True)
        capture = _Capture(child)
        last_heartbeat = time.monotonic()
        reason = None
        while child.poll() is None:
            capture.pump()
            current = manager._read(job_id)
            run_status = manager.store.get_run(job["run_id"])["status"]
            if current["status"] not in ACTIVE:
                reason = "uncertain"
            elif current["status"] == "cancel_requested" or run_status == "cancelled":
                reason = "cancelled"
            elif time.time() >= job["deadline"]:
                reason = "timed_out"
            elif run_status in {"succeeded", "failed", "budget_exhausted"}:
                reason = "run_terminal"
            if reason:
                if not _stop_owned_child(child, capture):
                    reason = "uncertain"
                break
            if time.monotonic() - last_heartbeat >= 0.25:
                if not manager._heartbeat(job):
                    reason = "uncertain"
                    _stop_owned_child(child, capture)
                    break
                last_heartbeat = time.monotonic()
        logs = capture.finish(manager.store)
        if reason:
            status = "failed" if reason == "run_terminal" else reason
            result_status = {"cancelled": "cancelled", "uncertain": "uncertain"}.get(status, "error")
            result = failure(reason, "Owned child stopped" if reason != "uncertain" else "Child outcome cannot be confirmed", result_status)
        elif child.returncode != 0:
            status, result = "uncertain", failure("worker_lost", "Handler process exited without a confirmed outcome; do not replay", "uncertain")
        else:
            try:
                with _outcome_path(job).open("rb") as stream:
                    content = stream.read(MAX_RESULT_BYTES + 1)
                if len(content) > MAX_RESULT_BYTES:
                    raise ValueError("oversized result")
                result = validate_result(manager.store, job, ToolResult(**json.loads(content)))
                if result.error is not None and result.error.get("code") == "analysis_timeout":
                    status = "timed_out"
                else:
                    status = {"ok": "succeeded", "cancelled": "cancelled", "uncertain": "uncertain"}.get(
                        result.status, "failed")
            except Exception:
                status, result = "uncertain", failure("worker_lost", "Child did not publish a valid final outcome", "uncertain")
        manager._finish(job, status, result, logs=logs)
    except BaseException as exc:
        stopped = child is None or (capture is not None and _stop_owned_child(child, capture))
        logs = capture.finish(manager.store) if capture is not None and capture.selector.get_map() else None
        manager._finish(job, "uncertain", failure("supervisor_error",
            bounded_text(manager.store, f"{type(exc).__name__}: {exc}; child_stop_confirmed={stopped}", 2000), "uncertain"), logs=logs)


def main():
    mode, root, job_id, *extra = sys.argv[1:]
    if mode == "supervise" and not extra:
        supervise(root, job_id)
    elif mode == "invoke" and len(extra) == 1:
        invoke(root, job_id, extra[0])
    else:
        raise ValueError("Invalid local worker invocation")


if __name__ == "__main__":
    main()
