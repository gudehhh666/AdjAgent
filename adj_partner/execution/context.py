"""Cooperative job helpers for trusted host handlers, not an OS sandbox."""

from pathlib import Path
import time

from ..tools.catalog import canonical_json
from ..tools.policy import ToolCancelled, ToolContext, _credential_path


class JobContext(ToolContext):
    def __init__(self, manager, job):
        super().__init__(manager.store, job["session_id"], job["run_id"],
                         manager.store.get_session(job["session_id"])["revision"],
                         call_id=job["idempotency_key"])
        self.job_id, self.workspace = job["id"], Path(job["workspace"])
        self._manager, self._job = manager, job

    def check_cancelled(self):
        super().check_cancelled()
        job = self._manager.get(self.job_id, self.session_id)
        if job["status"] != "running" or time.time() >= job["deadline"]:
            raise ToolCancelled("Job cancellation/deadline or terminal state observed")

    def check_path(self, path, write=False, must_exist=True):
        self.check_cancelled()
        candidate = Path(path).expanduser()
        if ".." in candidate.parts or _credential_path(candidate):
            raise PermissionError("Traversal and credential paths are not accessible")
        resolved = candidate.resolve(strict=False)
        if _credential_path(resolved):
            raise PermissionError("Credential paths are not accessible")
        if resolved.is_relative_to(self.workspace):
            if must_exist and not resolved.exists():
                raise FileNotFoundError(resolved)
            return resolved
        if resolved.is_relative_to(self.store.root):
            raise PermissionError("Private Store files and other job workspaces are inaccessible")
        if write:
            raise PermissionError("Job writes are restricted to its private workspace")
        return super().check_path(path, write=False, must_exist=must_exist)

    def progress(self, message, **details):
        from .manager import bounded_text
        self.check_cancelled()
        # Bound before encoding: a huge diagnostic payload never becomes an
        # unbounded durable event. Keep useful simple JSON details if they fit.
        try:
            cleaned = self.store.redact(details)
            if len(canonical_json(cleaned).encode()) > 6000:
                cleaned = {"truncated": True}
        except Exception:
            cleaned = {"invalid_details": True}
        payload = {"message": bounded_text(self.store, message, 2000), "details": cleaned}
        with self.store.transaction(write=True) as conn:
            changed = conn.execute("""UPDATE execution_jobs SET progress_count=progress_count+1
                WHERE id=? AND status='running' AND progress_count<100""", (self.job_id,)).rowcount
            if changed:
                payload = self._manager._event(conn, self._job, "job.progress", payload)
        if changed:
            self._manager._mirror(self._job, "job.progress", payload)
