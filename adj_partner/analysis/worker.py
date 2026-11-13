"""Trusted handler for one sandboxed analysis job."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import signal
import stat
import sys

from ..contracts import ToolResult
from ..native.safe_files import read_input
from ..tools import ToolFailure
from ..tools.catalog import canonical_json
from ..tools.policy import ToolCancelled
from .history import AnalysisHistory, finalize_analysis_job
from .sandbox import SandboxUnavailable, select_sandbox


MAX_SOURCE_BYTES = 128 * 1024
MAX_INPUT_FILES = 8
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_FILES = 32
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_TOTAL_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_ENTRIES = 128
MAX_SUMMARY_BYTES = 64 * 1024

_NUMPY_JSON_SERIALIZATION_ERROR = re.compile(
    r"Object of type (?:numpy\.)?(?:u?int(?:8|16|32|64|128|_)|float(?:16|32|64|128|_)|ndarray) "
    r"is not JSON serializable"
)
_NUMPY_JSON_REPAIR_HINT = (
    "Convert NumPy integer/float scalars with .item(), int(...), or float(...) before JSON output; "
    "convert arrays with .tolist(). Preserve the computed values."
)


MEDIA_TYPES = {
    ".csv": "text/csv", ".json": "application/json", ".md": "text/markdown",
    ".py": "text/x-python", ".sh": "text/x-shellscript", ".tsv": "text/tab-separated-values",
    ".txt": "text/plain", ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def _failure(code, message, status="error", *, analysis_id=None, details=None,
             sandbox_backend=None, artifacts=None):
    data = {"analysis_id": analysis_id} if analysis_id else {}
    if sandbox_backend:
        data["sandbox_backend"] = sandbox_backend
    data = data or None
    error = {"code": code, "message": message}
    if details:
        error["details"] = details
    return ToolResult(status=status, data=data, error=error, artifacts=artifacts or [])


def _output_descriptors(artifacts, roles):
    by_id = {artifact_id: role for role, artifact_id in roles}
    return [{"ordinal": ordinal, "role": by_id[item["id"]], "artifact_id": item["id"],
             "name": item["name"], "media_type": item["media_type"], "size": item["size"],
             "sha256": item["sha256"]}
            for ordinal, item in enumerate(artifacts, 1)]


def analysis_job_finalize(conn, store, job, status, result):
    finalize_analysis_job(conn, store, job, status, result)


def _validate_manifest(item):
    if (not isinstance(item, dict) or set(item) != {"path", "name", "size", "sha256"}
            or not isinstance(item["path"], str) or not isinstance(item["name"], str)
            or Path(item["name"]).name != item["name"] or not item["name"]
            or not re.fullmatch(r"[a-f0-9]{64}", item["sha256"])
            or type(item["size"]) is not int or not 0 <= item["size"] <= MAX_INPUT_BYTES):
        raise ToolFailure("analysis_input_denied", "Analysis input manifest is malformed", "denied")


def snapshot_inputs(context, manifests):
    if not isinstance(manifests, list) or len(manifests) > MAX_INPUT_FILES:
        raise ToolFailure("analysis_input_denied", f"Analysis accepts at most {MAX_INPUT_FILES} input files", "denied")
    destination = Path(context.workspace) / "inputs"
    if destination.is_symlink() or not destination.is_dir():
        raise ToolFailure("analysis_input_denied", "Analysis input workspace is missing or unsafe", "denied")
    copied, total, names = [], 0, set()
    for number, expected in enumerate(manifests, 1):
        _validate_manifest(expected)
        if expected["name"] in names:
            raise ToolFailure("analysis_input_denied", "Analysis input names must be unique", "denied")
        names.add(expected["name"])
        try:
            item = read_input(context, expected["path"], maximum_bytes=MAX_INPUT_BYTES)
        except PermissionError as exc:
            raise ToolFailure("analysis_input_denied", str(exc), "denied") from exc
        except (OSError, ValueError) as exc:
            raise ToolFailure("analysis_input_changed", str(exc)) from exc
        if item.size != expected["size"] or item.sha256 != expected["sha256"]:
            raise ToolFailure("analysis_input_changed", "Analysis input changed after submission")
        total += item.size
        if total > MAX_TOTAL_INPUT_BYTES:
            raise ToolFailure("analysis_input_denied", "Analysis inputs exceed the total byte limit", "denied")
        target = destination / expected["name"]
        if target.exists() or target.is_symlink():
            raise ToolFailure("analysis_input_denied", "Analysis input snapshot target already exists", "denied")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o400)
        try:
            view = memoryview(item.content)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        copied.append({**expected, "snapshot": f"inputs/{expected['name']}"})
        context.progress("Snapshotted authorized analysis input", completed=number, total=len(manifests))
    return copied


def _media_type(path):
    return MEDIA_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _read_regular_output(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ToolFailure("analysis_output_invalid", "Analysis outputs must be regular non-linked files")
        if before.st_size > MAX_OUTPUT_BYTES:
            raise ToolFailure("analysis_output_limit", "An analysis output exceeds the per-file byte limit")
        chunks, length = [], 0
        while True:
            chunk = os.read(descriptor, min(65536, MAX_OUTPUT_BYTES + 1 - length))
            if not chunk:
                break
            chunks.append(chunk)
            length += len(chunk)
            if length > MAX_OUTPUT_BYTES:
                raise ToolFailure("analysis_output_limit", "An analysis output grew past the byte limit")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        signature = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
                                   value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if signature(before) != signature(after) or signature(after) != signature(current):
            raise ToolFailure("analysis_output_invalid", "Analysis output changed during collection")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


_RESULT_REPAIR_HINT = (' Tool artifact schema, separate from the final answer: '
    'required summary (object); optional metrics (object) and limitations (array of strings). '
    'No other top-level keys. Put method and requested counts inside summary. '
    'Example: {"summary":{"count":1,"method":"read actual inputs"},"limitations":["scope"]}.')


def _validated_result(content):
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ToolFailure("analysis_output_invalid", "analysis_result.json must be valid UTF-8 JSON") from exc
    if not isinstance(value, dict) or not isinstance(value.get("summary"), dict):
        raise ToolFailure("analysis_output_invalid", "analysis_result.json requires an object summary" + _RESULT_REPAIR_HINT)
    if set(value) - {"summary", "metrics", "limitations"}:
        raise ToolFailure("analysis_output_invalid", "analysis_result.json contains unsupported top-level fields" + _RESULT_REPAIR_HINT)
    if "metrics" in value and not isinstance(value["metrics"], dict):
        raise ToolFailure("analysis_output_invalid", "analysis result metrics must be an object" + _RESULT_REPAIR_HINT)
    if ("limitations" in value and (not isinstance(value["limitations"], list)
            or not all(isinstance(item, str) for item in value["limitations"]))):
        raise ToolFailure("analysis_output_invalid", "analysis result limitations must be strings" + _RESULT_REPAIR_HINT)
    try:
        encoded = canonical_json(value).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ToolFailure("analysis_output_invalid", "analysis_result.json must contain finite JSON values") from exc
    if len(encoded) > MAX_SUMMARY_BYTES:
        raise ToolFailure("analysis_output_limit", "analysis_result.json exceeds the summary byte limit")
    return value


def collect_outputs(context, output_dir, *, analysis_id=None):
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ToolFailure("analysis_output_invalid", "Analysis output directory is missing or unsafe")
    files, entries = [], 0
    for path in output_dir.rglob("*"):
        entries += 1
        if entries > MAX_OUTPUT_ENTRIES:
            raise ToolFailure("analysis_output_limit", "Analysis output tree has too many entries")
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
            raise ToolFailure("analysis_output_invalid", "Analysis output tree contains a link or special file")
        if stat.S_ISREG(info.st_mode):
            files.append(path)
    if len(files) > MAX_OUTPUT_FILES:
        raise ToolFailure("analysis_output_limit", "Analysis produced too many output files")
    payloads, total = [], 0
    for path in sorted(files, key=lambda item: item.relative_to(output_dir).as_posix()):
        content = _read_regular_output(path)
        total += len(content)
        if total > MAX_TOTAL_OUTPUT_BYTES:
            raise ToolFailure("analysis_output_limit", "Analysis outputs exceed the total byte limit")
        payloads.append((path, content))
    result_document = None
    for path, content in payloads:
        if path.relative_to(output_dir).as_posix() == "analysis_result.json":
            result_document = _validated_result(content)
    artifacts, roles = [], []
    for path, content in payloads:
        relative = path.relative_to(output_dir).as_posix()
        public_name = relative.replace("/", "__")
        role = "result" if relative == "analysis_result.json" else "output"
        artifact = context.store.put_artifact(content, public_name, _media_type(path),
            {"analysis_id": analysis_id, "role": role, "relative_path": relative,
             "sha256": hashlib.sha256(content).hexdigest()},
            session_id=context.session_id, run_id=context.run_id)
        artifacts.append(artifact)
        roles.append((role, artifact["id"]))
    return artifacts, roles, result_document


def _prepare_workspace(context, source, language, parameters):
    workspace = Path(context.workspace)
    if workspace.is_symlink() or workspace.resolve() != workspace:
        raise ToolFailure("analysis_input_denied", "Analysis workspace is unsafe", "denied")
    for name in ("inputs", "outputs"):
        target = workspace / name
        target.mkdir(mode=0o700, exist_ok=False)
    # JobManager creates this private directory before launching the trusted
    # handler so its own process environment has a safe TMPDIR. Accept only
    # that exact empty, owner-private directory; analysis still owns all later
    # contents and the OS sandbox is the only writer after this point.
    temporary = workspace / "tmp"
    if temporary.exists():
        info = temporary.lstat()
        if (temporary.is_symlink() or not stat.S_ISDIR(info.st_mode)
                or temporary.resolve().parent != workspace or info.st_mode & 0o077
                or any(temporary.iterdir())):
            raise ToolFailure("analysis_input_denied", "Analysis temporary workspace is unsafe", "denied")
    else:
        temporary.mkdir(mode=0o700, exist_ok=False)
    extension = ".py" if language == "python" else ".sh"
    source_path = workspace / ("analysis" + extension)
    source_path.write_text(source, encoding="utf-8", errors="strict")
    source_path.chmod(0o500)
    (workspace / "parameters.json").write_text(canonical_json(parameters), encoding="utf-8")
    return source_path


def _load_source(store, artifact_id):
    metadata = next((item for item in store.artifacts() if item["id"] == artifact_id), None)
    if metadata is None or metadata["size"] > MAX_SOURCE_BYTES:
        raise ToolFailure("analysis_source_invalid", "Analysis source artifact is missing or oversized")
    try:
        return store.read_artifact(artifact_id, max_bytes=MAX_SOURCE_BYTES).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ToolFailure("analysis_source_invalid", "Analysis source must be UTF-8") from exc


def _command(language, source_path):
    if language == "python":
        return [sys.executable, "-I", str(source_path)]
    if language == "bash":
        return ["/bin/bash", "--noprofile", "--norc", str(source_path)]
    raise ToolFailure("analysis_source_invalid", "Analysis language must be python or bash")


def _log_artifacts(context, analysis_id, stdout, stderr):
    values, roles = [], []
    for role, content in (("stdout", stdout), ("stderr", stderr)):
        artifact = context.store.put_artifact(content, role + ".txt", "text/plain",
            {"analysis_id": analysis_id, "role": role}, session_id=context.session_id, run_id=context.run_id)
        values.append(artifact)
        roles.append((role, artifact["id"]))
    return values, roles


def analysis_job(context, *, analysis_id, language, code_artifact_id, inputs,
                 parameters, timeout_seconds, method_id=None):
    history = AnalysisHistory(context.store)
    backend = "not_started"
    published, roles = [], []
    try:
        history.mark_running(analysis_id, context.job_id)
        if language == "bash" and sys.platform == "darwin":
            raise ToolFailure("sandbox_unavailable",
                "The macOS analysis backend cannot safely contain forking Bash workloads; "
                "use code_interpreter or a Linux Bubblewrap host")
        source = _load_source(context.store, code_artifact_id)
        source_path = _prepare_workspace(context, source, language, parameters)
        snapshots = snapshot_inputs(context, inputs)
        context.progress("Executing model-authored analysis in the OS sandbox", language=language)
        try:
            sandbox = select_sandbox()
        except SandboxUnavailable as exc:
            raise ToolFailure("sandbox_unavailable", str(exc)) from exc
        try:
            result = sandbox.run(_command(language, source_path), context.workspace,
                timeout_seconds=timeout_seconds, cancel_check=context.check_cancelled)
        except (OSError, SandboxUnavailable) as exc:
            raise ToolFailure("sandbox_unavailable",
                "The host could not start the analysis sandbox") from exc
        backend = result.backend
        logs, log_roles = _log_artifacts(context, analysis_id, result.stdout, result.stderr)
        published.extend(logs)
        roles.extend(log_roles)
        if result.timed_out:
            raise ToolFailure("analysis_timeout", "Analysis exceeded its wall-clock limit")
        if result.limit_reason:
            raise ToolFailure("analysis_resource_limit",
                              f"Analysis exceeded its {result.limit_reason.replace('_', ' ')}")
        if result.returncode != 0:
            if result.returncode == 71 and "sandbox_apply" in result.stderr:
                raise ToolFailure("sandbox_unavailable", "The host refused to apply the analysis sandbox")
            if result.returncode < 0:
                terminating_signal = -result.returncode
                if terminating_signal in {getattr(signal, "SIGXCPU", -1),
                                          getattr(signal, "SIGXFSZ", -1)}:
                    raise ToolFailure("analysis_resource_limit",
                                      f"Analysis exceeded an enforced resource limit (signal {terminating_signal})")
                raise ToolFailure("analysis_signal", f"Analysis ended by signal {terminating_signal}")
            details = {"stderr_preview": result.stderr[:2000]}
            message = f"Analysis exited with status {result.returncode}"
            if _NUMPY_JSON_SERIALIZATION_ERROR.search(result.stderr):
                repair_hint = _NUMPY_JSON_REPAIR_HINT
                details["repair_hint"] = repair_hint
                # Error previews retain the message but may omit details when
                # the complete tool result exceeds the model context limit.
                message += f". Repair hint: {repair_hint}"
            raise ToolFailure("analysis_exit_nonzero", message, details=details)
        outputs, output_roles, document = collect_outputs(context, Path(context.workspace) / "outputs",
                                                           analysis_id=analysis_id)
        published.extend(outputs)
        roles.extend(output_roles)
        summary = document["summary"] if document is not None else {"stdout_preview": result.stdout[:4000]}
        return ToolResult(data={"analysis_id": analysis_id, "method_id": method_id, "summary": summary,
            "inputs": snapshots, "sandbox_backend": backend,
            "outputs": _output_descriptors(published, roles)}, artifacts=published,
            job_id=context.job_id)
    except ToolCancelled as exc:
        error = {"code": "analysis_cancelled", "message": str(exc)}
        return _failure(error["code"], error["message"], "cancelled", analysis_id=analysis_id,
                        sandbox_backend=backend, artifacts=published)
    except ToolFailure as exc:
        error = {"code": exc.code, "message": str(exc)}
        if exc.details:
            error["details"] = exc.details
        return _failure(exc.code, str(exc), analysis_id=analysis_id, details=exc.details,
                        sandbox_backend=backend, artifacts=published)
    except BaseException:
        # Unexpected exceptions may contain Store-private paths or library
        # internals. Preserve a stable public outcome without reflecting those
        # details into model-visible history.
        error = {"code": "analysis_internal", "message": "Analysis handler failed internally"}
        return _failure(error["code"], error["message"], analysis_id=analysis_id,
                        sandbox_backend=backend, artifacts=published)
