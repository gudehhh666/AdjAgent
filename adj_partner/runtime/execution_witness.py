"""Read-only verification of a bound host execution witness.

Witness mode proves that a trusted analysis action completed against the
host-bound public inputs.  It deliberately does not compare the analysis
summary with the model's final narrative or claim semantic correctness.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePath

from ..analysis.history import AnalysisHistory
from ..contracts import ToolResult
from ..execution.manager import JobManager
from ..research.acceptance import task_binding as make_task_binding
from ..tools.catalog import canonical_json
from .execution_delivery import (
    MAX_JOB_RESULT_BYTES, MAX_SOURCE_BYTES, MAX_SUMMARY_BYTES,
    TRUSTED_ANALYSIS_HANDLER, TRUSTED_ANALYSIS_TOOLS, VerificationFailure,
    _action_hash, _analysis_request, _artifact_by_role, _bounded_canonical,
    _check_artifact, _equal_json, _job_fingerprint, _job_request,
    _read_only_history, _read_only_manager, _safe_input_manifest,
    _strict_response_json, _successful_candidates,
)


MAX_INPUTS = 128
MAX_INPUT_BYTES = 64 * 1024 * 1024


def _fail(message, cause=None):
    if cause is None:
        raise VerificationFailure(message)
    raise VerificationFailure(message) from cause


def _canonical_path(value, label):
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value or "\x00" in value:
        _fail(f"{label} is malformed")
    path = Path(value).expanduser()
    if not path.is_absolute():
        _fail(f"{label} must be absolute")
    try:
        if path.is_symlink():
            _fail(f"{label} is a symlink")
    except OSError as exc:
        _fail(f"{label} cannot be inspected", exc)
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail(f"{label} cannot be resolved", exc)
    return resolved


def _root_id(path):
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()


def _validate_root(path, label="allowed root"):
    resolved = _canonical_path(path, label)
    try:
        if resolved.is_symlink() or not resolved.is_dir():
            _fail(f"{label} must be a non-symlink directory")
    except OSError as exc:
        _fail(f"{label} cannot be inspected", exc)
    return resolved


def _is_beneath(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_regular_file(path, label):
    try:
        if path.is_symlink() or not path.is_file():
            _fail(f"{label} must be a non-symlink regular file")
        size = path.stat().st_size
    except OSError as exc:
        _fail(f"{label} cannot be inspected", exc)
    if size < 0 or size > MAX_INPUT_BYTES:
        _fail(f"{label} exceeds its byte limit")
    return size


def _reject_internal_symlinks(path, root, label):
    try:
        relative = path.relative_to(root)
        cursor = root
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                _fail(f"{label} traverses a symlink")
    except ValueError:
        _fail(f"{label} escapes its root")
    except OSError as exc:
        _fail(f"{label} cannot be inspected", exc)


def _reject_lexical_symlinks(path, root, label):
    """Reject symlink components before resolve() erases their identity."""
    try:
        relative = Path(path).expanduser().relative_to(root)
    except ValueError:
        return
    cursor = root
    try:
        for component in relative.parts:
            cursor = cursor / component
            if cursor.is_symlink():
                _fail(f"{label} traverses a symlink")
    except OSError as exc:
        _fail(f"{label} cannot be inspected", exc)


def _reject_any_symlink(path, label):
    """Reject aliases in a trusted worker path before canonicalization."""
    path = Path(path)
    cursor = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            cursor /= component
            if cursor.is_symlink():
                _fail(f"{label} traverses a symlink")
    except OSError as exc:
        _fail(f"{label} cannot be inspected", exc)


def _sha256(path):
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail("Bound input cannot be read", exc)
    return digest.hexdigest()


def _binding_digest(binding):
    body = dict(binding)
    body.pop("binding_sha256", None)
    try:
        return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    except (TypeError, ValueError, RecursionError) as exc:
        _fail("Execution binding is not finite JSON", exc)


def _validate_task_binding(value):
    if not isinstance(value, dict) or set(value) != {"id", "requirements_hash", "observed_revision"}:
        _fail("Execution task binding is malformed")
    if (not isinstance(value["id"], str) or not value["id"]
            or not isinstance(value["requirements_hash"], str)
            or not isinstance(value["observed_revision"], int)
            or isinstance(value["observed_revision"], bool) or value["observed_revision"] < 0):
        _fail("Execution task binding fields are malformed")


def _validate_binding(binding, *, task_binding=None, response_schema=None, session_roots=None):
    if not isinstance(binding, dict):
        _fail("Execution binding is malformed")
    required = {"task_binding", "allowed_roots", "inputs", "result_relative_path",
                "trusted_tools", "response_schema_sha256", "binding_sha256"}
    if set(binding) != required:
        _fail("Execution binding fields are malformed")
    _validate_task_binding(binding["task_binding"])
    if task_binding is not None and binding["task_binding"] != task_binding:
        _fail("Execution task binding does not match the current task")
    if response_schema is not None:
        schema_hash = hashlib.sha256(canonical_json(response_schema).encode("utf-8")).hexdigest()
        if binding["response_schema_sha256"] != schema_hash:
            _fail("Execution response schema binding does not match the task")
    if binding["result_relative_path"] != "analysis_result.json":
        _fail("Execution result path is not the trusted analysis result")
    if (not isinstance(binding["trusted_tools"], list)
            or set(binding["trusted_tools"]) != set(TRUSTED_ANALYSIS_TOOLS)):
        _fail("Execution trusted tool binding is malformed")
    roots = binding["allowed_roots"]
    if not isinstance(roots, list) or not 1 <= len(roots) <= 32:
        _fail("Execution allowed roots are malformed")
    root_map = {}
    configured = None
    if session_roots is not None:
        configured = {_validate_root(item, "session allowed root") for item in session_roots}
    for item in roots:
        if not isinstance(item, dict) or set(item) != {"root_id", "path"}:
            _fail("Execution allowed root entry is malformed")
        root = _validate_root(item["path"])
        if item["root_id"] != _root_id(root):
            _fail("Execution allowed root identity is invalid")
        if configured is not None and root not in configured:
            _fail("Execution binding root is outside the session contract")
        if item["root_id"] in root_map:
            _fail("Execution allowed roots are ambiguous")
        root_map[item["root_id"]] = root
    inputs = binding["inputs"]
    if not isinstance(inputs, list) or not inputs or len(inputs) > MAX_INPUTS:
        _fail("Execution bound inputs are malformed")
    seen = set()
    for item in inputs:
        if (not isinstance(item, dict)
                or set(item) != {"root_id", "relative_path", "name", "size", "sha256"}
                or item["root_id"] not in root_map
                or not isinstance(item["relative_path"], str)
                or not item["relative_path"] or Path(item["relative_path"]).is_absolute()
                or "\\" in item["relative_path"]
                or PurePath(item["relative_path"]).name != item["name"]
                or not isinstance(item["name"], str) or not item["name"]
                or type(item["size"]) is not int or item["size"] < 0
                or not isinstance(item["sha256"], str)
                or len(item["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in item["sha256"])):
            _fail("Execution bound input entry is malformed")
        key = (item["root_id"], item["relative_path"])
        if key in seen:
            _fail("Execution bound inputs are ambiguous")
        seen.add(key)
        _reject_lexical_symlinks(root_map[item["root_id"]] / item["relative_path"],
                                 root_map[item["root_id"]], "Execution bound input")
        source = (root_map[item["root_id"]] / item["relative_path"]).resolve()
        if not _is_beneath(source, root_map[item["root_id"]]):
            _fail("Execution bound input escapes its root")
        _reject_internal_symlinks(source, root_map[item["root_id"]], "Execution bound input")
        size = _validate_regular_file(source, "Execution bound input")
        if size != item["size"] or _sha256(source) != item["sha256"]:
            _fail("Execution bound input hash does not match the task-start snapshot")
    if not isinstance(binding["binding_sha256"], str) or binding["binding_sha256"] != _binding_digest(binding):
        _fail("Execution binding fingerprint is invalid")
    return root_map


def bind_execution_inputs(task_binding, allowed_roots, input_paths, response_schema=None,
                          trusted_tools=TRUSTED_ANALYSIS_TOOLS,
                          result_relative_path="analysis_result.json"):
    """Create a host-owned input binding without granting tool capabilities."""
    _validate_task_binding(task_binding)
    if not isinstance(allowed_roots, (list, tuple)) or not allowed_roots:
        _fail("Execution allowed roots are required")
    roots = []
    root_map = {}
    for value in allowed_roots:
        root = _validate_root(value)
        identifier = _root_id(root)
        if identifier in root_map:
            _fail("Execution allowed roots are ambiguous")
        root_map[identifier] = root
        roots.append({"root_id": identifier, "path": str(root)})
    if not isinstance(input_paths, (list, tuple)) or not input_paths or len(input_paths) > MAX_INPUTS:
        _fail("Execution bound inputs are malformed")
    inputs = []
    for value in input_paths:
        for root in root_map.values():
            _reject_lexical_symlinks(value, root, "Execution input")
        source = _canonical_path(value, "Execution input path")
        owners = [(identifier, root) for identifier, root in root_map.items() if _is_beneath(source, root)]
        if len(owners) != 1:
            _fail("Execution input path is outside or ambiguous across allowed roots")
        identifier, root = owners[0]
        _reject_internal_symlinks(source, root, "Execution input")
        size = _validate_regular_file(source, "Execution input")
        relative = source.relative_to(root).as_posix()
        inputs.append({"root_id": identifier, "relative_path": relative,
                       "name": source.name, "size": size, "sha256": _sha256(source)})
    if len({(item["root_id"], item["relative_path"]) for item in inputs}) != len(inputs):
        _fail("Execution bound inputs are ambiguous")
    if not isinstance(trusted_tools, (list, tuple, set, frozenset)) or set(trusted_tools) != set(TRUSTED_ANALYSIS_TOOLS):
        _fail("Execution trusted tool set is not the fixed host set")
    if result_relative_path != "analysis_result.json":
        _fail("Execution result path is not the trusted analysis result")
    if response_schema is None:
        schema_hash = None
    else:
        try:
            schema_hash = hashlib.sha256(canonical_json(response_schema).encode("utf-8")).hexdigest()
        except (TypeError, ValueError, RecursionError) as exc:
            _fail("Execution response schema is not finite JSON", exc)
    binding = {"task_binding": dict(task_binding), "allowed_roots": roots, "inputs": inputs,
               "result_relative_path": result_relative_path, "trusted_tools": sorted(TRUSTED_ANALYSIS_TOOLS),
               "response_schema_sha256": schema_hash}
    binding["binding_sha256"] = _binding_digest(binding)
    return binding


def validate_execution_binding(binding, *, task_binding=None, response_schema=None, session_roots=None):
    """Validate a previously host-created binding without changing state."""
    _validate_binding(binding, task_binding=task_binding, response_schema=response_schema,
                      session_roots=session_roots)
    return binding


def _record_matches_binding(record, binding):
    raw_inputs = record.get("inputs")
    manifest = _safe_input_manifest(raw_inputs)
    if len(manifest) != len(binding["inputs"]):
        return False

    roots = {item["root_id"]: _canonical_path(item["path"], "bound root")
             for item in binding["allowed_roots"]}
    expected_by_path = {}
    for item in binding["inputs"]:
        root = roots[item["root_id"]]
        expected_path = (root / item["relative_path"]).resolve(strict=True)
        expected_by_path[(item["root_id"], item["relative_path"])] = {
            "path": expected_path,
            "name": item["name"], "size": item["size"], "sha256": item["sha256"],
        }

    matched = set()
    for raw, public in zip(raw_inputs, manifest):
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            _fail("Analysis input manifest paths are malformed")
        _reject_any_symlink(raw["path"], "Analysis input path")
        recorded_path = _canonical_path(raw["path"], "Analysis input path")
        candidates = [key for key, expected in expected_by_path.items()
                      if expected["path"] == recorded_path]
        if len(candidates) != 1:
            # Same-content copies are ineligible for this host binding.
            return False
        key = candidates[0]
        if key in matched:
            _fail("Successful analysis input paths are ambiguous")
        expected = expected_by_path[key]
        if (public["name"], public["size"], public["sha256"]) != (
                expected["name"], expected["size"], expected["sha256"]):
            _fail("Successful analysis input manifest contradicts the host binding")
        matched.add(key)
    return len(matched) == len(expected_by_path)


def _verify_candidate(store, run_id, session_id, candidate, binding):
    action, action_result, action_data = candidate
    call_id = action.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        _fail("Analysis action call identity is malformed")
    job_id = action_result.get("job_id")
    analysis_id = action_data.get("analysis_id")
    if not isinstance(job_id, str) or not isinstance(analysis_id, str):
        _fail("Analysis action identity is malformed")
    jobs = _read_only_manager(store)
    try:
        raw_job = jobs._read(job_id, session_id)
        job = jobs.get(job_id, session_id)
        job_result = jobs.result(job_id, session_id)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail("Analysis job identity cannot be resolved", exc)
    if (raw_job.get("run_id") != run_id or raw_job.get("tool_name") != action.get("tool_name")
            or raw_job.get("handler_ref") != TRUSTED_ANALYSIS_HANDLER
            or raw_job.get("status") != "succeeded" or raw_job.get("idempotency_key") != call_id):
        _fail("Analysis job identity or trusted handler protocol is invalid")
    _job_fingerprint(raw_job)
    if not isinstance(job_result, ToolResult) or job_result.status != "ok":
        return None
    if job_result.job_id not in {None, job_id} or not isinstance(job_result.data, dict):
        _fail("Job result identity or data envelope is malformed")
    if job_result.data.get("analysis_id") != analysis_id:
        _fail("Job result names a different analysis record")
    history = _read_only_history(store)
    try:
        record = history.get_record(analysis_id)
    except (KeyError, TypeError) as exc:
        _fail("Analysis history identity cannot be resolved", exc)
    if (record.get("session_id") != session_id or record.get("run_id") != run_id
            or record.get("job_id") != job_id or record.get("status") != "succeeded"):
        _fail("Analysis history identity or status is invalid")
    if not _record_matches_binding(record, binding):
        return None
    code = _artifact_by_role(record, "code")
    result_artifact = _artifact_by_role(record, "result")
    source_name = "analysis.py" if record.get("language") == "python" else "analysis.sh"
    source_media = "text/x-python" if record.get("language") == "python" else "text/x-shellscript"
    code_bytes, _ = _check_artifact(store, code, "Analysis source", session_id=session_id, run_id=run_id,
                                     expected_name=source_name, expected_media_type=source_media,
                                     expected_provenance={"kind": "model_authored_analysis", "language": record.get("language"), "sha256": code.get("sha256")}, max_bytes=MAX_SOURCE_BYTES)
    _action_hash(store, run_id, call_id, action)
    _analysis_request(action, record, code_bytes)
    _job_request(job, record, analysis_id)
    result_bytes, result_metadata = _check_artifact(store, result_artifact, "Analysis result", session_id=session_id, run_id=run_id,
                                                     expected_name="analysis_result.json", expected_media_type="application/json",
                                                     expected_provenance={"analysis_id": analysis_id, "role": "result", "relative_path": "analysis_result.json", "sha256": result_artifact.get("sha256")}, max_bytes=MAX_SUMMARY_BYTES)
    job_result_ref = raw_job.get("result_artifact")
    if not isinstance(job_result_ref, dict):
        _fail("Job result artifact reference is malformed")
    _check_artifact(store, {"artifact_id": job_result_ref.get("id"), "sha256": job_result_ref.get("sha256")}, "Job result", session_id=session_id, run_id=run_id,
                    expected_name="job-result.json", expected_media_type="application/json",
                    expected_provenance={"job_id": job_id, "definition_hash": raw_job.get("definition_hash"), "role": "job_result"}, max_bytes=MAX_JOB_RESULT_BYTES)
    try:
        result_document = _strict_response_json(result_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        _fail("Analysis result artifact is not strict JSON", exc)
    if (not isinstance(result_document, dict) or set(result_document) - {"summary", "metrics", "limitations"}
            or not isinstance(result_document.get("summary"), dict)):
        _fail("Analysis result data envelope is malformed")
    summary = result_document["summary"]
    _bounded_canonical(result_document, "Analysis result", max_bytes=MAX_SUMMARY_BYTES)
    if store.redact(summary) != summary:
        _fail("Analysis summary contains redacted data")
    for source, label in ((action_data, "Action summary"), (job_result.data, "Job summary"), (record, "Analysis history summary")):
        if not isinstance(source.get("summary"), dict):
            _fail(f"{label} is missing from the successful result")
        _equal_json(source["summary"], summary, label)
    outputs = job_result.data.get("outputs")
    result_outputs = [item for item in outputs or [] if isinstance(item, dict) and item.get("role") == "result"]
    if len(result_outputs) != 1:
        _fail("Job result output descriptors are malformed")
    output = result_outputs[0]
    if (output.get("artifact_id") != result_artifact.get("artifact_id") or output.get("sha256") != result_artifact.get("sha256")
            or output.get("name") != result_metadata.get("name") or output.get("media_type") != result_metadata.get("media_type")
            or output.get("size") != result_metadata.get("size")):
        _fail("Job result and history artifact identities disagree")
    claimed = [item for item in job_result.artifacts if isinstance(item, dict) and item.get("id") == result_artifact.get("artifact_id")]
    if len(claimed) != 1 or claimed[0].get("sha256") != result_artifact.get("sha256"):
        _fail("Job result artifact claims are malformed")
    summary_json = _bounded_canonical(summary, "Analysis summary").decode("utf-8")
    provenance = {"call_id": call_id, "run_id": run_id, "session_id": session_id,
                  "analysis_id": analysis_id, "job_id": job_id, "tool_name": action.get("tool_name"),
                  "handler_ref": TRUSTED_ANALYSIS_HANDLER, "analysis_source_artifact_id": code["artifact_id"],
                  "analysis_source_sha256": code["sha256"], "result_artifact_id": result_artifact["artifact_id"],
                  "result_artifact_sha256": result_artifact["sha256"], "job_result_sha256": raw_job["result_artifact"]["sha256"],
                  "input_manifest": _safe_input_manifest(record.get("inputs")), "sandbox_backend": record.get("sandbox_backend"),
                  "summary_sha256": hashlib.sha256(summary_json.encode("utf-8")).hexdigest(),
                  "task_binding": dict(binding["task_binding"]), "binding_sha256": binding["binding_sha256"],
                  "semantic_validation": "not_performed", "narrative_derivation_validated": False,
                  "bound_input_evidence": "worker_manifest_path_root_relative_name_size_sha256_match; program_actual_read_use_not_proven",
                  "verification": ["job_terminal_succeeded", "analysis_record_terminal_succeeded", "result_artifact_hash_verified", "bound_input_identity_verified"]}
    if store.redact(provenance) != provenance:
        _fail("Execution witness provenance contains redacted data")
    return {"call_id": call_id, "analysis_id": analysis_id, "job_id": job_id,
            "summary_sha256": provenance["summary_sha256"], "provenance": provenance}


def verify_execution_witness(store, run_id, binding, *, response_schema=None):
    """Return all valid bound witnesses, or ``None`` when none is eligible."""
    if not isinstance(run_id, str) or not run_id:
        _fail("Execution witness run identity is malformed")
    try:
        run = store.get_run(run_id)
        session = store.get_session(run["session_id"])
        task = store.get_task(run["task_id"])
    except (KeyError, TypeError) as exc:
        _fail("Execution witness run identity cannot be loaded", exc)
    expected_binding = make_task_binding(task)
    _validate_binding(binding, task_binding=expected_binding, response_schema=response_schema,
                      session_roots=session.get("config", {}).get("allowed_roots", []))
    candidates = _successful_candidates(store, run_id)
    witnesses = []
    for candidate in candidates:
        witness = _verify_candidate(store, run_id, run["session_id"], candidate, binding)
        if witness is not None:
            witnesses.append(witness)
    if not witnesses:
        return None
    return {"kind": "execution_witness", "witnesses": witnesses,
            "witness_count": len(witnesses), "semantic_validation": "not_performed",
            "narrative_derivation_validated": False,
            "assurance": "At least one trusted analysis completed with worker input identity matching the "
                         "host-bound public snapshot by root, relative path, basename, size, and SHA-256. "
                         "This verifies input path identity in the durable record; it does not prove that the "
                         "program actually read the file or that the computation was semantically correct. "
                         "The final narrative was not independently validated."}

