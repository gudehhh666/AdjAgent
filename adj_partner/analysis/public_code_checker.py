"""Bounded host checks for code against an explicitly public contract.

This module is deliberately independent of the analysis worker and durable
history.  It is a controller-side adapter: candidate code is placed in a
temporary workspace and run by the existing OS sandbox, while all pass/fail
decisions are made from the child process' bounded value descriptors here.
The result is a development diagnostic, never an official benchmark score.
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import math
import os
from contextlib import nullcontext
from pathlib import Path
import re
import tempfile
from typing import Any

from .sandbox import SandboxUnavailable, select_sandbox
from ..tools.policy import ToolCancelled


MAX_SOURCE_BYTES = 128 * 1024
MAX_SUBMISSION_BYTES = 512 * 1024
MAX_MANIFEST_BYTES = 512 * 1024
MAX_RESULT_BYTES = 64 * 1024
MAX_CHECKS = 256
MAX_STEPS = 64
MAX_JSON_DEPTH = 32
MAX_TIMEOUT_SECONDS = 30.0
MAX_RESULT_ELEMENTS = 100_000
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PUBLIC_CHECKER_VERSION = "public-code-check-v1"
PUBLIC_RESULT_VERSION = "public-code-check-result-v1"
_ALLOWED_IMPORT_ROOTS = {"collections", "functools", "itertools", "math", "numpy", "scipy", "time", "typing"}
_CHECK_KEYS = {"id", "kind", "args", "kwargs", "constraints", "public_basis"}
_CHECK_KEYS_WITH_REF = _CHECK_KEYS | {"public_ref"}
_STEP_KEYS = {"step_id", "function_name", "signature", "checks"}
_SUPPORTED_CONSTRAINTS = {
    "return_type", "shape", "finite", "finite_elements", "return_arity", "items",
    "same_shape_as_arg", "l2_norm", "column_unit_norm", "column_pairwise_orthogonal",
    "net_force_sum_zero", "exact_first_line", "syntax", "rank", "axis_1_size",
}


class PublicCodeCheckError(ValueError):
    """The public checker input or its bounded result is malformed."""


def required_imports_from_dependencies(source: str) -> list[dict]:
    """Parse the public solver's dependency block into bounded host-owned imports."""
    if not isinstance(source, str) or len(source.encode("utf-8")) > 16 * 1024:
        raise PublicCodeCheckError("public required_dependencies is malformed or too large")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise PublicCodeCheckError("public required_dependencies is not valid import syntax") from exc
    imports = []
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if alias.name.split(".", 1)[0] not in _ALLOWED_IMPORT_ROOTS:
                    raise PublicCodeCheckError(f"public dependency module is not allowlisted: {alias.name}")
                imports.append({"kind": "import", "module": alias.name,
                                "alias": alias.asname or alias.name.split(".", 1)[0]})
        elif isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module:
            if statement.module.split(".", 1)[0] not in _ALLOWED_IMPORT_ROOTS:
                raise PublicCodeCheckError(f"public dependency module is not allowlisted: {statement.module}")
            if not statement.names or any(alias.name == "*" for alias in statement.names):
                raise PublicCodeCheckError("star or empty public dependency imports are not supported")
            imports.append({"kind": "from", "module": statement.module,
                            "names": [{"name": alias.name, "alias": alias.asname} for alias in statement.names]})
        else:
            raise PublicCodeCheckError("public required_dependencies may contain import statements only")
    # Preserve declaration order while rejecting ambiguous duplicate bindings.
    encoded = [_canonical(item, maximum=1024, label="required import") for item in imports]
    if len(set(encoded)) != len(encoded):
        raise PublicCodeCheckError("public required_dependencies contains duplicate imports")
    if len(imports) > 32:
        raise PublicCodeCheckError("too many public required imports")
    return imports


def _validate_required_imports(value: Any) -> list[dict]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 32:
        raise PublicCodeCheckError("manifest required_imports is malformed")
    validated = []
    for item in value:
        if not isinstance(item, dict):
            raise PublicCodeCheckError("manifest required import is malformed")
        kind, module = item.get("kind"), item.get("module")
        if (not isinstance(module, str) or not module or module.split(".", 1)[0] not in _ALLOWED_IMPORT_ROOTS):
            raise PublicCodeCheckError("manifest required import module is not allowlisted")
        if kind == "import" and set(item) == {"kind", "module", "alias"}:
            alias = item["alias"]
            if not isinstance(alias, str) or not alias.isidentifier():
                raise PublicCodeCheckError("manifest import alias is malformed")
        elif kind == "from" and set(item) == {"kind", "module", "names"}:
            names = item["names"]
            if not isinstance(names, list) or not names or len(names) > 32:
                raise PublicCodeCheckError("manifest from-import names are malformed")
            for name in names:
                if (not isinstance(name, dict) or set(name) != {"name", "alias"}
                        or not isinstance(name["name"], str) or not name["name"].isidentifier()
                        or (name["alias"] is not None and (not isinstance(name["alias"], str)
                            or not name["alias"].isidentifier()))):
                    raise PublicCodeCheckError("manifest from-import name is malformed")
        else:
            raise PublicCodeCheckError("manifest required import shape is malformed")
        validated.append(copy.deepcopy(item))
    encodings = [_canonical(item, maximum=1024, label="required import") for item in validated]
    if len(set(encodings)) != len(encodings):
        raise PublicCodeCheckError("manifest required_imports contains duplicates")
    return validated


def _walk_json(value: Any, *, depth=0, seen=None):
    if depth > MAX_JSON_DEPTH:
        raise PublicCodeCheckError("JSON exceeds the public checker depth limit")
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list, tuple)):
        identity = id(value)
        if identity in seen:
            raise PublicCodeCheckError("JSON contains a cycle")
        seen.add(identity)
        try:
            if isinstance(value, dict):
                if not all(isinstance(key, str) for key in value):
                    raise PublicCodeCheckError("JSON object keys must be strings")
                for key, item in value.items():
                    _walk_json(key, depth=depth + 1, seen=seen)
                    _walk_json(item, depth=depth + 1, seen=seen)
            else:
                for item in value:
                    _walk_json(item, depth=depth + 1, seen=seen)
        finally:
            seen.remove(identity)
    elif isinstance(value, float) and not math.isfinite(value):
        raise PublicCodeCheckError("JSON values must be finite")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise PublicCodeCheckError(f"Unsupported JSON value type: {type(value).__name__}")


def _canonical(value: Any, *, maximum: int, label: str) -> bytes:
    _walk_json(value)
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PublicCodeCheckError(f"{label} is not finite JSON") from exc
    if len(encoded) > maximum:
        raise PublicCodeCheckError(f"{label} exceeds {maximum} bytes")
    return encoded


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def checker_source_digest(path: os.PathLike[str] | str | None = None) -> str:
    """Return the SHA-256 pin for the checker source used by a controller."""
    source_path = Path(path) if path is not None else Path(__file__)
    return _sha(source_path.read_bytes())


def _hash_arg(value: Any, *, label: str, maximum: int) -> str:
    return _sha(_canonical(value, maximum=maximum, label=label))


def _validate_manifest(manifest: dict) -> list[dict]:
    if manifest.get("schema_version") != "public-code-check-v1":
        raise PublicCodeCheckError("unsupported public checker manifest schema")
    if not isinstance(manifest.get("source_path"), str) or not manifest["source_path"]:
        raise PublicCodeCheckError("manifest source_path is required")
    if not isinstance(manifest.get("source_sha256"), str) or not HASH_RE.fullmatch(manifest["source_sha256"]):
        raise PublicCodeCheckError("manifest source_sha256 is malformed")
    _validate_required_imports(manifest.get("required_imports"))
    steps = copy.deepcopy(manifest.get("ordered_steps"))
    if not isinstance(steps, list) or not 0 < len(steps) <= MAX_STEPS:
        raise PublicCodeCheckError("manifest ordered_steps is malformed")
    step_ids, check_ids, count = set(), set(), 0
    for step in steps:
        if not isinstance(step, dict) or set(step) != _STEP_KEYS:
            raise PublicCodeCheckError("manifest step shape is malformed")
        step_id = step["step_id"]
        function_name = step["function_name"]
        if (not isinstance(step_id, str) or not step_id or step_id in step_ids
                or not isinstance(function_name, str) or not function_name.isidentifier()):
            raise PublicCodeCheckError("manifest step identity is malformed")
        if not isinstance(step["signature"], str) or not step["signature"].strip():
            raise PublicCodeCheckError("manifest signature is malformed")
        checks = step["checks"]
        if not isinstance(checks, list):
            raise PublicCodeCheckError("manifest checks are malformed")
        step_ids.add(step_id)
        has_fixture = False
        for check in checks:
            count += 1
            if count > MAX_CHECKS or not isinstance(check, dict) or set(check) not in (_CHECK_KEYS, _CHECK_KEYS_WITH_REF):
                raise PublicCodeCheckError("manifest check shape is malformed")
            if (not isinstance(check["id"], str) or not check["id"] or check["id"] in check_ids
                    or check["kind"] not in {"compile", "signature", "fixture", "unavailable"}):
                raise PublicCodeCheckError("manifest check identity is malformed")
            if not isinstance(check["args"], list) or not isinstance(check["kwargs"], dict):
                raise PublicCodeCheckError("manifest check arguments are malformed")
            if not isinstance(check["constraints"], dict) or not isinstance(check["public_basis"], str):
                raise PublicCodeCheckError("manifest check metadata is malformed")
            if "public_ref" in check and (not isinstance(check["public_ref"], str)
                                           or not check["public_ref"].startswith("prompt.")):
                raise PublicCodeCheckError("manifest public_ref is malformed")
            unknown = set(check["constraints"]) - _SUPPORTED_CONSTRAINTS - {"status", "field", "reason"}
            if unknown:
                raise PublicCodeCheckError(f"unsupported public constraint: {sorted(unknown)[0]}")
            has_fixture |= check["kind"] == "fixture"
            check_ids.add(check["id"])
        if not has_fixture and not any(c["kind"] == "unavailable" for c in checks):
            # A structural-only step must declare that its fixture is unavailable;
            # otherwise an omitted fixture could be mistaken for validation.
            step["checks"] = [*checks, {
                "id": f"{step_id}.fixture",
                "kind": "unavailable", "args": [], "kwargs": {},
                "constraints": {"field": "fixture", "status": "unavailable",
                                "reason": "No public fixture was defined."},
                "public_basis": "The public manifest defines no executable fixture.",
            }]
    return steps


def validate_manifest(manifest: dict) -> list[dict]:
    """Validate and return a private copy of the public manifest steps."""
    _canonical(manifest, maximum=MAX_MANIFEST_BYTES, label="manifest")
    return _validate_manifest(manifest)


def _validate_submission(submission: dict, step_ids: set[str]) -> tuple[dict, bytes, bytes]:
    if not isinstance(submission, dict) or not isinstance(submission.get("generated_code"), dict):
        raise PublicCodeCheckError("submission.generated_code must be an object")
    encoded = _canonical(submission, maximum=MAX_SUBMISSION_BYTES, label="submission")
    generated = submission["generated_code"]
    if len(generated) > MAX_STEPS or any(not isinstance(key, str) for key in generated):
        raise PublicCodeCheckError("submission.generated_code is too large or malformed")
    unknown = set(generated) - step_ids
    if unknown:
        raise PublicCodeCheckError(f"submission contains unknown step: {sorted(unknown)[0]}")
    for step_id, source in generated.items():
        if not isinstance(source, str) or not source or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise PublicCodeCheckError(f"generated code for {step_id} is malformed or too large")
    return generated, encoded, _canonical(generated, maximum=MAX_SUBMISSION_BYTES, label="generated_code")


def _child_program(payload: dict) -> str:
    payload_bytes = _canonical(payload, maximum=MAX_SUBMISSION_BYTES, label="child payload")
    encoded = base64.b64encode(payload_bytes).decode("ascii")
    # This source is executed only by the OS sandbox. Candidate code never runs
    # in the trusted controller process.
    program = r'''import base64, contextlib, copy, importlib, json, math, os, sys

PAYLOAD = json.loads(base64.b64decode(__PAYLOAD_BASE64__))
MAX_DEPTH = __MAX_DEPTH__
MAX_ELEMENTS = __MAX_ELEMENTS__

class FixtureUnavailable(Exception):
    pass

def _shape(value, depth=0):
    if depth > MAX_DEPTH:
        raise ValueError("value depth limit")
    if isinstance(value, (list, tuple)):
        if not value:
            return [0]
        child = [_shape(item, depth + 1) for item in value]
        if any(item != child[0] for item in child):
            return None
        return [len(value), *child[0]]
    return []

def _encode(value, depth=0, counter=None):
    if counter is None:
        counter = [0]
    if depth > MAX_DEPTH:
        raise ValueError("value depth limit")
    counter[0] += 1
    if counter[0] > MAX_ELEMENTS:
        raise ValueError("value element limit")
    if value is None:
        return {"kind": "none", "value": None, "shape": []}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value, "shape": []}
    if isinstance(value, int):
        return {"kind": "int", "value": value, "shape": []}
    if isinstance(value, float):
        return {"kind": "scalar", "value": value, "shape": []}
    module = type(value).__module__.split(".", 1)[0]
    if module == "numpy" and hasattr(value, "shape") and hasattr(value, "tolist"):
        if tuple(value.shape) == ():
            return _encode(value.item(), depth, counter)
        raw = value.tolist()
        return {"kind": "array", "numpy": True, "shape": [int(x) for x in value.shape],
                "value": _encode(raw, depth + 1, counter)["value"]}
    if isinstance(value, (list, tuple)):
        return {"kind": "sequence", "sequence_type": type(value).__name__,
                "shape": _shape(value), "value": [_encode(item, depth + 1, counter) for item in value]}
    if module == "numpy" and hasattr(value, "item"):
        return _encode(value.item(), depth, counter)
    return {"kind": "unsupported", "type": type(value).__name__, "shape": []}

def _resolve(value, namespace):
    if isinstance(value, dict) and set(value) == {"function_ref"}:
        name = value["function_ref"]
        try:
            return namespace[name]
        except KeyError as exc:
            raise FixtureUnavailable("unknown function_ref") from exc
    if isinstance(value, dict) and value.get("$type") == "numpy.ndarray":
        if set(value) != {"$type", "dtype", "value"} or value["dtype"] != "float64":
            raise FixtureUnavailable("malformed public ndarray argument")
        try:
            import numpy as np
            return np.asarray(_resolve(value["value"], namespace), dtype=value["dtype"])
        except (ImportError, TypeError, ValueError) as exc:
            raise FixtureUnavailable("public ndarray argument unavailable") from exc
    if isinstance(value, list):
        return [_resolve(item, namespace) for item in value]
    if isinstance(value, dict):
        return {key: _resolve(item, namespace) for key, item in value.items()}
    return value

def _run():
    namespace = {"__name__": "__candidate__"}
    dependency_error = None
    for spec in PAYLOAD.get("required_imports", []):
        try:
            module = importlib.import_module(spec["module"])
            if spec["kind"] == "import":
                if "." in spec["module"] and spec["alias"] == spec["module"].split(".", 1)[0]:
                    namespace[spec["alias"]] = importlib.import_module(spec["module"].split(".", 1)[0])
                else:
                    namespace[spec["alias"]] = module
            else:
                for item in spec["names"]:
                    namespace[item["alias"] or item["name"]] = getattr(module, item["name"])
        except Exception as exc:
            dependency_error = spec["module"] + ": " + type(exc).__name__
            break
    errors = {}
    compile_errors = {}
    compiled = {}
    with open(os.devnull, "w", encoding="utf-8") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        for step in PAYLOAD["steps"]:
            step_id = step["step_id"]
            source = PAYLOAD["generated_code"].get(step_id)
            if source is None:
                compile_errors[step_id] = "missing generated code"
                continue
            try:
                compiled[step_id] = compile(source, "<candidate:" + step_id + ">", "exec")
            except BaseException as exc:
                compile_errors[step_id] = type(exc).__name__ + ": " + str(exc)[:300]
        for step in PAYLOAD["steps"]:
            step_id = step["step_id"]
            if step_id in compile_errors or dependency_error:
                continue
            try:
                exec(compiled[step_id], namespace, namespace)
            except BaseException as exc:
                errors[step_id] = type(exc).__name__ + ": " + str(exc)[:300]
        results = {}
        for step in PAYLOAD["steps"]:
            step_id = step["step_id"]
            function_name = step["function_name"]
            for check in step["checks"]:
                if check["kind"] != "fixture":
                    continue
                check_id = check["id"]
                if step_id in compile_errors:
                    results[check_id] = {"status": "failed", "message": compile_errors[step_id]}
                    continue
                if dependency_error:
                    results[check_id] = {"status": "unavailable", "message": "declared dependency unavailable: " + dependency_error}
                    continue
                if step_id in errors:
                    results[check_id] = {"status": "failed", "message": errors[step_id]}
                    continue
                function = namespace.get(function_name)
                if not callable(function):
                    results[check_id] = {"status": "failed", "message": "function is not defined"}
                    continue
                try:
                    args = _resolve(copy.deepcopy(check["args"]), namespace)
                    kwargs = _resolve(copy.deepcopy(check["kwargs"]), namespace)
                    value = function(*args, **kwargs)
                    results[check_id] = {"status": "returned", "value": _encode(value)}
                except FixtureUnavailable as exc:
                    results[check_id] = {"status": "unavailable", "message": str(exc)}
                except BaseException as exc:
                    results[check_id] = {"status": "failed",
                        "message": type(exc).__name__ + ": " + str(exc)[:300]}
    return {"errors": errors, "compile_errors": compile_errors,
            "dependency_error": dependency_error, "results": results}

try:
    print(json.dumps(_run(), allow_nan=False, separators=(",", ":")))
except BaseException as exc:
    print(json.dumps({"error": type(exc).__name__ + ": " + str(exc)[:300]}, separators=(",", ":")))
'''
    return (program.replace("__PAYLOAD_BASE64__", repr(encoded))
            .replace("__MAX_DEPTH__", str(MAX_JSON_DEPTH))
            .replace("__MAX_ELEMENTS__", str(MAX_RESULT_ELEMENTS)))

def _shape_of(value: Any, depth=0):
    if depth > MAX_JSON_DEPTH:
        return None
    if isinstance(value, dict) and value.get("$type") == "numpy.ndarray":
        return _shape_of(value.get("value"), depth + 1)
    if isinstance(value, list):
        if not value:
            return [0]
        child = [_shape_of(item, depth + 1) for item in value]
        return [len(value), *child[0]] if child and all(item == child[0] for item in child) else None
    return []


def _numbers(descriptor: dict):
    kind = descriptor.get("kind")
    if kind in {"int", "scalar"}:
        value = descriptor.get("value")
        return [value] if type(value) in {int, float} else None
    if kind == "sequence":
        values = []
        for item in descriptor.get("value", []):
            child = _numbers(item)
            if child is None:
                return None
            values.extend(child)
        return values
    if kind == "array":
        return _numbers({"kind": "sequence", "value": descriptor.get("value", [])})
    return None


def _numeric_tree(value: Any):
    """Decode bounded child descriptors into nested finite numeric lists."""
    if isinstance(value, list):
        decoded = [_numeric_tree(item) for item in value]
        return None if any(item is None for item in decoded) else decoded
    if not isinstance(value, dict):
        return None
    kind = value.get("kind")
    if kind in {"int", "scalar"}:
        number = value.get("value")
        if type(number) not in {int, float}:
            return None
        try:
            return number if math.isfinite(number) else None
        except (OverflowError, TypeError, ValueError):
            return None
    if kind in {"array", "sequence"}:
        return _numeric_tree(value.get("value"))
    return None


def _rectangular_numeric_matrix(descriptor: dict):
    if descriptor.get("kind") not in {"array", "sequence"}:
        return None
    rows = _numeric_tree(descriptor.get("value"))
    if (not isinstance(rows, list) or not rows or not all(isinstance(row, list) for row in rows)
            or not rows[0] or any(len(row) != len(rows[0]) for row in rows)):
        return None
    return rows


def _finite(descriptor: dict) -> bool:
    values = _numbers(descriptor)
    if values is None:
        return False
    try:
        return all(math.isfinite(value) for value in values)
    except (OverflowError, TypeError, ValueError):
        return False


def _check_value(descriptor: dict, constraints: dict, args: list, tolerance: dict | None = None) -> tuple[str, str]:
    tolerance = tolerance or {"atol": 1e-8, "rtol": 1e-6}
    try:
        atol = float(tolerance.get("atol", 1e-8))
        rtol = float(tolerance.get("rtol", 1e-6))
    except (TypeError, ValueError):
        return "unavailable", "numeric tolerance is unavailable"
    kind = descriptor.get("kind")
    expected = constraints.get("return_type")
    if expected == "scalar" and kind not in {"int", "scalar"}:
        return "failed", "return_type must be scalar"
    if expected == "numpy.ndarray" and kind != "array":
        return "failed", "return_type must be numpy.ndarray"
    if expected == "array_like" and kind not in {"array", "sequence"}:
        return "failed", "return_type must be array_like"
    if "shape" in constraints and descriptor.get("shape") != constraints["shape"]:
        return "failed", f"shape {descriptor.get('shape')} does not match {constraints['shape']}"
    if constraints.get("finite") or constraints.get("finite_elements"):
        if not _finite(descriptor):
            return "failed", "return contains non-finite or non-numeric values"
    if "same_shape_as_arg" in constraints:
        index = constraints["same_shape_as_arg"]
        if type(index) is not int or index < 0 or index >= len(args):
            return "unavailable", "shape reference argument is undefined"
        if descriptor.get("shape") != _shape_of(args[index]):
            return "failed", "return shape differs from the referenced argument"
    if "return_arity" in constraints:
        if kind != "sequence" or len(descriptor.get("value", [])) != constraints["return_arity"]:
            return "failed", "return arity does not match the public contract"
    for index, item_constraints in enumerate(constraints.get("items", [])):
        values = descriptor.get("value", [])
        if index >= len(values):
            return "failed", "return item is missing"
        status, message = _check_value(values[index], item_constraints, args, tolerance)
        if status != "passed":
            return status, f"item {index}: {message}"
    values = _numbers(descriptor)
    if constraints.get("l2_norm") == "approximately_one":
        try:
            norm = math.sqrt(sum(v * v for v in values)) if values is not None else math.nan
            norm_matches = math.isclose(norm, 1.0, rel_tol=rtol, abs_tol=atol)
        except (OverflowError, TypeError, ValueError):
            norm_matches = False
        if not norm_matches:
            return "failed", "L2 norm is not approximately one"
    if "rank" in constraints and len(descriptor.get("shape", [])) != constraints["rank"]:
        return "failed", "rank does not match the public contract"
    if "axis_1_size" in constraints:
        shape = descriptor.get("shape", [])
        if len(shape) < 2 or shape[1] != constraints["axis_1_size"]:
            return "failed", "axis-1 size does not match the public contract"
    matrix = _rectangular_numeric_matrix(descriptor)
    if constraints.get("net_force_sum_zero"):
        if matrix is None:
            return "failed", "net-force check requires a numeric matrix"
        try:
            sums = [sum(float(row[col]) for row in matrix) for col in range(len(matrix[0]))]
        except (IndexError, TypeError, ValueError, OverflowError):
            return "failed", "net-force check requires a rectangular numeric matrix"
        if any(not math.isclose(value, 0.0, rel_tol=rtol, abs_tol=atol) for value in sums):
            return "failed", "net force sum is not approximately zero"
    if constraints.get("column_unit_norm") or constraints.get("column_pairwise_orthogonal"):
        if matrix is None:
            return "failed", "column checks require a numeric matrix"
        try:
            columns = list(zip(*matrix))
            columns = [[float(item) for item in column] for column in columns]
        except (IndexError, TypeError, ValueError, OverflowError):
            return "failed", "column checks require a rectangular numeric matrix"
        try:
            unit_norm_bad = constraints.get("column_unit_norm") and any(
                not math.isclose(math.sqrt(sum(x * x for x in column)), 1.0,
                                 rel_tol=rtol, abs_tol=atol)
                for column in columns
            )
        except (OverflowError, TypeError, ValueError):
            unit_norm_bad = True
        if unit_norm_bad:
            return "failed", "column norm is not approximately one"
        try:
            nonorthogonal = constraints.get("column_pairwise_orthogonal") and any(
                not math.isclose(sum(a * b for a, b in zip(columns[i], columns[j])), 0.0,
                                 rel_tol=rtol, abs_tol=atol)
                for i in range(len(columns)) for j in range(i + 1, len(columns))
            )
        except (OverflowError, TypeError, ValueError):
            nonorthogonal = True
        if nonorthogonal:
            return "failed", "columns are not pairwise orthogonal"
    return "passed", "host public contract check passed"


def _signature_matches(source: str | None, expected: str, function_name: str) -> bool:
    if not source:
        return False
    try:
        actual_tree = ast.parse(source)
        expected_tree = ast.parse(expected + "\n    pass\n")
        expected_node = next(node for node in ast.walk(expected_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
        all_nodes = [node for node in ast.walk(actual_tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == function_name]
        actual_nodes = [node for node in actual_tree.body
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == function_name]
        if len(all_nodes) != 1 or len(actual_nodes) != 1 or expected_node.name != function_name:
            return False
        actual_node = actual_nodes[0]
        if type(actual_node) is not type(expected_node):
            return False
        return ast.dump(actual_node.args, include_attributes=False) == ast.dump(
            expected_node.args, include_attributes=False)
    except (SyntaxError, StopIteration):
        return False


def _result_base(status, *, submission_sha256, generated_code_sha256, manifest_sha256,
                 checker_source_sha256, revision, sandbox_backend=None):
    result = {
        "schema_version": PUBLIC_RESULT_VERSION,
        "status": status,
        "host_generated": True,
        "official_correctness": "not_evaluated",
        "semantic_validation": "not_performed",
        "revision": revision,
        "submission_sha256": submission_sha256,
        "generated_code_sha256": generated_code_sha256,
        "manifest_sha256": manifest_sha256,
        "checker_source_sha256": checker_source_sha256,
        "checks": [],
        "unavailable_assertions": [],
        "coverage": {"executable_checks": 0, "passed": 0, "failed": 0,
                      "unavailable": 0, "declared_unavailable": 0},
    }
    if sandbox_backend:
        result["sandbox_backend"] = sandbox_backend
    return result


def _check_record(check: dict, status: str, message: str) -> dict:
    record = {"id": check["id"], "kind": check["kind"], "status": status, "message": message}
    if "public_ref" in check:
        record["public_ref"] = check["public_ref"]
    return record


def _validate_child_protocol(value: Any) -> dict:
    """Validate the bounded launcher envelope before any result lookup."""
    if not isinstance(value, dict):
        raise PublicCodeCheckError("checker child result must be an object")
    if "error" in value:
        if set(value) != {"error"} or not isinstance(value["error"], str):
            raise PublicCodeCheckError("checker child error envelope is malformed")
        return value
    if set(value) != {"errors", "compile_errors", "dependency_error", "results"}:
        raise PublicCodeCheckError("checker child result keys are malformed")
    if (not isinstance(value["errors"], dict) or not isinstance(value["compile_errors"], dict)
            or not isinstance(value["results"], dict)
            or (value["dependency_error"] is not None and not isinstance(value["dependency_error"], str))):
        raise PublicCodeCheckError("checker child errors/results/dependency status are malformed")
    for error_map in (value["errors"], value["compile_errors"]):
        for key, message in error_map.items():
            if not isinstance(key, str) or not isinstance(message, str):
                raise PublicCodeCheckError("checker child error entry is malformed")
    for key, item in value["results"].items():
        if not isinstance(key, str) or not isinstance(item, dict):
            raise PublicCodeCheckError("checker child fixture entry is malformed")
        status = item.get("status")
        if status == "returned":
            if set(item) != {"status", "value"} or not isinstance(item["value"], dict):
                raise PublicCodeCheckError("checker child returned value is malformed")
        elif status in {"failed", "unavailable"}:
            if set(item) - {"status", "message"} or not isinstance(item.get("message"), str):
                raise PublicCodeCheckError("checker child fixture diagnostic is malformed")
        else:
            raise PublicCodeCheckError("checker child fixture status is malformed")
    return value


def check_public_submission(submission: dict, manifest: dict, *, sandbox=None,
                            checker_source_sha256: str | None = None,
                            revision: str = "D0", timeout_seconds: float = 10.0,
                            workspace: os.PathLike[str] | str | None = None,
                            cancel_check=None, _submission_identity: dict | None = None) -> dict:
    """Run public structural/fixture checks and return a bounded host summary."""
    manifest_bytes = _canonical(manifest, maximum=MAX_MANIFEST_BYTES, label="manifest")
    steps = _validate_manifest(manifest)
    step_ids = {step["step_id"] for step in steps}
    generated, submission_bytes, generated_bytes = _validate_submission(submission, step_ids)
    if not isinstance(revision, str) or not revision or len(revision) > 64:
        raise PublicCodeCheckError("revision is malformed")
    if not 0 < timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise PublicCodeCheckError("timeout exceeds the public checker bound")
    actual_checker_sha = checker_source_digest()
    checker_sha = checker_source_sha256 or actual_checker_sha
    if (not isinstance(checker_sha, str) or not HASH_RE.fullmatch(checker_sha)
            or checker_sha != actual_checker_sha):
        raise PublicCodeCheckError("checker_source_sha256 mismatch or malformed")
    base = _result_base("unavailable", submission_sha256=_sha(submission_bytes),
                        generated_code_sha256=_sha(generated_bytes), manifest_sha256=_sha(manifest_bytes),
                        checker_source_sha256=checker_sha, revision=revision)
    if _submission_identity is not None:
        base["submission_sha256"] = submission_sha256(_submission_identity)
    for step in steps:
        source = generated.get(step["step_id"])
        for check in step["checks"]:
            if check["kind"] == "unavailable":
                base["checks"].append(_check_record(
                    check, "unavailable", check["constraints"].get("reason", "public assertion unavailable")))
                base["unavailable_assertions"].append(check["id"])
            elif check["kind"] == "signature":
                expected = check["constraints"].get("exact_first_line", step["signature"])
                passed = _signature_matches(source, expected, step["function_name"])
                base["checks"].append(_check_record(
                    check, "passed" if passed else "failed",
                    "signature matches" if passed else "signature first line mismatch"))
    required_imports = _validate_required_imports(manifest.get("required_imports"))
    payload = {"generated_code": generated, "steps": steps,
               "required_imports": required_imports}
    sandbox = sandbox or select_sandbox()
    if workspace is None:
        workspace_context = tempfile.TemporaryDirectory(prefix="adj-public-code-check-")
    else:
        workspace_path = Path(workspace)
        if workspace_path.is_symlink() or workspace_path.resolve() != workspace_path:
            raise PublicCodeCheckError("checker workspace cannot be a symlink")
        workspace_path.mkdir(parents=True, exist_ok=True)
        workspace_context = nullcontext(workspace_path)
    with workspace_context as temporary:
        workspace = Path(temporary).resolve()
        for name in ("inputs", "outputs", "tmp"):
            child = workspace / name
            if child.is_symlink():
                raise PublicCodeCheckError("checker workspace directory cannot be a symlink")
            child.mkdir(exist_ok=True)
        program = workspace / "tmp" / "public_code_checker_child.py"
        program.write_text(_child_program(payload), encoding="utf-8")
        try:
            execution = sandbox.run([os.fspath(Path(os.sys.executable)), "-I", str(program)], workspace,
                                    timeout_seconds=timeout_seconds, cancel_check=cancel_check)
        except SandboxUnavailable:
            raise
        except ToolCancelled:
            raise
        except Exception as exc:
            execution = None
            base["environment_error"] = type(exc).__name__ + ": " + str(exc)[:300]
        if execution is not None:
            base["sandbox_backend"] = getattr(execution, "backend", "unknown")
        child = None
        child_protocol_error = False
        if execution is not None and not execution.timed_out and not execution.limit_reason:
            try:
                parsed_child = json.loads(execution.stdout)
                _canonical(parsed_child, maximum=MAX_RESULT_BYTES, label="checker child result")
                child = _validate_child_protocol(parsed_child)
            except (ValueError, TypeError, PublicCodeCheckError):
                child = None
                child_protocol_error = True
        environment_unavailable = (execution is None or execution.timed_out
                                    or bool(execution.limit_reason) or child_protocol_error)
        child_failure = ((child or {}).get("error") if isinstance(child, dict) else None)
        if child_failure is None and child is None and execution is not None and not environment_unavailable:
            child_failure = "sandbox child returned no valid finite JSON"
        for step in steps:
            source = generated.get(step["step_id"])
            for check in step["checks"]:
                if check["kind"] == "compile":
                    error = (child or {}).get("compile_errors", {}).get(step["step_id"])
                    passed = source is not None and not error and child_failure is None and not environment_unavailable
                    status = "unavailable" if environment_unavailable else "passed" if passed else "failed"
                    base["checks"].append(_check_record(
                        check, status,
                        ("source compiled" if passed else
                         "sandbox execution unavailable" if environment_unavailable else
                         (error or child_failure or "source missing or child unavailable"))))
                elif check["kind"] == "fixture":
                    results = child.get("results", {}) if isinstance(child, dict) else {}
                    item = results.get(check["id"]) if isinstance(results, dict) else None
                    if environment_unavailable:
                        status, message = "unavailable", "sandbox execution unavailable"
                    elif child_failure:
                        status, message = "failed", child_failure
                    elif isinstance(child, dict) and child.get("dependency_error"):
                        status, message = "unavailable", "declared dependency unavailable"
                    elif not item:
                        status, message = "unavailable", "fixture result unavailable"
                    elif item.get("status") == "unavailable":
                        status, message = "unavailable", item.get("message", "fixture unavailable")
                    elif item.get("status") != "returned" or not isinstance(item.get("value"), dict):
                        status, message = "failed", item.get("message", "fixture execution failed")
                    else:
                        status, message = _check_value(item["value"], check["constraints"], check["args"], manifest.get("numeric_tolerance"))
                    base["checks"].append(_check_record(check, status, message))
    statuses = [item["status"] for item in base["checks"]]
    declared = set(base["unavailable_assertions"])
    executable_statuses = [item["status"] for item in base["checks"] if item["id"] not in declared]
    base["coverage"] = {
        "executable_checks": len(executable_statuses),
        "passed": executable_statuses.count("passed"),
        "failed": executable_statuses.count("failed"),
        "unavailable": executable_statuses.count("unavailable"),
        "declared_unavailable": len(declared),
    }
    base["status"] = ("unavailable" if "unavailable" in executable_statuses or not executable_statuses else
                       "failed" if "failed" in executable_statuses else
                       "passed")
    output = _canonical(base, maximum=MAX_RESULT_BYTES, label="checker result")
    return json.loads(output)


def canonical_submission(submission: dict) -> bytes:
    """Return the bounded canonical bytes used for the submission identity."""
    return _canonical(submission, maximum=MAX_SUBMISSION_BYTES, label="submission")


def submission_sha256(submission: dict) -> str:
    return _sha(canonical_submission(submission))


def canonical_manifest(manifest: dict) -> bytes:
    return _canonical(manifest, maximum=MAX_MANIFEST_BYTES, label="manifest")


def manifest_sha256(manifest: dict) -> str:
    return _sha(canonical_manifest(manifest))


def run_public_code_checks(submission: dict, manifest: dict, workspace: os.PathLike[str] | str,
                           *, sandbox=None, checker_source_sha256: str | None = None,
                           revision: str = "D0", timeout_seconds: float = 10.0,
                           cancel_check=None) -> dict:
    """Run the public checker in a caller-owned isolated sandbox workspace.

    ``submission`` is normally the generated-code mapping.  For compatibility
    with the immutable method wrapper, an envelope containing
    ``generated_code`` is accepted as well.  No controller-side execution of
    candidate source occurs; ``workspace`` is passed to the existing sandbox.
    """
    envelope = submission if isinstance(submission, dict) and "generated_code" in submission else {
        "generated_code": submission,
    }
    identity = submission if isinstance(submission, dict) and "generated_code" not in submission else None
    return check_public_submission(envelope, manifest, sandbox=sandbox,
                                   checker_source_sha256=checker_source_sha256,
                                   revision=revision, timeout_seconds=timeout_seconds,
                                   workspace=workspace, cancel_check=cancel_check,
                                   _submission_identity=identity)


def run_public_code_checker(parameters_path: os.PathLike[str] | str, *, sandbox=None,
                            checker_source_sha256: str | None = None,
                            revision: str = "D0", timeout_seconds: float = 10.0) -> dict:
    """Run the checker from an immutable method workspace layout.

    ``parameters_path`` is the method's ``parameters.json``; the public
    manifest is always its sibling ``inputs/manifest.json``.  Files are read
    with strict size limits and no raw path is included in the result.
    """
    parameters_path = Path(parameters_path)
    if parameters_path.stat().st_size > MAX_SUBMISSION_BYTES:
        raise PublicCodeCheckError("parameters exceeds the public checker byte limit")
    parameters = json.loads(parameters_path.read_text(encoding="utf-8"))
    manifest_path = Path(parameters_path).parent / "inputs" / "manifest.json"
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise PublicCodeCheckError("manifest exceeds the public checker byte limit")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    submission = parameters.get("submission") if isinstance(parameters, dict) else None
    return check_public_submission(submission, manifest, sandbox=sandbox,
                                   checker_source_sha256=checker_source_sha256,
                                   revision=revision, timeout_seconds=timeout_seconds)

