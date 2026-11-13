"""Strict structural acceptance, separate from semantic scientific assurance."""

import csv
import hashlib
import io
import json
import math
import re
from decimal import Decimal

from .validation import bounded_json, digest, strings
from .coverage import evidence_coverage
from ..tools.catalog import (canonical_json, validate_schema as validate_host_schema,
    validator as host_validator)


COVERAGE_REQUIREMENTS = frozenset({"min_distinct_sources", "min_distinct_evidence"})
COUNT_REQUIREMENTS = COVERAGE_REQUIREMENTS | {"paragraphs_per_section"}
RESPONSE_SCHEMA = "response_json_schema"
EXECUTION_BOUND_DELIVERY = "execution_bound_delivery"
REQUIRE_BOUND_EXECUTION_WITNESS = "require_bound_execution_witness"
REQUIRE_EXECUTION_ACTION = "require_execution_action"
PUBLIC_CODE_VALIDATION = "public_code_validation"
PUBLIC_REFERENCE_FEEDBACK = "public_reference_feedback"
PUBLIC_CODE_CHECKER_VERSION = "public-code-check-v1"
REQUIREMENTS = frozenset({"require_concise_discovery", "require_critic", "require_literal_observations", "require_research_design", "require_report", "require_evidence", "require_limitations", "require_data_artifact", "required_sections", "report_layout", RESPONSE_SCHEMA, EXECUTION_BOUND_DELIVERY, REQUIRE_BOUND_EXECUTION_WITNESS, REQUIRE_EXECUTION_ACTION, PUBLIC_CODE_VALIDATION, PUBLIC_REFERENCE_FEEDBACK}) | COUNT_REQUIREMENTS
RESPONSE_REPAIR_MAX_BYTES = 64 * 1024
RESPONSE_REPAIR_MAX_PARSE_ATTEMPTS = 32
RESPONSE_JSON_MAX_DEPTH = 128
RESPONSE_JSON_MAX_NODES = 10_000


class ReportNotAccepted(ValueError):
    def __init__(self, issues):
        self.issues = list(dict.fromkeys(issues))
        super().__init__("Report not accepted: " + "; ".join(self.issues))


def task_binding(task):
    return {"id": task["id"], "requirements_hash": digest({"goal": task["goal"], "acceptance": task["acceptance"]}),
            "observed_revision": task["revision"]}


def requirements(value):
    if not isinstance(value, dict) or set(value) - REQUIREMENTS:
        raise ValueError("Unknown or non-object acceptance requirements")
    bounded_json(value)
    if PUBLIC_REFERENCE_FEEDBACK in value:
        _validate_reference_contract(value[PUBLIC_REFERENCE_FEEDBACK])
        if set(value) != {PUBLIC_REFERENCE_FEEDBACK}:
            raise ValueError(f"{PUBLIC_REFERENCE_FEEDBACK} must be standalone in v1")
    if RESPONSE_SCHEMA in value:
        schema = value[RESPONSE_SCHEMA]
        if not isinstance(schema, dict):
            raise ValueError(f"{RESPONSE_SCHEMA} must be a JSON Schema object")
        if value.get(EXECUTION_BOUND_DELIVERY) is True:
            schema_type = schema.get("type")
            if schema_type != "object" and not (isinstance(schema_type, list) and "object" in schema_type):
                raise ValueError(f"{RESPONSE_SCHEMA} must accept an object summary for {EXECUTION_BOUND_DELIVERY}")
        for node in _schema_nodes(schema):
            for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                reference = node.get(key)
                if reference is not None and (not isinstance(reference, str) or not reference.startswith("#")):
                    raise ValueError(f"{RESPONSE_SCHEMA} may use only local references")
        try:
            validate_host_schema(schema)
            schema_validator = host_validator(schema)
            for node in _schema_nodes(schema):
                for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                    reference = node.get(key)
                    if reference is None:
                        continue
                    try:
                        schema_validator._resolver.lookup(reference)
                    except Exception as exc:
                        raise ValueError(f"{RESPONSE_SCHEMA} reference is unresolved: {reference}") from exc
        except Exception as exc:
            raise ValueError(f"Invalid {RESPONSE_SCHEMA}: {exc}") from exc
    if PUBLIC_CODE_VALIDATION in value:
        _validate_public_code_contract(value[PUBLIC_CODE_VALIDATION])
        if RESPONSE_SCHEMA not in value:
            raise ValueError(f"{PUBLIC_CODE_VALIDATION} requires {RESPONSE_SCHEMA}")
        schema = value[RESPONSE_SCHEMA]
        schema_type = schema.get("type")
        if not (schema_type == "object" or isinstance(schema_type, list) and "object" in schema_type):
            raise ValueError(f"{PUBLIC_CODE_VALIDATION} requires an object {RESPONSE_SCHEMA}")
        properties = schema.get("properties")
        generated_code = properties.get("generated_code") if isinstance(properties, dict) else None
        code_type = generated_code.get("type") if isinstance(generated_code, dict) else None
        required_fields = schema.get("required", [])
        if (not isinstance(generated_code, dict)
                or not (code_type == "object" or isinstance(code_type, list) and "object" in code_type)
                or not isinstance(required_fields, list) or "generated_code" not in required_fields):
            raise ValueError(
                f"{PUBLIC_CODE_VALIDATION} requires {RESPONSE_SCHEMA} to require generated_code as an object mapping")
        if set(value) != {RESPONSE_SCHEMA, PUBLIC_CODE_VALIDATION}:
            raise ValueError(f"{PUBLIC_CODE_VALIDATION} cannot be combined with other acceptance modes in v1")
    for key in REQUIREMENTS - {"required_sections", "report_layout", RESPONSE_SCHEMA, PUBLIC_CODE_VALIDATION, PUBLIC_REFERENCE_FEEDBACK} - COUNT_REQUIREMENTS:
        if key in value and type(value[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if value.get(EXECUTION_BOUND_DELIVERY) is True:
        if RESPONSE_SCHEMA not in value:
            raise ValueError(f"{EXECUTION_BOUND_DELIVERY} requires {RESPONSE_SCHEMA}")
        if set(value) - {RESPONSE_SCHEMA, EXECUTION_BOUND_DELIVERY,
                         REQUIRE_BOUND_EXECUTION_WITNESS, REQUIRE_EXECUTION_ACTION}:
            raise ValueError(f"{EXECUTION_BOUND_DELIVERY} cannot be combined with research/report acceptance")
    if value.get(REQUIRE_BOUND_EXECUTION_WITNESS) is True:
        if RESPONSE_SCHEMA not in value:
            raise ValueError(f"{REQUIRE_BOUND_EXECUTION_WITNESS} requires {RESPONSE_SCHEMA}")
        if set(value) - {RESPONSE_SCHEMA, REQUIRE_BOUND_EXECUTION_WITNESS,
                         REQUIRE_EXECUTION_ACTION, EXECUTION_BOUND_DELIVERY}:
            raise ValueError(f"{REQUIRE_BOUND_EXECUTION_WITNESS} cannot be combined with research/report acceptance")
    if value.get(REQUIRE_EXECUTION_ACTION) is True:
        if value.get(REQUIRE_BOUND_EXECUTION_WITNESS) is not True:
            raise ValueError(f"{REQUIRE_EXECUTION_ACTION} requires {REQUIRE_BOUND_EXECUTION_WITNESS}")
        if RESPONSE_SCHEMA not in value:
            raise ValueError(f"{REQUIRE_EXECUTION_ACTION} requires {RESPONSE_SCHEMA}")
    for key in COUNT_REQUIREMENTS:
        if key in value and (type(value[key]) is not int or not 1 <= value[key] <= 100):
            raise ValueError(f"{key} must be an integer from 1 to 100")
    if "report_layout" in value and (not isinstance(value["report_layout"], str)
                                    or value["report_layout"] not in {"audit", "review"}):
        raise ValueError("report_layout must be audit or review")
    sections = value.get("required_sections", [])
    strings(sections, "required_sections", maximum=50, item_max=200, unique=True)
    if len({heading.strip().casefold() for heading in sections}) != len(sections):
        raise ValueError("Required sections must be unique")
    contract_fields = set(value) - {EXECUTION_BOUND_DELIVERY, REQUIRE_BOUND_EXECUTION_WITNESS,
                                    REQUIRE_EXECUTION_ACTION}
    contract_only = bool(contract_fields) and contract_fields <= {RESPONSE_SCHEMA, PUBLIC_CODE_VALIDATION, PUBLIC_REFERENCE_FEEDBACK}
    return {"require_report": not contract_only, "require_evidence": False, "require_limitations": False,
            "require_data_artifact": False, "required_sections": [], **value}


def _schema_nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _schema_nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _schema_nodes(child)


def _validate_public_code_contract(value):
    expected = {"manifest_artifact_id", "manifest_sha256", "checker_version", "max_attempts", "enforce"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{PUBLIC_CODE_VALIDATION} must contain exactly the supported contract fields")
    identifier = value["manifest_artifact_id"]
    if (not isinstance(identifier, str) or len(identifier) > 128
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", identifier)):
        raise ValueError(f"{PUBLIC_CODE_VALIDATION}.manifest_artifact_id is invalid")
    if (not isinstance(value["manifest_sha256"], str)
            or not re.fullmatch(r"[a-f0-9]{64}", value["manifest_sha256"])):
        raise ValueError(f"{PUBLIC_CODE_VALIDATION}.manifest_sha256 must be lowercase SHA-256")
    if value["checker_version"] != PUBLIC_CODE_CHECKER_VERSION:
        raise ValueError(f"{PUBLIC_CODE_VALIDATION}.checker_version is unsupported")
    if type(value["max_attempts"]) is not int or value["max_attempts"] != 2:
        raise ValueError(f"{PUBLIC_CODE_VALIDATION}.max_attempts must be exactly 2")
    if type(value["enforce"]) is not bool:
        raise ValueError(f"{PUBLIC_CODE_VALIDATION}.enforce must be boolean")


def _validate_reference_contract(value):
    expected = {"corpus_artifact_id", "corpus_sha256", "task_type", "versions", "arm"}
    if (not isinstance(value, dict)
            or set(value) not in (expected, expected | {"comparison_protocol"})):
        raise ValueError(f"{PUBLIC_REFERENCE_FEEDBACK} must contain exactly the supported contract fields")
    controlled = "comparison_protocol" in value
    if controlled and value["comparison_protocol"] != "content_control_v1":
        raise ValueError("Unsupported reference comparison protocol")
    if (not isinstance(value["corpus_artifact_id"], str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value["corpus_artifact_id"])):
        raise ValueError("Invalid reference corpus artifact identifier")
    if not isinstance(value["corpus_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", value["corpus_sha256"]):
        raise ValueError("Reference corpus hash must be lowercase SHA-256")
    if value["task_type"] not in ("numerical", "literature", "protocol"):
        raise ValueError("Unsupported reference task type")
    allowed_arms = ("plain_revision", "reference_feedback", "neutral_control") if controlled else ("plain_revision", "reference_feedback")
    if value["arm"] not in allowed_arms:
        raise ValueError("Unsupported reference revision arm")
    versions = value["versions"]
    if (not isinstance(versions, dict) or len(versions) > 32
            or any(not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key)
                   or not isinstance(version, str) or not version.strip()
                   or len(version.encode("utf-8")) > 64
                   for key, version in versions.items())):
        raise ValueError("Reference versions must map library roots to bounded nonempty versions")


def research_acceptance(value):
    """Whether acceptance invokes the evidence-backed research-report workflow."""
    return bool(set(value) - {RESPONSE_SCHEMA, EXECUTION_BOUND_DELIVERY,
                              REQUIRE_BOUND_EXECUTION_WITNESS, REQUIRE_EXECUTION_ACTION,
                              PUBLIC_CODE_VALIDATION, PUBLIC_REFERENCE_FEEDBACK}) if isinstance(value, dict) else bool(value)


def _reject_json_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _response_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def _response_decimal(value):
    number = Decimal(value)
    if not number.is_finite():
        raise ValueError("nonfinite JSON number")
    return number


def _response_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_response_json(content, *, exact_numbers=False):
    return json.loads(content, object_pairs_hook=_response_object,
                      parse_constant=_reject_json_constant,
                      parse_float=_response_decimal if exact_numbers else _response_float)


def _response_number_token(value):
    number = value if isinstance(value, Decimal) else Decimal(str(value))
    sign, digits, exponent = number.as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    if not any(digits):
        sign, digits, exponent = 0, [0], 0
    return ["number", sign, digits, exponent]


def _response_value_signature(value):
    if isinstance(value, dict):
        return ["object", [[key, _response_value_signature(item)] for key, item in sorted(value.items())]]
    if isinstance(value, list):
        return ["array", [_response_value_signature(item) for item in value]]
    if isinstance(value, bool):
        return ["boolean", value]
    if value is None:
        return ["null"]
    if isinstance(value, (Decimal, int, float)):
        return _response_number_token(value)
    return ["string", value]


def _response_value_within_limits(value):
    """Bound traversal before recursive canonicalization or schema validation."""
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > RESPONSE_JSON_MAX_DEPTH or nodes > RESPONSE_JSON_MAX_NODES:
            return False
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    return True


def response_repair_seed(content):
    """Return a strict JSON value or one unambiguous embedded JSON object."""
    if len(content) > RESPONSE_REPAIR_MAX_BYTES:
        return False, None
    try:
        if len(content.encode("utf-8")) > RESPONSE_REPAIR_MAX_BYTES:
            return False, None
    except UnicodeError:
        return False, None
    try:
        value = _strict_response_json(content, exact_numbers=True)
        if _response_value_within_limits(value):
            return True, value
        return False, None
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
        pass

    decoder = json.JSONDecoder(object_pairs_hook=_response_object,
                               parse_constant=_reject_json_constant, parse_float=_response_decimal)
    candidates = {}
    offset = 0
    attempts = 0
    exhausted = False
    while True:
        start = content.find("{", offset)
        if start < 0:
            break
        if attempts >= RESPONSE_REPAIR_MAX_PARSE_ATTEMPTS:
            exhausted = True
            break
        attempts += 1
        try:
            value, end = decoder.raw_decode(content, start)
        except (json.JSONDecodeError, TypeError, ValueError, RecursionError):
            offset = start + 1
            continue
        if isinstance(value, dict):
            if not _response_value_within_limits(value):
                return False, None
            candidates.setdefault(canonical_json(_response_value_signature(value)), value)
        offset = max(end, start + 1)
    if exhausted:
        return False, None
    if len(candidates) == 1:
        return True, next(iter(candidates.values()))
    return False, None


def response_scalar_fingerprint(value):
    """Hash typed scalar values bound to their original JSON object/array paths."""
    if not _response_value_within_limits(value):
        raise ValueError("Response value exceeds scalar path binding limits")
    leaves = []
    pending = [(value, ())]
    while pending:
        item, path = pending.pop()
        if isinstance(item, dict):
            pending.extend((child, path + (("key", key),))
                           for key, child in item.items())
        elif isinstance(item, list):
            pending.extend((child, path + (("index", index),))
                           for index, child in enumerate(item))
        elif item is None:
            leaves.append([[list(part) for part in path], ["null"]])
        elif isinstance(item, bool):
            leaves.append([[list(part) for part in path], ["boolean", item]])
        elif isinstance(item, (Decimal, int, float)):
            leaves.append([[list(part) for part in path], _response_number_token(item)])
        else:
            leaves.append([[list(part) for part in path], ["string", item]])
    leaves.sort(key=lambda leaf: canonical_json(leaf[0]))
    encoded = canonical_json(leaves)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def response_schema_issues(content, schema):
    """Return bounded structural errors for a strict JSON final response."""
    try:
        instance = _strict_response_json(content)
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as exc:
        return [f"invalid_json: {str(exc)[:240]}"]
    if not _response_value_within_limits(instance):
        return ["response_complexity_limit: JSON nesting or value count exceeds the structural validation limit"]
    try:
        validator = host_validator(schema)
        errors = sorted(validator.iter_errors(instance),
                        key=lambda item: (tuple(map(str, item.absolute_path)), item.message))
    except Exception as exc:
        return [f"schema_validation_error: {type(exc).__name__}: {str(exc)[:200]}"]
    return [f"{'.'.join(map(str, error.absolute_path)) or '$'}: {error.message[:240]}" for error in errors[:8]]


def is_computed_data(source, content):
    if source["kind"] != "computed":
        return False
    media_type = source["media_type"].split(";", 1)[0].strip()
    try:
        if media_type == "application/json":
            bounded_json(json.loads(content))
            return True
        if media_type in {"text/csv", "text/tab-separated-values"}:
            rows = csv.reader(io.StringIO(content.decode("utf-8")), delimiter="\t" if media_type.endswith("tab-separated-values") else ",")
            first = next(rows, None)
            second = next(rows, None)
            return bool(first and second and len(first) == len(second))
    except (ValueError, UnicodeError, TypeError):
        return False
    return False


def structural_issues(report, task, materials):
    issues = []
    try:
        required = requirements(task["acceptance"])
    except (ValueError, TypeError) as exc:
        return [f"invalid_requirements: {exc}"]
    if required["require_report"] and not report.get("artifact_id"):
        issues.append("report_artifact_required")
    if required.get("require_research_design") and not report.get("research_design"):
        issues.append("research_design_required")
    headings = [s["heading"].strip().casefold() for s in report["sections"]]
    if not headings:
        issues.append("sections_required")
    if len(set(headings)) != len(headings):
        issues.append("duplicate_sections")
    for heading in required["required_sections"]:
        if heading.strip().casefold() not in headings:
            issues.append(f"required_section_missing: {heading}")
    if "paragraphs_per_section" in required:
        expected = required["paragraphs_per_section"]
        for section in report["sections"]:
            actual = len(section.get("paragraphs", []))
            if actual != expected:
                issues.append(f"paragraphs_per_section_mismatch: {section['heading']}: {actual}/{expected}")
    if any(not section["claim_ids"] for section in report["sections"]):
        issues.append("nonempty_sections_required")
    if not report["claim_snapshots"]:
        issues.append("referenced_claim_required")
    evidence = [e for snapshot in report["claim_snapshots"] for e in snapshot["evidence"]]
    if required["require_evidence"] and not evidence:
        issues.append("referenced_evidence_required")
    if COVERAGE_REQUIREMENTS & required.keys():
        coverage = evidence_coverage(report)
        for field in sorted(COVERAGE_REQUIREMENTS & required.keys()):
            metric = field.removeprefix("min_")
            if coverage[metric] < required[field]:
                issues.append(f"{metric}_below_minimum: {coverage[metric]}/{required[field]}")
    need_limitations = required["require_limitations"]
    has_computed_data = False
    for snapshot in report["claim_snapshots"]:
        claim, assessment = snapshot["claim"], snapshot["assessment"]
        if required.get("require_literal_observations") and claim["kind"] == "observation" and claim.get("origin") != "host_observation":
            issues.append(f"literal_observation_required: {claim['id']}")
        supported = assessment and snapshot["assessment_current"] and assessment["status"] == "supported"
        if claim["kind"] == "observation" and not supported:
            issues.append(f"observation_not_supported: {claim['id']}")
        if claim["kind"] != "observation" or not supported or (assessment and assessment["provisional"]):
            need_limitations = True
        for source in snapshot["sources"]:
            content = materials.get(source["artifact_id"])
            if content is not None and is_computed_data(source, content):
                has_computed_data = True
    if need_limitations and not report["limitations"]:
        issues.append("explicit_limitations_required")
    if required["require_data_artifact"] and not has_computed_data:
        issues.append("referenced_computed_data_required")
    return issues
