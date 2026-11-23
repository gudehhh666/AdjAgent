import hashlib
import json
from pathlib import Path

import pytest

from adj_partner.research.acceptance import RESPONSE_SCHEMA, requirements, research_acceptance, task_binding
from adj_partner.runtime.execution_delivery import VerificationFailure
from adj_partner.runtime.execution_witness import bind_execution_inputs, verify_execution_witness
from tests.test_execution_delivery import _fixture


SCHEMA = {
    "type": "object",
    "properties": {"hypothesis": {"type": "string"}},
    "required": ["hypothesis"],
    "additionalProperties": False,
}


def test_witness_contract_is_independent_of_exact_summary_delivery():
    accepted = requirements({RESPONSE_SCHEMA: SCHEMA, "require_bound_execution_witness": True})
    assert accepted["require_bound_execution_witness"] is True
    assert accepted["require_report"] is False
    assert not research_acceptance({RESPONSE_SCHEMA: SCHEMA, "require_bound_execution_witness": True})


@pytest.mark.parametrize("value", [1, "yes", None])
def test_witness_contract_requires_exact_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        requirements({RESPONSE_SCHEMA: SCHEMA, "require_bound_execution_witness": value})


def test_witness_requires_schema_and_rejects_report_acceptance():
    with pytest.raises(ValueError, match="response_json_schema"):
        requirements({"require_bound_execution_witness": True})
    with pytest.raises(ValueError, match="research|report|require_bound_execution_witness"):
        requirements({RESPONSE_SCHEMA: SCHEMA, "require_bound_execution_witness": True,
                      "require_evidence": True})


def test_binding_canonicalizes_allowed_root_and_input_hash(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    data = root / "table.csv"
    data.write_bytes(b"x,y\n1,2\n")
    binding = bind_execution_inputs(
        task_binding={"id": "task-1", "requirements_hash": "a" * 64, "observed_revision": 0},
        allowed_roots=[str(root)], input_paths=[str(data)], response_schema=SCHEMA,
    )
    assert binding["inputs"][0]["relative_path"] == "table.csv"
    assert binding["inputs"][0]["sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()
    assert binding["inputs"][0]["root_id"] == binding["allowed_roots"][0]["root_id"]


def test_empty_bound_input_list_is_rejected(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    with pytest.raises(VerificationFailure, match="input"):
        bind_execution_inputs(
            task_binding={"id": "task-1", "requirements_hash": "a" * 64, "observed_revision": 0},
            allowed_roots=[str(root)], input_paths=[], response_schema=SCHEMA,
        )


def _witness_binding(fixture, *, acceptance=None, input_paths=()):
    store = fixture["store"]
    task = store.get_task(store.get_run(fixture["run"])["task_id"])
    session = store.get_session(fixture["session"])
    return bind_execution_inputs(task_binding(task), session["config"]["allowed_roots"], input_paths,
                                 response_schema=SCHEMA if acceptance else None)


def test_verified_witness_accepts_real_durable_analysis_shape_without_schema_match(tmp_path):
    (tmp_path / "table.csv").write_bytes(b"a,b\n1,2\n")
    fixture = _fixture(tmp_path)
    binding = _witness_binding(fixture, input_paths=[tmp_path / "table.csv"])

    witness = verify_execution_witness(fixture["store"], fixture["run"], binding)

    assert witness["kind"] == "execution_witness"
    assert witness["witness_count"] == 1
    assert witness["witnesses"][0]["provenance"]["call_id"] == "call_fixture"
    assert witness["narrative_derivation_validated"] is False
    assert "input path identity" in witness["assurance"]
    assert str(tmp_path) not in json.dumps(witness)
    assert "relative_path" not in witness["witnesses"][0]["provenance"]


def test_same_content_copy_at_different_path_is_not_a_bound_witness(tmp_path):
    worker_root = tmp_path / "worker"
    worker_root.mkdir()
    worker_input = worker_root / "table.csv"
    worker_input.write_bytes(b"a,b\n1,2\n")
    other_input = worker_root / "staged" / "table.csv"
    other_input.parent.mkdir()
    other_input.write_bytes(worker_input.read_bytes())
    fixture = _fixture(worker_root)
    binding = _witness_binding(fixture, input_paths=[other_input])

    assert verify_execution_witness(fixture["store"], fixture["run"], binding) is None


def test_symlinked_worker_input_path_fails_closed(tmp_path):
    real = tmp_path / "real.csv"
    real.write_bytes(b"a,b\n1,2\n")
    recorded = tmp_path / "table.csv"
    recorded.symlink_to(real)
    fixture = _fixture(tmp_path)
    binding = _witness_binding(fixture, input_paths=[real])

    with pytest.raises(VerificationFailure, match="symlink"):
        verify_execution_witness(fixture["store"], fixture["run"], binding)


def test_wrong_bound_input_hash_fails_closed(tmp_path):
    (tmp_path / "table.csv").write_bytes(b"a,b\n1,2\n")
    fixture = _fixture(tmp_path)
    binding = _witness_binding(fixture, input_paths=[tmp_path / "table.csv"])
    (tmp_path / "table.csv").write_bytes(b"tampered\n")

    with pytest.raises(VerificationFailure, match="input hash"):
        verify_execution_witness(fixture["store"], fixture["run"], binding)


def test_failed_or_absent_analysis_has_no_witness(tmp_path):
    fixture = _fixture(tmp_path, runtime_action_shape=True)
    (tmp_path / "table.csv").write_bytes(b"a,b\n1,2\n")
    binding = _witness_binding(fixture, input_paths=[tmp_path / "table.csv"])
    with fixture["store"].transaction(write=True) as conn:
        conn.execute("DELETE FROM actions WHERE run_id=?", (fixture["run"],))

    assert verify_execution_witness(fixture["store"], fixture["run"], binding) is None


def test_directory_symlink_input_is_rejected(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    real = root / "real"
    real.mkdir()
    (real / "table.csv").write_bytes(b"x\n1\n")
    (root / "alias").symlink_to(real, target_is_directory=True)
    with pytest.raises(VerificationFailure, match="symlink"):
        bind_execution_inputs(
            task_binding={"id": "task-1", "requirements_hash": "a" * 64, "observed_revision": 0},
            allowed_roots=[str(root)], input_paths=[str(root / "alias" / "table.csv")], response_schema=SCHEMA,
        )


def test_task_binding_and_artifact_tampering_fail_closed(tmp_path):
    (tmp_path / "table.csv").write_bytes(b"a,b\n1,2\n")
    fixture = _fixture(tmp_path)
    binding = _witness_binding(fixture, input_paths=[tmp_path / "table.csv"])
    binding["task_binding"] = {**binding["task_binding"], "id": "other-task"}
    binding["binding_sha256"] = __import__("adj_partner.runtime.execution_witness", fromlist=["_binding_digest"])._binding_digest(binding)
    with pytest.raises(VerificationFailure, match="task binding"):
        verify_execution_witness(fixture["store"], fixture["run"], binding)

    binding = _witness_binding(fixture, input_paths=[tmp_path / "table.csv"])
    with fixture["store"].transaction(write=True) as conn:
        conn.execute("UPDATE artifacts SET sha256=? WHERE id=?", ("0" * 64, fixture["result_artifact"]))
    with pytest.raises(VerificationFailure, match="hash|integrity|provenance"):
        verify_execution_witness(fixture["store"], fixture["run"], binding)

