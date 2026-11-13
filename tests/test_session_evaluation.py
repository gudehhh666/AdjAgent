"""Reproducible engineering scenarios, explicitly not live model benchmarks."""

import importlib.util
import json
from pathlib import Path

import pytest


def module():
    assert importlib.util.find_spec("adj_partner.evaluation"), "Repeatable historical-case evaluation runner must exist"
    from adj_partner import evaluation
    return evaluation


def test_cases_preserve_exact_historical_questions_and_criteria():
    cases = module().load_cases()
    assert [case["id"] for case in cases] == ["historical_deg", "historical_ferritin", "protein_extension"]
    assert len(cases[1]["acceptance"]["required_sections"]) == 14
    assert cases[0]["question"].endswith("and through which mechanism?")


def test_live_evaluation_requires_explicit_remote_call_confirmation(tmp_path):
    with pytest.raises(ValueError, match="confirm"):
        module().run_cases(tmp_path / "live", case_ids=["protein_extension"], mode="live")
    assert not (tmp_path / "live").exists()


def test_offline_protein_case_uses_actual_job_gate_and_trace(tmp_path):
    summary = module().run_cases(tmp_path / "evaluation", case_ids=["protein_extension"], mode="offline")
    item = summary["cases"][0]
    assert item["runtime_completed"] and item["structural_acceptance"]
    assert item["numerical_checks"]["passed"] and not item["scientific_answer_validated"]
    assert item["model_mode"] == "offline-scripted"
    assert Path(item["trace"]["manifest"]).is_file()
    assert Path(summary["summary_path"]).is_file()
    assert item["job_count"] == 1 and item["report_sha256"]


def test_evaluation_never_overwrites_prior_results(tmp_path):
    output = tmp_path / "used"
    output.mkdir()
    (output / "keep.txt").write_text("prior result")
    with pytest.raises((ValueError, FileExistsError)):
        module().run_cases(output, case_ids=["protein_extension"], mode="offline")
    assert (output / "keep.txt").read_text() == "prior result"


def test_numeric_regression_cannot_return_success_exit_code(tmp_path, monkeypatch):
    evaluation = module()
    monkeypatch.setattr(evaluation, "run_cases", lambda *args, **kwargs: {"cases": [{
        "runtime_completed": True, "structural_acceptance": True, "trace": {"manifest": "retained"},
        "numerical_checks": {"applicable": True, "passed": False},
    }]})
    assert evaluation.main(["--output", str(tmp_path / "unused")]) == 1
