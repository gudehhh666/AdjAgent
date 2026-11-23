"""Single-question benchmark runner keeps hidden fields host-side and out of state."""

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("suffix,content,selector", [
    (".json", json.dumps([{"id": "qa_1", "question": "Public question?", "difficulty_level": "L2",
                          "paper_id": "SECRET-PAPER", "answer": "SECRET-ANSWER",
                          "symbolic_formalization": {"reasoning": "SECRET-CHAIN"}}]), "qa_1"),
    (".jsonl", json.dumps({"Question:": "Public question?", "difficulty_level": "L3",
                            "paper_id": "SECRET-PAPER", "Answer:": "SECRET-ANSWER",
                            "Reasoning_Process:": "SECRET-CHAIN"}) + "\n", "1"),
])
def test_load_public_case_exposes_only_question_and_difficulty(tmp_path, suffix, content, selector):
    from adj_partner.open_research_case import load_public_case

    path = tmp_path / ("cases" + suffix)
    path.write_text(content)

    case = load_public_case(path, selector)

    assert set(case) == {"question", "difficulty_level", "dataset_sha256", "selector"}
    assert case["question"] == "Public question?"
    assert "SECRET" not in json.dumps(case)


def test_json_selector_rejects_unknown_id_instead_of_guessing(tmp_path):
    from adj_partner.open_research_case import load_public_case

    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{"id": "qa_1", "question": "Question?", "difficulty_level": "L1"}]))

    with pytest.raises(ValueError, match="selector"):
        load_public_case(path, "1")


class _FakeResearch:
    def list_sources(self, session_id, run_id):
        return [{"id": "source_1", "title": "Fixture source", "kind": "literature",
                 "provenance": {"url": "https://example.org/paper", "text_availability": "full_text"}}]


class _FakeService:
    instances = []

    def __init__(self, root, **kwargs):
        self.root = Path(root)
        self.root.mkdir(parents=True)
        self.kwargs = kwargs
        self.research = _FakeResearch()
        self.goal = None
        self.closed = False
        type(self).instances.append(self)

    def create_session(self, **kwargs):
        self.session_kwargs = kwargs
        return {"id": "session_fixture"}

    def set_model(self, session_id, model):
        self.model = (session_id, model)

    def submit(self, session_id, goal, acceptance=None):
        self.goal, self.acceptance = goal, acceptance
        (self.root / "public-state.json").write_text(json.dumps({"goal": goal, "acceptance": acceptance}))
        return {"id": "run_fixture"}

    def drive(self, run_id):
        content = "# Research report\n\nFixture.\n"
        return {"id": run_id, "session_id": "session_fixture", "status": "succeeded",
                "state": {"delivery": {"kind": "accepted_report", "content": content,
                                       "report_id": "report_1", "artifact_id": "artifact_1",
                                       "sha256": hashlib.sha256(content.encode()).hexdigest()}}}

    def trace(self, session_id, destination, run_id):
        target = Path(destination) / "trace_fixture"
        target.mkdir(parents=True)
        (target / "trace.md").write_text("# Trace\n")
        return {"markdown": str(target / "trace.md")}

    def close(self):
        self.closed = True


def test_runner_never_persists_hidden_fields_and_publishes_exact_markdown(tmp_path):
    from adj_partner.open_research_case import run_case

    dataset = tmp_path / "case.json"
    dataset.write_text(json.dumps([{"id": "qa_1", "question": "Public question?", "difficulty_level": "L4",
        "paper_id": "SECRET-PAPER", "answer": "SECRET-ANSWER", "reasoning": "SECRET-CHAIN"}]))
    output = tmp_path / "output"

    result = run_case(dataset, "qa_1", output, service_factory=_FakeService, model="gpt-5.6-luna")

    assert Path(result["report_path"]).read_text() == "# Research report\n\nFixture.\n"
    persisted = b"".join(path.read_bytes() for path in output.rglob("*") if path.is_file())
    assert b"SECRET-PAPER" not in persisted
    assert b"SECRET-ANSWER" not in persisted
    assert b"SECRET-CHAIN" not in persisted
    instance = _FakeService.instances[-1]
    assert "Public question?" in instance.goal and "L4" in instance.goal
    assert instance.model == ("session_fixture", "gpt-5.6-luna")
    assert instance.session_kwargs["allowed_roots"] == []
    assert instance.session_kwargs["budget"] == {
        "max_steps": 64, "max_input_tokens": 800000,
        "max_output_tokens": 80000, "wall_time_seconds": 1800.0,
    }
    assert instance.closed is True


def test_runner_refuses_to_overwrite_existing_output_directory(tmp_path):
    from adj_partner.open_research_case import run_case

    dataset = tmp_path / "case.json"
    dataset.write_text(json.dumps([{"id": "qa_1", "question": "Question?", "difficulty_level": "L1"}]))
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run_case(dataset, "qa_1", output, service_factory=_FakeService)


def test_cli_exposes_single_case_arguments():
    script = Path(__file__).resolve().parents[1] / "scripts/run_open_research_case.py"
    if not script.exists():
        pytest.skip("scripts/run_open_research_case.py is not part of this checkout")

    completed = subprocess.run([sys.executable, str(script), "--help"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    assert completed.returncode == 0
    assert "--dataset" in completed.stdout and "--selector" in completed.stdout
    assert "--output" in completed.stdout and "--model" in completed.stdout


def test_cli_budget_overrides_reach_runner_and_invalid_values_do_not_run(monkeypatch):
    from scripts import run_open_research_case as cli
    captured=[]
    monkeypatch.setattr(cli,'run_case',lambda *args,**kwargs: captured.append(kwargs) or {'status':'succeeded'})
    base=['--dataset','public.json','--selector','qa_007','--output','fresh']
    assert cli.main(base+['--max-steps','160','--max-input-tokens','2400000','--max-output-tokens','240000','--wall-time-seconds','3600'])==0
    assert captured[-1]['budget']=={'max_steps':160,'max_input_tokens':2400000,'max_output_tokens':240000,'wall_time_seconds':3600.0}
    assert cli.main(base+['--max-input-tokens','0'])==1
    assert len(captured)==1
    assert cli.main(base)==0
    assert captured[-1]['budget'] is None
