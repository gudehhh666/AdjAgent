import json
from datetime import datetime, timedelta, timezone

import pytest

from adj_partner.contracts import Budget, SessionConfig
from adj_partner.storage import Store
from adj_partner.tools.catalog import canonical_json


def make_run(store):
    session = store.create_session(SessionConfig().to_dict())
    task = store.create_task(session["id"], "analyze data")
    run = store.create_run(session["id"], task["id"], Budget().to_dict())
    return session["id"], run["id"]


def code_artifact(store, session_id, run_id, text="print('ok')\n"):
    return store.put_artifact(text, "analysis.py", "text/x-python", {}, session_id, run_id)


def test_records_and_analysis_artifacts_are_visible_to_a_later_session(tmp_path):
    from adj_partner.analysis.history import AnalysisHistory

    store = Store(tmp_path / "state")
    first_session, first_run = make_run(store)
    history = AnalysisHistory(store)
    code = code_artifact(store, first_session, first_run)
    record = history.create_record(first_session, first_run, language="python", title="Aminosalicylic counts",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={"timeout_seconds": 30})
    result = store.put_artifact(canonical_json({"summary": {"rows": 34899}}),
        "analysis_result.json", "application/json", {}, first_session, first_run)
    finished = history.finish(record["id"], status="succeeded", summary={"rows": 34899},
        artifacts=[("result", result["id"])], sandbox_backend="fixture")

    later_session, _ = make_run(store)
    found = history.search(later_session, query="Aminosalicylic")
    page = history.read_text(later_session, result["id"], start=0, max_characters=8000)

    assert finished["status"] == "succeeded"
    assert found["records"][0]["id"] == record["id"]
    assert found["records"][0]["artifacts"] == [
        {"ordinal": 0, "role": "code", "artifact_id": code["id"], "name": "analysis.py",
         "media_type": "text/x-python", "size": code["size"], "sha256": code["sha256"]},
        {"ordinal": 1, "role": "result", "artifact_id": result["id"],
         "name": "analysis_result.json", "media_type": "application/json",
         "size": result["size"], "sha256": result["sha256"]},
    ]
    assert json.loads(page["content"])["summary"]["rows"] == 34899
    assert page["next_start"] is None


def test_history_read_does_not_expose_unrelated_cross_session_artifacts(tmp_path):
    from adj_partner.analysis.history import AnalysisHistory

    store = Store(tmp_path / "state")
    first_session, first_run = make_run(store)
    unrelated = store.put_artifact("private", "private.txt", "text/plain", {}, first_session, first_run)
    later_session, _ = make_run(store)

    with pytest.raises(PermissionError, match="analysis history"):
        AnalysisHistory(store).read_text(later_session, unrelated["id"])


def test_history_read_rejects_invalid_utf8_with_a_stable_error(tmp_path):
    from adj_partner.analysis.history import AnalysisHistory

    store = Store(tmp_path / "state")
    session_id, run_id = make_run(store)
    history = AnalysisHistory(store)
    code = code_artifact(store, session_id, run_id)
    record = history.create_record(session_id, run_id, language="python", title="binary text",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={})
    invalid = store.put_artifact(b"\xff", "invalid.txt", "text/plain", {}, session_id, run_id)
    history.finish(record["id"], status="succeeded", summary={},
                   artifacts=[("output", invalid["id"])], sandbox_backend="fixture")

    with pytest.raises(ValueError, match="valid UTF-8"):
        history.read_text(session_id, invalid["id"])


def test_methods_are_versioned_from_successful_records_and_schema_is_local(tmp_path):
    from adj_partner.analysis.history import AnalysisHistory

    store = Store(tmp_path / "state")
    session_id, run_id = make_run(store)
    history = AnalysisHistory(store)
    code = code_artifact(store, session_id, run_id)
    record = history.create_record(session_id, run_id, language="python", title="count rows",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={})
    history.finish(record["id"], status="succeeded", summary={"rows": 3},
        artifacts=[], sandbox_backend="fixture")
    schema = {"type": "object", "properties": {"threshold": {"type": "number"}},
              "required": ["threshold"], "additionalProperties": False}

    first = history.register_method(session_id, record["id"], name="count_rows",
        description="Count rows", parameter_schema=schema)
    second = history.register_method(session_id, record["id"], name="count_rows",
        description="Count rows again", parameter_schema=schema)

    assert (first["version"], second["version"]) == (1, 2)
    assert first["code_sha256"] == code["sha256"]
    assert history.get_method(second["id"])["parameter_schema"] == schema
    with pytest.raises(ValueError, match="local JSON-schema"):
        history.register_method(session_id, record["id"], name="remote_schema",
            description="invalid", parameter_schema={"$ref": "https://example.invalid/schema"})
    with pytest.raises(ValueError, match="unsupported keywords"):
        history.register_method(session_id, record["id"], name="regex_schema",
            description="unsafe regex", parameter_schema={"type": "string", "pattern": "(a+)+$"})


def test_terminal_record_cannot_be_finished_twice_or_used_after_failure(tmp_path):
    from adj_partner.analysis.history import AnalysisHistory

    store = Store(tmp_path / "state")
    session_id, run_id = make_run(store)
    history = AnalysisHistory(store)
    code = code_artifact(store, session_id, run_id)
    record = history.create_record(session_id, run_id, language="bash", title="failed",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={})
    history.finish(record["id"], status="failed", summary={}, artifacts=[],
        sandbox_backend="fixture", error={"code": "analysis_exit_nonzero", "message": "exit 2"})

    with pytest.raises(ValueError, match="terminal"):
        history.finish(record["id"], status="succeeded", summary={}, artifacts=[], sandbox_backend="fixture")
    with pytest.raises(ValueError, match="successful"):
        history.register_method(session_id, record["id"], name="failed_method",
            description="not allowed", parameter_schema={"type": "object"})


def test_outer_job_terminalization_is_atomic_and_does_not_rewrite_terminal_history(tmp_path):
    from adj_partner.analysis.history import AnalysisHistory

    store = Store(tmp_path / "state")
    session_id, run_id = make_run(store)
    history = AnalysisHistory(store)
    code = code_artifact(store, session_id, run_id)
    record = history.create_record(session_id, run_id, language="python", title="cancelled outside",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={})

    cancelled = history.terminalize_if_active(record["id"], status="cancelled",
        sandbox_backend="outer_job_supervisor",
        error={"code": "cancelled", "message": "Outer job was cancelled"})
    unchanged = history.terminalize_if_active(record["id"], status="timed_out",
        sandbox_backend="outer_job_supervisor",
        error={"code": "timed_out", "message": "Too late"})

    assert cancelled["status"] == "cancelled"
    assert unchanged == cancelled


def test_submitter_can_attach_after_worker_wins_the_start_race(tmp_path):
    from adj_partner.analysis.history import AnalysisHistory

    store = Store(tmp_path / "state")
    session_id, run_id = make_run(store)
    history = AnalysisHistory(store)
    code = code_artifact(store, session_id, run_id)
    record = history.create_record(session_id, run_id, language="python", title="race",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={})

    history.mark_running(record["id"], "job_fast")
    history.finish(record["id"], status="succeeded", summary={"fast": True}, artifacts=[],
                   sandbox_backend="fixture")
    attached = history.attach_job(record["id"], "job_fast")

    assert attached["status"] == "succeeded" and attached["job_id"] == "job_fast"
    with pytest.raises(ValueError, match="cannot attach"):
        history.attach_job(record["id"], "job_different")


def test_history_filters_method_name_and_creation_window(tmp_path):
    from adj_partner.analysis.history import AnalysisHistory

    store = Store(tmp_path / "state")
    session_id, run_id = make_run(store)
    history = AnalysisHistory(store)
    code = code_artifact(store, session_id, run_id)
    record = history.create_record(session_id, run_id, language="python", title="filter source",
        code_artifact_id=code["id"], inputs=[], parameters={}, limits={})
    history.finish(record["id"], status="succeeded", summary={}, artifacts=[], sandbox_backend="fixture")
    method = history.register_method(session_id, record["id"], name="filter_rows",
        description="Filter rows", parameter_schema={"type": "object"})

    found = history.search(session_id, method_name="filter_rows",
        created_before=(datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat())
    future = history.search(session_id,
        created_after=(datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat())

    assert found["methods"][0]["id"] == method["id"]
    assert future == {"records": [], "methods": []}
    with pytest.raises(ValueError, match="timestamp"):
        history.search(session_id, created_after="not-a-timestamp")
