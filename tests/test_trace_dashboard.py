import csv
import importlib.util
import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from adj_partner.trace_dashboard import DashboardRequestError, TraceDashboardData, create_server


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def dashboard_batch(tmp_path):
    root = tmp_path / "batch"
    root.mkdir()
    analysis = {
        "schema_version": 3,
        "generated_at": "2026-09-02T00:00:00Z",
        "model": "gpt-5.6-luna",
        "scope": {
            "selected_cases": 1,
            "attempt_directories": 2,
            "manifested_attempts": 2,
            "selected_attempts": 1,
            "all_tool_calls": 3,
            "selected_tool_calls": 2,
        },
        "level_summary": {
            "L1": {
                "cases": 1,
                "attempts": 2,
                "selected_successes": 1,
                "all_tool_calls": 3,
                "selected_tool_calls": 2,
                "attempt_statuses": {"failed": 1, "succeeded": 1},
                "stop_reasons": {"provider_error": 1, "accepted_report": 1},
                "report_metrics": {"words": {"median": 1200}},
            }
        },
        "category_distribution": [
            {"scope": "all_attempts", "level": "L1", "category": "discovery", "calls": 2,
             "completed": 1, "failed": 1, "started_or_other": 0},
            {"scope": "all_attempts", "level": "L1", "category": "reporting", "calls": 1,
             "completed": 1, "failed": 0, "started_or_other": 0},
        ],
        "tool_distribution": [
            {"scope": "all_attempts", "level": "L1", "category": "discovery",
             "tool": "literature_search", "calls": 2, "completed": 1, "failed": 1,
             "started_or_other": 0, "completion_rate": 0.5, "share_of_level_calls": 2 / 3,
             "attempts_using": 2, "status_counts": {"completed": 1, "failed": 1}},
            {"scope": "all_attempts", "level": "L1", "category": "reporting",
             "tool": "report_create", "calls": 1, "completed": 1, "failed": 0,
             "started_or_other": 0, "completion_rate": 1.0, "share_of_level_calls": 1 / 3,
             "attempts_using": 1, "status_counts": {"completed": 1}},
        ],
        "cases": [],
        "attempts": [],
    }
    (root / "quality-analysis.json").write_text(json.dumps(analysis), encoding="utf-8")

    case_fields = [
        "level", "case_id", "selected_attempt", "report", "tool_calls", "failed_tool_calls",
        "steps", "elapsed_active_seconds", "input_tokens", "output_tokens", "report_bytes",
        "words", "sections", "limitations", "claims", "fully_bound_claims", "unbound_claims",
        "bound_sources", "bound_fulltext_sources", "source_records", "unassessed_claims",
        "gate_delivery_gates_pass", "question",
    ]
    _write_csv(root / "quality-cases.csv", case_fields, [{
        "level": "L1", "case_id": "qa_001", "selected_attempt": "L1/qa_001-retry-01",
        "report": "L1/qa_001-retry-01/report.md", "tool_calls": 2, "failed_tool_calls": 0,
        "steps": 2, "elapsed_active_seconds": 4.5, "input_tokens": 100, "output_tokens": 20,
        "report_bytes": 99, "words": 1200, "sections": 3, "limitations": 2, "claims": 2,
        "fully_bound_claims": 2, "unbound_claims": 0, "bound_sources": 1,
        "bound_fulltext_sources": 1, "source_records": 4, "unassessed_claims": 1,
        "gate_delivery_gates_pass": "True", "question": "How should the adjuvant be designed?",
    }])

    attempt_fields = [
        "level", "case_id", "attempt", "retry_index", "selected", "manifested", "status",
        "run_state", "stop_reason", "tool_calls", "completed_tool_calls", "failed_tool_calls",
        "unfinished_tool_calls", "steps", "elapsed_active_seconds", "input_tokens", "output_tokens",
        "accounted_input_tokens", "accounted_output_tokens", "unknown_requests", "db_present",
    ]
    _write_csv(root / "quality-attempts.csv", attempt_fields, [
        {"level": "L1", "case_id": "qa_001", "attempt": "L1/qa_001", "retry_index": 0,
         "selected": "False", "manifested": "True", "status": "failed", "run_state": "failed",
         "stop_reason": "provider_error", "tool_calls": 1, "completed_tool_calls": 0,
         "failed_tool_calls": 1, "unfinished_tool_calls": 0, "steps": 1,
         "elapsed_active_seconds": 1.0, "input_tokens": 10, "output_tokens": 2,
         "accounted_input_tokens": 10, "accounted_output_tokens": 2, "unknown_requests": 0,
         "db_present": "True"},
        {"level": "L1", "case_id": "qa_001", "attempt": "L1/qa_001-retry-01", "retry_index": 1,
         "selected": "True", "manifested": "True", "status": "succeeded", "run_state": "succeeded",
         "stop_reason": "accepted_report", "tool_calls": 2, "completed_tool_calls": 2,
         "failed_tool_calls": 0, "unfinished_tool_calls": 0, "steps": 2,
         "elapsed_active_seconds": 4.5, "input_tokens": 100, "output_tokens": 20,
         "accounted_input_tokens": 100, "accounted_output_tokens": 20, "unknown_requests": 0,
         "db_present": "True"},
    ])
    _write_csv(root / "quality-tools-by-level.csv",
               ["scope", "level", "category", "tool", "calls", "completed", "failed",
                "started_or_other", "status_counts", "attempts_using", "share_of_level_calls",
                "completion_rate"], [
        {"scope": "all_attempts", "level": "L1", "category": "discovery",
         "tool": "literature_search", "calls": 2, "completed": 1, "failed": 1,
         "started_or_other": 0, "status_counts": json.dumps({"completed": 1, "failed": 1}),
         "attempts_using": 2, "share_of_level_calls": 2 / 3, "completion_rate": 0.5},
        {"scope": "all_attempts", "level": "L1", "category": "reporting",
         "tool": "report_create", "calls": 1, "completed": 1, "failed": 0,
         "started_or_other": 0, "status_counts": json.dumps({"completed": 1}),
         "attempts_using": 1, "share_of_level_calls": 1 / 3, "completion_rate": 1.0},
    ])

    for attempt in ("qa_001", "qa_001-retry-01"):
        attempt_root = root / "L1" / attempt
        trace_root = attempt_root / "trace" / ("trace_" + attempt.replace("-", "_"))
        trace_root.mkdir(parents=True)
        records = [
            {"record_type": "action", "data": {
                "id": "action_1", "call_id": "call_1", "tool_name": "literature_search",
                "status": "completed", "created_at": "2026-09-02T00:00:01Z",
                "finished_at": "2026-09-02T00:00:02Z", "args": {"query": "adjuvant"},
                "result": {"status": "ok", "error": None, "artifacts": [], "data": {"returned": 1}},
            }},
            {"record_type": "event", "data": {
                "seq": 2, "kind": "model.completed", "created_at": "2026-09-02T00:00:03Z",
                "payload": {"input_tokens": 10, "output_tokens": 2},
            }},
            {"record_type": "message", "data": {
                "seq": 3, "created_at": "2026-09-02T00:00:04Z",
                "message": {"role": "assistant", "content": "Finished"},
            }},
        ]
        (trace_root / "trace.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
        (trace_root / "trace.md").write_text("# Public trace\n", encoding="utf-8")
        (attempt_root / "report.md").write_text("# Research report\n", encoding="utf-8")
        (attempt_root / "manifest.json").write_text(json.dumps({
            "status": "succeeded" if "retry" in attempt else "failed",
            "trace": {
                "jsonl": str(trace_root / "trace.jsonl"),
                "markdown": str(trace_root / "trace.md"),
            },
        }), encoding="utf-8")

    (root / "delivery-index.json").write_text(json.dumps({"cases": 1}), encoding="utf-8")
    return root


def test_reader_exposes_overview_and_indexed_rows(dashboard_batch):
    data = TraceDashboardData(dashboard_batch)

    assert data.health()["cases"] == 1
    assert data.health()["attempts"] == 2
    assert data.overview()["totals"]["tool_calls"] == 3
    assert data.overview()["levels"]["L1"]["attempts"] == 2
    from adj_partner.profiles import PROFILES
    assert data.overview()["registered_tools"] == sorted({t for p in PROFILES.values() for t in p["tools"]})
    assert {"adjuvant_search", "adjuvant_record", "adjuvant_relations", "adjuvant_skill", "adjuvant_excerpt",
            "design_record_read", "experiment_state", "experiment_design_revise"} <= set(data.overview()["registered_tools"])
    assert {"chembl_compound", "chembl_activity", "reactome_pathway", "geo_search"} <= set(data.overview()["registered_tools"])
    assert data.overview()["profiles"]["general"] == list(dict.fromkeys(PROFILES["general"]["tools"]))
    assert {"chembl_compound", "chembl_activity", "reactome_pathway", "geo_search"} <= set(data.overview()["profiles"]["general"])
    assert data.cases()[0]["case_id"] == "qa_001"
    assert data.cases()[0]["delivery_pass"] is True
    assert data.attempts()[1]["selected"] is True
    assert data.tools()[0]["status_counts"] == {"completed": 1, "failed": 1}


def test_reader_rejects_missing_machine_readable_inputs(tmp_path):
    root = tmp_path / "incomplete"
    root.mkdir()

    with pytest.raises(FileNotFoundError, match="quality-analysis.json"):
        TraceDashboardData(root)


def test_trajectory_is_normalized_paginated_and_preserves_public_details(dashboard_batch):
    data = TraceDashboardData(dashboard_batch)

    page = data.trajectory("L1/qa_001-retry-01", offset=0, limit=2)

    assert page["attempt"] == "L1/qa_001-retry-01"
    assert page["total"] == 3
    assert page["next_offset"] == 2
    assert [row["type"] for row in page["records"]] == ["action", "event"]
    assert page["records"][0]["tool"] == "literature_search"
    assert page["records"][0]["args"] == {"query": "adjuvant"}
    assert page["records"][0]["result_data"] == {"returned": 1}
    assert page["records"][1]["kind"] == "model.completed"

    final = data.trajectory("L1/qa_001-retry-01", offset=2, limit=2)
    assert final["next_offset"] is None
    assert final["records"][0]["role"] == "assistant"


def test_trajectory_keeps_malformed_line_visible(dashboard_batch):
    trace = next((dashboard_batch / "L1/qa_001-retry-01/trace").glob("*/trace.jsonl"))
    with trace.open("a", encoding="utf-8") as handle:
        handle.write("{not-json}\n")

    page = TraceDashboardData(dashboard_batch).trajectory(
        "L1/qa_001-retry-01", offset=3, limit=1)

    assert page["total"] == 4
    assert page["records"] == [{
        "type": "parse_error", "index": 3, "error": "Malformed public trace JSON",
        "preview": "{not-json}",
    }]


@pytest.mark.parametrize("attempt", ["../../.env", "L1/qa_999", "L1%2Fqa_001"])
def test_trajectory_rejects_unknown_or_traversal_attempts(dashboard_batch, attempt):
    with pytest.raises(DashboardRequestError, match="Unknown attempt"):
        TraceDashboardData(dashboard_batch).trajectory(attempt, offset=0, limit=10)


@pytest.mark.parametrize("offset,limit", [(-1, 10), (0, 0), (0, 101)])
def test_trajectory_rejects_invalid_pagination(dashboard_batch, offset, limit):
    with pytest.raises(DashboardRequestError, match="pagination"):
        TraceDashboardData(dashboard_batch).trajectory(
            "L1/qa_001-retry-01", offset=offset, limit=limit)


def test_raw_export_allows_only_known_public_files(dashboard_batch):
    data = TraceDashboardData(dashboard_batch)

    report, report_type = data.raw_export("L1/qa_001-retry-01", "report")
    trace, trace_type = data.raw_export("L1/qa_001-retry-01", "trace")
    jsonl, jsonl_type = data.raw_export("L1/qa_001-retry-01", "jsonl")

    assert report.read_text(encoding="utf-8").startswith("# Research report")
    assert report_type == "text/markdown; charset=utf-8"
    assert trace_type == "text/markdown; charset=utf-8"
    assert jsonl_type == "application/x-ndjson; charset=utf-8"
    assert jsonl.name == "trace.jsonl"
    with pytest.raises(DashboardRequestError, match="Unsupported raw export"):
        data.raw_export("L1/qa_001-retry-01", "state")


def test_raw_export_rejects_symlink_outside_batch(dashboard_batch, tmp_path):
    report = dashboard_batch / "L1/qa_001-retry-01/report.md"
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    report.unlink()
    report.symlink_to(outside)

    with pytest.raises(DashboardRequestError, match="not a safe public export"):
        TraceDashboardData(dashboard_batch).raw_export("L1/qa_001-retry-01", "report")


@pytest.fixture
def dashboard_http(dashboard_batch, tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><h1>Trace dashboard</h1>", encoding="utf-8")
    (static / "styles.css").write_text("body { color: white; }", encoding="utf-8")
    (static / "app.js").write_text("document.body.dataset.ready = 'true';", encoding="utf-8")
    server = create_server(TraceDashboardData(dashboard_batch), static, host="127.0.0.1", port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def _get_json(url):
    with urlopen(url, timeout=3) as response:
        return response, json.loads(response.read())


def test_http_serves_health_lists_and_static_assets(dashboard_http):
    response, health = _get_json(dashboard_http + "/api/health")
    assert response.status == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert health["cases"] == 1

    for endpoint, expected in (("overview", "totals"), ("cases", None),
                               ("attempts", None), ("tools", None)):
        _, payload = _get_json(dashboard_http + "/api/" + endpoint)
        assert expected in payload if expected else isinstance(payload, list)

    with urlopen(dashboard_http + "/", timeout=3) as response:
        assert response.headers.get_content_type() == "text/html"
        assert b"Trace dashboard" in response.read()
    with urlopen(dashboard_http + "/?case=qa_001&attempt=L1%2Fqa_001-retry-01", timeout=3) as response:
        assert response.headers.get_content_type() == "text/html"
        assert b"Trace dashboard" in response.read()
    with urlopen(dashboard_http + "/styles.css", timeout=3) as response:
        assert response.headers.get_content_type() == "text/css"


def test_http_trajectory_and_raw_exports(dashboard_http):
    _, page = _get_json(
        dashboard_http + "/api/trajectory?attempt=L1%2Fqa_001-retry-01&offset=0&limit=2")
    assert page["total"] == 3
    assert page["records"][0]["tool"] == "literature_search"

    with urlopen(dashboard_http + "/api/raw?attempt=L1%2Fqa_001-retry-01&kind=report", timeout=3) as response:
        assert response.headers.get_content_type() == "text/markdown"
        assert response.read().startswith(b"# Research report")


@pytest.mark.parametrize("path,status", [
    ("/api/health?unexpected=1", 400),
    ("/api/trajectory?attempt=..%2F..%2F.env&offset=0&limit=10", 404),
    ("/api/trajectory?attempt=L1%2Fqa_001&offset=-1&limit=10", 400),
    ("/api/raw?attempt=L1%2Fqa_001&kind=state", 400),
    ("/api/missing", 404),
    ("/../.env", 404),
])
def test_http_returns_bounded_json_errors(dashboard_http, path, status):
    with pytest.raises(HTTPError) as raised:
        urlopen(dashboard_http + path, timeout=3)
    assert raised.value.code == status
    assert raised.value.headers.get_content_type() == "application/json"
    payload = json.loads(raised.value.read())
    assert payload["error"]


def test_server_cli_defaults_to_loopback_port_8080(dashboard_batch):
    script = Path(__file__).parents[1] / "scripts/serve_trace_dashboard.py"
    spec = importlib.util.spec_from_file_location("serve_trace_dashboard", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args(["--batch-root", str(dashboard_batch)])

    assert args.batch_root == dashboard_batch
    assert args.host == "127.0.0.1"
    assert args.port == 8080


def test_static_contract_contains_dashboard_working_surfaces():
    root = Path(__file__).parents[1] / "trace_dashboard"
    html = (root / "index.html").read_text(encoding="utf-8")

    for marker in (
        'id="kpi-grid"', 'id="level-grid"', 'id="tool-chart"',
        'id="tool-registry"',
        'id="case-search"', 'id="level-filter"', 'id="quality-filter"',
        'id="case-table-body"', 'id="case-detail"', 'id="attempt-tabs"',
        'id="record-type-filter"', 'id="tool-filter"', 'id="status-filter"',
        'id="timeline"', 'id="timeline-more"', 'aria-live="polite"',
        'id="trace-drawer"', 'id="trace-drawer-backdrop"',
        'id="trace-drawer-close"', 'id="trace-view-switcher"',
        'id="full-trace-frame"', 'role="dialog"', 'aria-modal="true"',
    ):
        assert marker in html
    assert '<link rel="stylesheet" href="/styles.css">' in html
    assert '<script src="/app.js" defer></script>' in html


def test_static_contract_supports_apis_filters_paging_and_accessibility():
    root = Path(__file__).parents[1] / "trace_dashboard"
    script = (root / "app.js").read_text(encoding="utf-8")
    styles = (root / "styles.css").read_text(encoding="utf-8")

    for endpoint in ("/api/overview", "/api/cases", "/api/attempts", "/api/tools",
                     "/api/trajectory", "/api/raw"):
        assert endpoint in script
    for behavior in ("selectCase", "selectAttempt", "loadTrajectory", "renderTimeline",
                     "applyCaseFilters", "applyTimelineFilters", "loadMoreRecords",
                     "displayValue", "openTraceDrawer", "closeTraceDrawer",
                     "selectTraceView"):
        assert f"function {behavior}" in script or f"async function {behavior}" in script
    assert '"查看轨迹"' in script
    assert 'row-action-button' in script
    assert 'full-trace-frame' in script
    assert 'addEventListener("keydown"' in script
    assert "URLSearchParams" in script
    assert "details" in script
    assert ":focus-visible" in styles
    assert "prefers-reduced-motion" in styles
    assert "@media" in styles
