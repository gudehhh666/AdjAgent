"""One bounded format repair; no unknown request or tool effects replayed."""

import pytest

from adj_partner.contracts import Budget
from adj_partner.models.base import ProviderFailure
from test_runtime_engine import setup, reply


def failure(code="malformed_reply", unknown=False):
    return ProviderFailure(code, "Synthetic rejected response", usage={"input_tokens": 7, "output_tokens": 3}, cost_unknown=unknown)


def test_review_task_repairs_one_known_format_error_and_accounts_both_requests(tmp_path):
    store, engine, sid, port, effects = setup(tmp_path, [failure(), reply("No accepted report yet")])
    run = engine.start(sid, "Design", acceptance={"report_layout": "review"})
    result = engine.resume(run["id"])
    assert len(port.requests) == 2
    assert result["usage"]["input_tokens"] == 57
    assert result["usage"]["output_tokens"] == 23
    assert effects == []
    assert result["status"] == "failed"  # format recovery cannot bypass report acceptance
    assert result["state"]["stop_reason"] == "acceptance_not_met"
    assert sum(e["kind"] == "runtime.model_reply_repair" for e in store.events(sid)) == 1


def test_review_format_repair_is_at_most_once(tmp_path):
    _, engine, sid, port, effects = setup(tmp_path, [failure(), failure(), reply("Must not be reached")])
    result = engine.resume(engine.start(sid, "Design", acceptance={"report_layout": "review"})["id"])
    assert len(port.requests) == 2 and effects == []
    assert result["state"]["error"]["code"] == "malformed_reply"


@pytest.mark.parametrize("error", [failure(unknown=True), failure(code="provider_connection")])
def test_uncertain_or_transport_failure_is_not_reissued(tmp_path, error):
    _, engine, sid, port, _ = setup(tmp_path, [error, reply("Must not be reached")])
    result = engine.resume(engine.start(sid, "Design", acceptance={"report_layout": "review"})["id"])
    assert len(port.requests) == 1 and result["status"] == "failed"


def test_format_repair_does_not_expand_step_budget(tmp_path):
    _, engine, sid, port, _ = setup(tmp_path, [failure(), reply("Must not be reached")], budget=Budget(max_steps=1))
    result = engine.resume(engine.start(sid, "Design", acceptance={"report_layout": "review"})["id"])
    assert len(port.requests) == 1 and result["status"] == "budget_exhausted"
