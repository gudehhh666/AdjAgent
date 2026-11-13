"""Small parent-owned integration extensions to the reviewed storage boundary."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from adj_partner.storage import Store


def test_register_provider_secrets_before_persistence_and_reopen(tmp_path):
    store = Store(tmp_path, secrets=["initial-private-value"])
    assert store.register_secrets(["new-provider-private-value", "", "new-provider-private-value"]) is None
    session = store.create_session({"instructions": "initial-private-value new-provider-private-value"})
    store.append_message(session["id"], {"role": "user", "content": "new-provider-private-value"})
    reopened = Store(tmp_path)
    assert reopened.get_session(session["id"])["config"]["instructions"] == "[REDACTED] [REDACTED]"
    assert reopened.messages(session["id"])[0]["content"] == "[REDACTED]"


def test_concurrent_registration_keeps_all_provider_secrets(tmp_path):
    store = Store(tmp_path)
    values = [f"provider-private-{index:03d}-value" for index in range(40)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: store.register_secrets([value]), values))
    assert store.redact(values) == ["[REDACTED]"] * len(values)


@pytest.mark.parametrize("values", ["not-a-list", [123], [None]])
def test_secret_registration_rejects_invalid_types_without_partial_update(tmp_path, values):
    store = Store(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        store.register_secrets(values)


def test_task_listing_includes_unstarted_tasks_after_reopen_and_preserves_scope(tmp_path):
    store = Store(tmp_path, secrets=["task-private-value"])
    first_session = store.create_session({})["id"]
    other_session = store.create_session({})["id"]
    first = store.create_task(first_session, "task-private-value", {"require_report": True})
    other = store.create_task(other_session, "other task")
    last = store.create_task(first_session, "started task")
    store.create_run(first_session, last["id"], {})
    first = store.update_task(first["id"], {"status": "waiting"}, first["revision"])

    reopened = Store(tmp_path)
    before = reopened.events(first_session)
    assert reopened.list_tasks(first_session) == [first, last]
    assert reopened.list_tasks(other_session) == [other]
    assert reopened.list_tasks() == [first, other, last]
    assert reopened.list_tasks(first_session)[0]["goal"] == "[REDACTED]"
    assert reopened.events(first_session) == before


def test_task_listing_empty_store_and_empty_session(tmp_path):
    store = Store(tmp_path)
    assert store.list_tasks() == []
    session = store.create_session({})
    assert store.list_tasks(session["id"]) == []


@pytest.mark.parametrize("session_id, error", [("session_missing", KeyError), ("../outside", ValueError)])
def test_task_listing_rejects_unknown_or_invalid_session(tmp_path, session_id, error):
    store = Store(tmp_path)
    with pytest.raises(error):
        store.list_tasks(session_id)
def test_runtime_accounting_counters_remain_numeric_but_similar_credentials_do_not(tmp_path):
    from adj_partner.storage import Store
    store = Store(tmp_path / "runtime-counter")
    assert store.redact({"accounted_input_tokens": 50, "accounted_output_tokens": 10,
                         "estimated_input_tokens": 500}) == {
        "accounted_input_tokens": 50, "accounted_output_tokens": 10, "estimated_input_tokens": 500}
    assert store.redact({"estimated_access_tokens": 123, "accounted_input_tokens": {"value": "credential"}}) == {
        "estimated_access_tokens": "[REDACTED]", "accounted_input_tokens": "[REDACTED]"}
