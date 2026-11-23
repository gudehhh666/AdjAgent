"""Default language policy is persisted and projected by the host service."""

import json

from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply, ToolCall
from adj_partner.models import ScriptedPort


ENGLISH_POLICY = (
    "Respond in English only in explanations, public plans, tool query narration, and final answers, "
    "regardless of the input language. Preserve scientific symbols, proper names, and source IDs "
    "exactly as needed. This language instruction does not change tool permissions, evidence "
    "requirements, or other session controls."
)


def test_default_bound_session_persists_and_projects_english_after_tool_roundtrip(tmp_path):
    port = ScriptedPort([
        ModelReply(content="Enabling a useful tool.", tool_calls=[
            ToolCall(id="enable1", name="tools_enable", arguments={"name": "catalog_search"})
        ]),
        ModelReply(content="Evidence lookup is ready."),
    ])
    service = AgentService(tmp_path / "state", model=port)
    try:
        session = service.create_session(instructions="Host note: use concise answers.")
        assert session["config"]["instructions"].endswith(ENGLISH_POLICY)
        assert session["config"]["instructions"].startswith("Host note: use concise answers.")
        assert session["config"]["allowed_roots"] == []
        assert session["config"]["allow_network"] is False
        assert "tools_enable" in session["config"]["allowed_tools"]

        run = service.submit(session["id"], "请先规划，然后查询相关证据并解释结果。")
        assert service.drive(run["id"])["status"] == "succeeded"
        assert len(port.requests) == 2
        for request in port.requests:
            system = next(message["content"] for message in request["messages"] if message["role"] == "system")
            assert ENGLISH_POLICY in system
        assert service.session(session["id"])["config"]["allow_network"] is False
    finally:
        service.close()


def test_reopened_default_session_keeps_language_policy_in_next_request(tmp_path):
    port = ScriptedPort([ModelReply(content="The answer is ready.")])
    service = AgentService(tmp_path / "state", model=port)
    session = service.create_session()
    service.close()

    service = AgentService(tmp_path / "state", model=port)
    try:
        run = service.submit(session["id"], "请用中文回答。")
        assert service.drive(run["id"])["status"] == "succeeded"
        system = next(message["content"] for message in port.requests[0]["messages"] if message["role"] == "system")
        assert ENGLISH_POLICY in system
    finally:
        service.close()


def test_explicit_general_instructions_are_unchanged(tmp_path):
    service = AgentService(tmp_path / "state", model=ScriptedPort([]))
    try:
        session = service.create_session(profile="general", instructions="Answer in the user's language.")
        assert session["config"]["instructions"] == "Answer in the user's language."
    finally:
        service.close()


def test_opening_existing_bound_session_does_not_migrate_its_config(tmp_path):
    state = tmp_path / "state"
    first = AgentService(state, model=ScriptedPort([]))
    saved = first.create_session(instructions="Older host instruction.")
    sid = saved["id"]
    legacy_config = dict(saved["config"])
    legacy_config["instructions"] = "Older host instruction."
    legacy_config_json = json.dumps(legacy_config, sort_keys=True, separators=(",", ":"))
    with first.store.transaction(write=True) as conn:
        conn.execute("UPDATE sessions SET config=? WHERE id=?", (legacy_config_json, sid))
        raw_config = conn.execute("SELECT config FROM sessions WHERE id=?", (sid,)).fetchone()["config"]
    first.close()

    reopened = AgentService(state, model=ScriptedPort([]))
    try:
        with reopened.store.transaction() as conn:
            migrated = conn.execute("SELECT config FROM sessions WHERE id=?", (sid,)).fetchone()["config"]
        assert migrated == raw_config
        assert reopened.session(sid)["config"]["instructions"] == "Older host instruction."
    finally:
        reopened.close()
