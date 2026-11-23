"""Headless real Textual/AgentService conversations; no UI-specific agent loop."""

import asyncio
import importlib.util
import threading
import time

from adj_partner.contracts import ModelReply, ToolCall
from adj_partner.models import ScriptedPort


def test_task_acceptance_label_distinguishes_json_schema_from_research_report():
    from adj_partner.tui import acceptance_label

    assert acceptance_label({"response_json_schema": {"type": "object"}}) == "JSON Schema"
    assert acceptance_label({"require_report": True}) == "严格报告"
    assert acceptance_label({}) == "普通聊天"


def build(tmp_path, replies):
    assert importlib.util.find_spec("adj_partner.tui"), "Conversational Textual client must exist"
    from adj_partner.application import AgentService
    from adj_partner.tui import AdjPartnerApp
    service = AgentService(tmp_path / "state", model=ScriptedPort(replies))
    sid = service.create_session()["id"]
    return service, AdjPartnerApp(service, sid), sid


async def enter(app, pilot, text):
    from textual.widgets import Input
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await pilot.pause(.05)


async def settled(app, pilot):
    for _ in range(100):
        await pilot.pause(.05)
        if not app.busy: return
    raise AssertionError("UI worker did not settle")


def test_tui_ctrl_c_is_the_only_keyboard_quit_binding(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.tui import AdjPartnerApp

    service = AgentService(tmp_path / "state", model=ScriptedPort([]))
    try:
        bindings = {(binding.key, binding.action) for binding in AdjPartnerApp.BINDINGS}
        assert ("ctrl+c", "request_quit") in bindings
        assert not any(key == "ctrl+q" for key, _ in bindings)
    finally:
        service.close()


def test_tui_typed_chat_and_followup_use_one_real_session(tmp_path):
    service, app, sid = build(tmp_path, [ModelReply(content="你好"), ModelReply(content="第二轮")])
    async def scenario():
        async with app.run_test(size=(110, 34)) as pilot:
            await pilot.press("h", "e", "l", "l", "o", "enter")
            await settled(app, pilot)
            assert service.messages(sid)[-1]["content"] == "你好"
            await enter(app, pilot, "followup")
            await settled(app, pilot)
            assert service.messages(sid)[-1]["content"] == "第二轮"
            assert len(service.runs(sid)) == 2
            assert "你好" in app.transcript_text
    asyncio.run(scenario())
    service.close()


def test_tui_queues_followup_while_provider_runs_and_stays_responsive(tmp_path):
    from adj_partner.models import ScriptedPort
    service, app, sid = build(tmp_path, [])
    gate = threading.Event()
    class SlowPort(ScriptedPort):
        def complete(self, *args, **kwargs):
            gate.wait(3)
            return super().complete(*args, **kwargs)
    service.close()
    from adj_partner.application import AgentService
    from adj_partner.tui import AdjPartnerApp
    service = AgentService(tmp_path / "state", model=SlowPort([ModelReply(content="one"), ModelReply(content="two")]))
    app = AdjPartnerApp(service, sid)
    async def scenario():
        async with app.run_test(size=(110, 34)) as pilot:
            await enter(app, pilot, "first")
            assert app.busy
            await enter(app, pilot, "second")
            assert len(app.queued_messages) == 1
            assert len(service.messages(sid)) == 1
            await enter(app, pilot, "/help")
            assert "/cancel" in app.transcript_text and app.busy
            gate.set()
            await settled(app, pilot)
            assert [m["content"] for m in service.messages(sid) if m["role"] == "user"] == ["first", "second"]
    try: asyncio.run(scenario())
    finally: gate.set(); service.close()


def test_tui_actual_user_clarification_resolves_without_new_run(tmp_path):
    service, app, sid = build(tmp_path, [ModelReply(tool_calls=[ToolCall("ask", "ask_user", {"question": "Which sample?"})], finish_reason="tool_calls"), ModelReply(content="answer received")])
    async def scenario():
        async with app.run_test() as pilot:
            await enter(app, pilot, "inspect")
            await settled(app, pilot)
            rid = service.runs(sid)[0]["id"]
            assert service.pending_inputs(rid)
            await enter(app, pilot, "sample A")
            await settled(app, pilot)
            assert service.run(rid)["status"] == "succeeded"
            assert len(service.runs(sid)) == 1 and not service.pending_inputs(rid)
    asyncio.run(scenario())
    service.close()


def test_reopening_waiting_session_shows_real_clarification_without_new_call(tmp_path):
    service, _, sid = build(tmp_path, [ModelReply(tool_calls=[
        ToolCall("ask_reopen", "ask_user", {"question": "UNIQUE_REOPEN_QUESTION"})], finish_reason="tool_calls")])
    rid = service.submit(sid, "question")["id"]
    service.drive(rid)
    pending = service.pending_inputs(rid)[0]
    from adj_partner.tui import AdjPartnerApp
    app = AdjPartnerApp(service, sid)
    async def scenario():
        async with app.run_test() as pilot:
            assert pending["question"] in app.transcript_text and pending["id"] in app.transcript_text
            assert len(service.models.injected.requests) == 1 and not app.busy
            assert service.run(rid)["status"] == "waiting"
    try: asyncio.run(scenario())
    finally: service.close()


def test_reopening_waiting_session_shows_exact_unexecuted_approval_arguments(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.contracts import ToolResult
    from adj_partner.tools import ToolSpec
    from adj_partner.tui import AdjPartnerApp
    effects = []
    service = AgentService(tmp_path / "state", model=ScriptedPort([ModelReply(tool_calls=[
        ToolCall("reopen_approval", "approved_echo", {"value": "UNIQUE_APPROVAL_ARGUMENT"})], finish_reason="tool_calls")]))
    def echo(context, value):
        effects.append(value)
        return ToolResult(data=value)
    service.catalog.register(ToolSpec("approved_echo", "Approval test", {"type": "object", "properties": {
        "value": {"type": "string"}}, "required": ["value"], "additionalProperties": False}, requires_approval=True), echo)
    sid = service.create_session(allowed_tools=service.catalog.names())["id"]
    service.enable_tool(sid, "approved_echo")
    rid = service.submit(sid, "question")["id"]
    service.drive(rid)
    app = AdjPartnerApp(service, sid)
    async def scenario():
        async with app.run_test() as pilot:
            assert "待审批（尚未执行）" in app.transcript_text
            assert "UNIQUE_APPROVAL_ARGUMENT" in app.transcript_text and "reopen_approval" in app.transcript_text
            assert not effects and len(service.models.injected.requests) == 1 and not app.busy
    try: asyncio.run(scenario())
    finally: service.close()


def test_tui_research_free_text_failure_is_visible_not_claimed_as_acceptance(tmp_path):
    service, app, sid = build(tmp_path, [ModelReply(content="unverified conclusion")])
    async def scenario():
        async with app.run_test() as pilot:
            await enter(app, pilot, "/research")
            await enter(app, pilot, "research question")
            await settled(app, pilot)
            assert service.runs(sid)[0]["status"] == "failed"
            assert "acceptance_not_met" in app.transcript_text
    asyncio.run(scenario())
    service.close()


def test_tui_session_reopen_and_trace_export(tmp_path):
    service, app, sid = build(tmp_path, [ModelReply(content="saved answer")])
    rid = service.submit(sid, "before opening")["id"]
    service.drive(rid)
    async def scenario():
        async with app.run_test() as pilot:
            assert "saved answer" in app.transcript_text
            await enter(app, pilot, "/trace " + str(tmp_path / "trace"))
            for _ in range(100):
                await pilot.pause(.05)
                if not app._exporting: break
            from pathlib import Path
            assert app.last_trace and Path(app.last_trace["manifest"]).is_file()
    asyncio.run(scenario())
    service.close()


def test_terminal_projection_neutralizes_control_sequences_without_rewriting_store(tmp_path):
    assert importlib.util.find_spec("adj_partner.tui"), "Conversational Textual client must exist"
    from adj_partner.tui import terminal_text
    raw = "safe\x1b]52;c;secret\x07\x1b[2J\u202eevil\x9dtitle\n[next]"
    rendered = terminal_text(raw)
    assert "\x1b" not in rendered and "\x07" not in rendered and "\u202e" not in rendered and "\x9d" not in rendered
    assert "safe" in rendered and "[next]" in rendered


def test_tui_polling_reads_never_block_input_event_loop(tmp_path, monkeypatch):
    service, app, sid = build(tmp_path, [])
    gate, entered, finished = threading.Event(), threading.Event(), threading.Event()
    calls = []
    original = service.model_info
    def slow_info(identifier):
        calls.append(threading.get_ident())
        entered.set()
        gate.wait(2)
        finished.set()
        return original(identifier)
    async def scenario():
        async with app.run_test(size=(110, 34)) as pilot:
            for _ in range(50):
                if not getattr(app, "_polling", False): break
                await pilot.pause(.02)
            monkeypatch.setattr(service, "model_info", slow_info)
            ui_thread = threading.get_ident()
            app._poll()
            for _ in range(50):
                if entered.is_set(): break
                await pilot.pause(.02)
            assert entered.is_set() and all(thread != ui_thread for thread in calls)
            await enter(app, pilot, "/help")
            assert "/cancel" in app.transcript_text and not finished.is_set()
            gate.set()
    try: asyncio.run(scenario())
    finally: gate.set(); service.close()


def test_tui_state_details_can_scroll_in_a_short_terminal(tmp_path, monkeypatch):
    from textual.containers import VerticalScroll
    service, app, sid = build(tmp_path, [])
    async def scenario():
        async with app.run_test(size=(80, 24)) as pilot:
            for _ in range(50):
                if not getattr(app, "_polling", False): break
                await pilot.pause(.02)
            monkeypatch.setattr(service, "model_info", lambda _: {"model": "long-model-name" * 15, "provider_kind": "scripted"})
            app._poll()
            for _ in range(50):
                await pilot.pause(.02)
                if not app._polling: break
            state = app.query_one("#status-panel", VerticalScroll)
            assert state.allow_vertical_scroll and state.max_scroll_y > 0
            # Textual defers scroll_end until after layout/refresh. Waiting a
            # fixed20ms races that callback on a freshly installed/busy host.
            # A late layout pass can grow max_scroll_y after the scroll lands; scroll again until it settles.
            for _ in range(5):
                scrolled = asyncio.Event()
                state.scroll_end(animate=False, on_complete=scrolled.set)
                await asyncio.wait_for(scrolled.wait(), timeout=2)
                await pilot.pause(.02)
                if state.scroll_y == state.max_scroll_y: break
            assert state.scroll_y == state.max_scroll_y
    try: asyncio.run(scenario())
    finally: service.close()


def test_late_background_snapshot_cannot_replace_the_new_session_view(tmp_path, monkeypatch):
    from textual.widgets import Static
    service, app, sid = build(tmp_path, [])
    other = service.create_session(profile="protein")["id"]
    gate, entered = threading.Event(), threading.Event()
    original = service.model_info
    def slow_old(identifier):
        if identifier == sid:
            entered.set(); gate.wait(2)
        return original(identifier)
    async def scenario():
        async with app.run_test(size=(110, 34)) as pilot:
            for _ in range(50):
                if not app._polling: break
                await pilot.pause(.02)
            monkeypatch.setattr(service, "model_info", slow_old)
            app._poll()
            for _ in range(50):
                if entered.is_set(): break
                await pilot.pause(.02)
            assert entered.is_set()
            app._load_session(other)
            gate.set()
            for _ in range(100):
                await pilot.pause(.02)
                if not app._polling and "Profile: protein" in str(app.query_one("#state", Static).content): break
            content = str(app.query_one("#state", Static).content)
            assert app.session_id == other and "Profile: protein" in content and sid not in content
    try: asyncio.run(scenario())
    finally: gate.set(); service.close()


def test_tui_approval_requires_explicit_host_command(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.contracts import ToolResult
    from adj_partner.tools import ToolSpec
    from adj_partner.tui import AdjPartnerApp
    effects = []
    service = AgentService(tmp_path / "state", model=ScriptedPort([
        ModelReply(tool_calls=[ToolCall("approved_call", "approved_echo", {})], finish_reason="tool_calls"), ModelReply(content="executed")]))
    def echo(context):
        effects.append("effect")
        return ToolResult(data="ok")
    service.catalog.register(ToolSpec("approved_echo", "Approved test effect", {"type": "object", "properties": {}, "additionalProperties": False}, requires_approval=True), echo)
    sid = service.create_session(allowed_tools=service.catalog.names())["id"]
    service.enable_tool(sid, "approved_echo")
    app = AdjPartnerApp(service, sid)
    async def scenario():
        async with app.run_test() as pilot:
            await enter(app, pilot, "perform test")
            await settled(app, pilot)
            assert service.approvals(sid) and not effects
            assert "approved_call" in app.transcript_text
            await enter(app, pilot, "yes")
            assert not effects and service.approvals(sid)
            await enter(app, pilot, "/approve approved_call")
            await settled(app, pilot)
            assert effects == ["effect"] and service.runs(sid)[0]["status"] == "succeeded"
    asyncio.run(scenario())
    service.close()


def test_tui_cancel_stays_responsive_and_retains_queued_draft(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.tui import AdjPartnerApp
    gate, entered = threading.Event(), threading.Event()
    class SlowPort(ScriptedPort):
        def complete(self, *args, **kwargs):
            entered.set()
            gate.wait(5)
            return super().complete(*args, **kwargs)
    port = SlowPort([ModelReply(content="not delivered"), ModelReply(content="not auto started")])
    service = AgentService(tmp_path / "state", model=port)
    sid = service.create_session()["id"]
    app = AdjPartnerApp(service, sid)
    async def scenario():
        async with app.run_test() as pilot:
            await enter(app, pilot, "first")
            for _ in range(50):
                if entered.is_set(): break
                await pilot.pause(.02)
            assert entered.is_set()
            await enter(app, pilot, "second")
            await pilot.press("ctrl+k")
            await pilot.pause(.05)
            assert service.runs(sid)[0]["status"] == "cancelled" and app.busy
            gate.set()
            await settled(app, pilot)
            assert len(app.queued_messages) == 1 and len(service.runs(sid)) == 1
            await enter(app, pilot, "/quit")
            assert not app._client_detaching and len(app.queued_messages) == 1
            await enter(app, pilot, "/queue clear")
            assert not app.queued_messages
    try: asyncio.run(scenario())
    finally: gate.set(); service.close()


def test_tui_quit_detaches_without_new_tool_effect_or_hidden_model_loop(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.tui import AdjPartnerApp
    gate, entered = threading.Event(), threading.Event()
    class SlowPort(ScriptedPort):
        def complete(self, *args, **kwargs):
            entered.set()
            gate.wait(5)
            return super().complete(*args, **kwargs)
    port = SlowPort([ModelReply(tool_calls=[ToolCall("plan", "update_plan", {"steps": [{"step": "must wait for reconnect", "status": "pending"}]})], finish_reason="tool_calls")])
    service = AgentService(tmp_path / "state", model=port)
    sid = service.create_session()["id"]
    app = AdjPartnerApp(service, sid)
    async def scenario():
        async with app.run_test() as pilot:
            await enter(app, pilot, "long task")
            for _ in range(50):
                if entered.is_set(): break
                await pilot.pause(.02)
            assert entered.is_set()
            app.action_request_quit()
            gate.set()
    try:
        asyncio.run(scenario())
        rid = service.runs(sid)[0]["id"]
        for _ in range(100):
            if service.run(rid)["status"] == "waiting": break
            time.sleep(.02)
        assert service.run(rid)["status"] == "waiting"
        assert not service.actions(rid) and len(port.requests) == 1
    finally: gate.set(); service.close()


def test_late_poll_during_widget_teardown_is_discarded_before_app_unmount(tmp_path, monkeypatch):
    from adj_partner.tui import _PollFinished, _DriverFinished, _ExportFinished
    from textual.widgets import Static
    service, app, sid = build(tmp_path, [])
    observed = []
    snapshot = {'session': service.session(sid), 'model': service.model_info(sid)}
    close_all = app._close_all
    async def close_with_late_snapshots():
        # Textual stops the message pump before pruning children, and dispatches
        # App.Unmount only afterwards. A completed worker may arrive in this gap.
        assert not app.is_running and not app._client_detaching
        try:
            await app.query_one('#state', Static).remove()
            transcript = app.transcript_text
            app.on__poll_finished(_PollFinished(sid, None, snapshot=snapshot))
            app.on__poll_finished(_PollFinished(sid, None, error='late service failure'))
            app.on__driver_finished(_DriverFinished(None, error='late provider completion'))
            app.on__export_finished(_ExportFinished(error='late export completion'))
            assert app.transcript_text == transcript
            observed.append('late snapshots discarded')
        finally:
            await close_all()
    monkeypatch.setattr(app, '_close_all', close_with_late_snapshots)
    async def scenario():
        async with app.run_test(size=(110, 34)) as pilot:
            await pilot.pause(.05)
    try:
        asyncio.run(scenario())
        assert observed == ['late snapshots discarded']
        assert not app._polling
    finally: service.close()


def test_poll_errors_are_still_visible_while_ui_is_running(tmp_path):
    from adj_partner.tui import _PollFinished
    from textual.widgets import Static
    service, app, sid = build(tmp_path, [])
    async def scenario():
        async with app.run_test(size=(110, 34)) as pilot:
            for _ in range(50):
                if not app._polling: break
                await pilot.pause(.02)
            app.on__poll_finished(_PollFinished(sid, None, error='actual storage read failed'))
            assert 'actual storage read failed' in str(app.query_one('#state', Static).content)
    try: asyncio.run(scenario())
    finally: service.close()
