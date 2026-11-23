"""Independent conversational Textual client of the one AgentService.

No provider/tool decisions live here. Draft input is visibly unsent client state;
sessions, runs, tool outcomes and obligations always come from the service.
"""

import argparse
from collections import deque
from html.parser import HTMLParser
import json
import shlex
from threading import Event
import unicodedata

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, Input, RichLog, Static

from .storage import TERMINAL_STATUSES
from .profiles import PROFILES
from .defaults import DEFAULT_PROFILE
from .research.acceptance import RESPONSE_SCHEMA, research_acceptance


HELP = f"""输入自然语言开始对话；当前会话会保留上下文。
/help                         显示帮助
/new [{'|'.join(PROFILES)}] [--allow-root PATH] [--network]
/sessions  /session ID         列出/重开会话
/tools  /enable NAME  /disable NAME   管理已获授权的会话工具
/profile [NAME]  /model [NAME]  查看/修改（仅运行结束后）
/research [验收JSON]            下一轮起严格报告验收；不等于科学实证
/chat                         下一轮起普通聊天
/continue                     恢复本会话未结束的运行
/answer [REQUEST_ID] 回答       回答真实待确认问题（也可直接输入）
/approve [CALL_ID] /deny [CALL_ID]    对精确操作明确审批
/jobs  /cancel [RUN_ID]        查看任务/请求取消（远端效果可能未停止）
/trace [DIRECTORY]             导出完整公开行动记录与可公开工件
/queue  /next  /queue clear    查看/提交下一条/明确丢弃未提交草稿
/quit                         退出客户端；独立任务不自动取消

运行中输入的跟进内容排队为“未提交草稿”，不会插入工具调用组。
草稿仅保存在当前客户端；强制关闭会丢失。正常退出需明确确认丢弃。
等待审批时不能用普通“yes”代替 /approve。等待澄清时直接输入答案。
已开始的远端请求有超时边界，但退出/取消不保证远端停止或免计费。
"""


def terminal_text(value):
    """Literal display projection: neutralize terminal controls and bidi formats."""
    text = str(value)
    return "".join("    " if char == "\t" else char if char == "\n" or unicodedata.category(char) not in {"Cc", "Cf", "Cs"}
                   else f"⟪U+{ord(char):04X}⟫" for char in text)


def acceptance_label(acceptance):
    if research_acceptance(acceptance): return "严格报告"
    if isinstance(acceptance, dict) and RESPONSE_SCHEMA in acceptance: return "JSON Schema"
    return "普通聊天"


class _ReportText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"head", "script", "style"}: self.hidden += 1
        elif not self.hidden and tag in {"p", "div", "section", "article", "h1", "h2", "h3", "li", "br", "tr"}: self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"head", "script", "style"}: self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in {"p", "li", "h1", "h2", "h3", "tr"}: self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)


def conversation_text(content):
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    clipped = len(text) > 40000
    text = text[:40000]
    if text.lstrip().lower().startswith(("<!doctype html", "<html")):
        parser = _ReportText()
        parser.feed(text)
        text = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    if clipped: text += "\n[界面预览已截断；完整内容见 /trace 和原始工件]"
    return text


class _DriverFinished(Message):
    def __init__(self, run_id, result=None, error=None):
        self.run_id, self.result, self.error = run_id, result, error
        super().__init__()


class _ExportFinished(Message):
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error
        super().__init__()


class _PollFinished(Message):
    def __init__(self, session_id, run_id, snapshot=None, error=None):
        self.session_id, self.run_id = session_id, run_id
        self.snapshot, self.error = snapshot, error
        super().__init__()


class AdjPartnerApp(App):
    TITLE = "Adj Partner · Scientific Harness"
    SUB_TITLE = "会话驱动 · 共享工具 · 可恢复任务"
    CSS = """
    Screen { background: $surface; }
    #workspace { height: 1fr; }
    #conversation { width: 3fr; border: round $primary; padding: 0 1; }
    #sidebar { width: 36; min-width: 24; }
    #status-panel { height: auto; max-height: 16; padding: 1; border: round $secondary; }
    #state { height: auto; }
    #activity { height: 1fr; border: round $secondary; padding: 0 1; }
    #mode { height: auto; padding: 0 1; color: $text-muted; }
    #prompt { dock: bottom; margin: 0 1; }
    """
    BINDINGS = [Binding("ctrl+c", "request_quit", "退出", priority=True),
                Binding("ctrl+k", "cancel_run", "取消运行", priority=True),
                Binding("f1", "show_help", "帮助", priority=True)]

    def __init__(self, service, session_id=None):
        super().__init__()
        self.service, self.session_id = service, session_id
        self.run_id, self.busy = None, False
        self.queued_messages = deque()
        self.acceptance = None
        self.stop_event = Event()
        self.transcript_text = ""
        self._message_count, self._seen_actions, self._event_cursor = 0, {}, 0
        self._client_detaching, self._exporting, self._polling = False, False, False
        self.last_trace = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="workspace"):
            yield RichLog(id="conversation", wrap=True, markup=False, highlight=False, max_lines=10000)
            with Vertical(id="sidebar"):
                with VerticalScroll(id="status-panel"):
                    yield Static("", id="state", markup=False)
                yield RichLog(id="activity", wrap=True, min_width=20, markup=False, max_lines=2000)
        yield Static("", id="mode", markup=False)
        yield Input(placeholder="输入问题或 /help；Enter 发送", id="prompt", max_length=100000)
        yield Footer()

    def on_mount(self):
        if self.session_id is None: self.session_id = self.service.create_session()["id"]
        self._load_session(self.session_id)
        self._write("系统", "这是独立 Textual 界面，不是 Codex 内嵌 TUI。/help 查看命令；/research 开启严格报告验收。")
        self._write("系统", f"退出后可用 tui --session {self.session_id} 重连；独立任务不会自动取消。")
        self.query_one("#prompt", Input).focus()
        self.set_interval(.5, self._poll)

    def on_unmount(self):
        self._client_detaching = True
        self.stop_event.set()

    def _write(self, label, value, *, activity=False):
        if self._client_detaching or not self.is_running: return
        if not isinstance(value, str): value = json.dumps(value, ensure_ascii=False, indent=2)
        # The actual Store holds current host-configured secret material. This
        # is a display projection, never an alternate scientific truth store.
        clean = terminal_text(self.service.store.redact(value))
        if len(clean) > 50000: clean = clean[:50000] + "\n[界面已截断，完整记录见 /trace]"
        line = f"{label}\n{clean}\n"
        self.query_one("#activity" if activity else "#conversation", RichLog).write(Text(line))
        self.transcript_text = (self.transcript_text + line)[-500000:]

    def _load_session(self, identifier):
        self.service.session(identifier)
        self.session_id, self._message_count, self._seen_actions, self._event_cursor = identifier, 0, {}, 0
        self.query_one("#conversation", RichLog).clear()
        self.query_one("#activity", RichLog).clear()
        self.query_one("#state", Static).update(Text("读取会话状态…\n" + terminal_text(identifier)))
        self.transcript_text = ""
        runs = self.service.runs(identifier)
        unfinished = [run for run in runs if run["status"] not in TERMINAL_STATUSES]
        self.run_id = (unfinished[-1] if unfinished else runs[-1])["id"] if runs else None
        self._messages()
        self._poll()
        if unfinished:
            self._write("恢复", "本会话有未结束运行。/continue 恢复；审批、澄清和原始工具调用ID仍保留。")
            self._show_obligations(self.run_id)

    def _messages(self):
        messages = self.service.messages(self.session_id)
        start = max(self._message_count, len(messages) - 200)
        if start > self._message_count: self._write("历史记录", "界面仅展开最近200条；完整历史与工具结果保存在 /trace。")
        for message in messages[start:]:
            role = message.get("role")
            if role == "user": self._write("你", conversation_text(message.get("content", "")))
            elif role == "assistant" and message.get("content"):
                self._write("助手", conversation_text(message["content"]))
        self._message_count = len(messages)

    def _poll(self):
        if self._client_detaching or not self.is_running or self.session_id is None or self._polling: return
        self._polling = True
        self._refresh(self.session_id, self.run_id, self._event_cursor)

    @work(thread=True, exit_on_error=False)
    def _refresh(self, session_id, run_id, after_seq):
        # A single coalesced read worker. This never drives/resumes a model or
        # job; late snapshots are discarded when the UI changes session/run.
        try:
            snapshot = {"session": self.service.session(session_id), "model": self.service.model_info(session_id)}
            if run_id:
                run = self.service.run(run_id)
                snapshot.update(run=run, task=self.service.task(run["task_id"]),
                    activity=self.service.activity_after(session_id, run_id, after_seq=after_seq))
            self.post_message(_PollFinished(session_id, run_id, snapshot))
        except Exception as exc:
            self.post_message(_PollFinished(session_id, run_id, error=str(exc)))

    def on__poll_finished(self, message: _PollFinished):
        self._polling = False
        # Textual stops the App message pump before removing child widgets;
        # App.Unmount (and our detach flag) arrives only after that teardown.
        if self._client_detaching or not self.is_running: return
        if message.session_id != self.session_id or message.run_id != self.run_id:
            self._poll()
            return
        if message.error:
            self.query_one("#state", Static).update(Text(terminal_text(self.service.store.redact(message.error))))
            return
        try:
            snapshot = message.snapshot
            session, model = snapshot["session"], snapshot["model"]
            lines = ["SESSION", self.session_id, f"Profile: {session['config'].get('profile', 'general')}",
                     f"Model: {model.get('model') or '(未配置)'}", f"Provider: {model.get('provider_kind', 'configured')}"]
            if snapshot.get("run"):
                run, task, activity = snapshot["run"], snapshot["task"], snapshot["activity"]
                usage = run.get("usage", {})
                lines += ["", f"Run: {run['status']}", f"Steps: {run['state'].get('steps', 0)}/{run['budget']['max_steps']}",
                          f"Tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out"]
                lines.append("当前任务验收: " + acceptance_label(task["acceptance"]))
                if usage.get("unknown_requests"): lines.append(f"Usage unknown: {usage['unknown_requests']} requests")
                if run["state"].get("waiting"): lines.append("Waiting: " + run["state"]["waiting"].get("kind", "unknown"))
                for action in activity["actions"]:
                    signature = (action["status"], action.get("finished_at"))
                    if self._seen_actions.get(action["call_id"]) != signature:
                        self._seen_actions[action["call_id"]] = signature
                        self._write("工具", f"{action['tool_name']} · {action['call_id']} · {action['status']}", activity=True)
                        if action.get("error"):
                            self._write("工具结果", action["error"], activity=True)
                for event in activity["events"]:
                    if event["seq"] <= self._event_cursor: continue
                    if event["kind"] == "job.progress": self._write("任务进度", event["payload"], activity=True)
                self._event_cursor = activity["next_cursor"]
                lines.append(f"Active jobs: {activity['active_job_count']}")
                for job in activity["active_jobs"]: lines.append(f"{job['tool_name']}: {job['status']}")
                if activity["actions_truncated"]: lines.append("工具状态仅最近100条；完整记录见 /trace")
            else: lines.append("Active jobs: 0")
            self.query_one("#state", Static).update(Text(terminal_text("\n".join(lines))))
            mode = "下一轮: 严格研究验收（结构通过≠科学实证）" if self.acceptance else "下一轮: 普通聊天（未做科学验收）"
            self.query_one("#mode", Static).update(Text(f"{mode} | {'运行中' if self.busy else '可输入'} | 未提交草稿 {len(self.queued_messages)}"))
        except Exception as exc:
            self.query_one("#state", Static).update(Text(terminal_text(self.service.store.redact(str(exc)))))

    def _outstanding(self):
        runs = self.service.runs(self.session_id)
        return next((run for run in reversed(runs) if run["status"] not in TERMINAL_STATUSES), None)

    def _start(self, text, acceptance=None):
        run = self.service.submit(self.session_id, text, acceptance=acceptance)
        self.run_id = run["id"]
        self._messages()
        self._launch(run["id"])

    def _launch(self, run_id):
        if self.busy: raise ValueError("当前客户端已有运行，不能启动第二个模型循环")
        if self.service.run(run_id)["session_id"] != self.session_id: raise ValueError("运行不属于本会话")
        self.run_id, self.busy, self.stop_event = run_id, True, Event()
        self._drive(run_id, self.stop_event)
        self._poll()

    @work(thread=True, exit_on_error=False)
    def _drive(self, run_id, stop_event):
        try: self.post_message(_DriverFinished(run_id, self.service.drive(run_id, stop_event=stop_event)))
        except Exception as exc: self.post_message(_DriverFinished(run_id, error=str(exc)))

    def on__driver_finished(self, message: _DriverFinished):
        self.busy = False
        if self._client_detaching or not self.is_running: return
        self._messages()
        if message.error:
            self._write("运行未完成", message.error + "\n状态已保留；检查 /continue、/jobs 或 /trace。")
        else:
            run = message.result
            reason = run["state"].get("stop_reason") or run["state"].get("waiting", {}).get("kind", "")
            self._write("运行状态", f"{run['status']} · {reason}\nrun_id: {run['id']}")
            delivery = run["state"].get("delivery", {})
            if delivery.get("kind") == "accepted_report":
                self._write("严格报告", f"结构化验收通过，不代表科学结论已验证。\nartifact: {delivery['artifact_id']}\nSHA256: {delivery['sha256']}")
            if run["status"] == "waiting": self._show_obligations(run["id"])
        self._poll()
        if not message.error and message.result["status"] == "succeeded" and self.queued_messages:
            text, acceptance = self.queued_messages[0]
            try:
                self._start(text, acceptance)
                self.queued_messages.popleft()
            except Exception as exc: self._write("草稿尚未提交", str(exc))
        elif self.queued_messages:
            self._write("草稿保留", "当前运行未成功结束，未自动提交后续草稿。处理当前状态后用 /next 或 /queue clear。")

    def _show_obligations(self, run_id):
        for item in self.service.pending_inputs(run_id):
            self._write("需要你的回答", f"{item['question']}\nrequest_id: {item['id']}\n可直接输入答案。")
        for item in self.service.approvals(self.session_id):
            if item["run_id"] == run_id:
                self._write("待审批（尚未执行）", {"call_id": item["call_id"], "tool": item["tool_name"], "arguments": item["args"],
                    "next": f"/approve {item['call_id']} 或 /deny {item['call_id']}"})

    def on_input_submitted(self, event: Input.Submitted):
        text = event.value.strip()
        if not text: return
        event.input.value = ""
        try:
            if text.startswith("/"): self._command(text)
            elif self.busy:
                if len(self.queued_messages) >= 20: raise ValueError("草稿队列已达20条；输入保留，请稍后提交")
                self.queued_messages.append((text, json.loads(json.dumps(self.acceptance))))
                self._write("未提交草稿", text + "\n将在当前运行结束后提交；只保存在此客户端。")
            else:
                outstanding = self._outstanding()
                if outstanding:
                    pending = self.service.pending_inputs(outstanding["id"])
                    if len(pending) == 1:
                        self.service.answer(outstanding["id"], pending[0]["id"], text)
                        self._write("你的澄清回答", text)
                        self._launch(outstanding["id"])
                    else: raise ValueError("本会话仍有未结束运行；请先 /continue、/approve、/deny 或 /cancel。普通文字不是审批。")
                elif self.queued_messages: raise ValueError("仍有未提交草稿；先 /next 按顺序提交或 /queue clear")
                else: self._start(text, self.acceptance)
        except Exception as exc:
            self._write("未提交/操作失败", str(exc))
            event.input.value = text
        self._poll()

    def _command(self, text):
        command, _, rest = text.partition(" ")
        rest = rest.strip()
        if command == "/help": self.action_show_help()
        elif command == "/quit": self._quit(discard=rest == "discard")
        elif command == "/next":
            if self.busy or self._outstanding(): raise ValueError("请先处理当前未结束运行")
            if not self.queued_messages: raise ValueError("没有未提交草稿")
            value, acceptance = self.queued_messages[0]
            self._start(value, acceptance)
            self.queued_messages.popleft()
        elif command == "/queue":
            if rest == "clear":
                count = len(self.queued_messages)
                self.queued_messages.clear()
                self._write("草稿", f"按你的命令丢弃{count}条未提交草稿；已保存的会话历史未删除。")
            elif not rest: self._write("未提交草稿", [item[0] for item in self.queued_messages])
            else: raise ValueError("使用 /queue 或 /queue clear")
        elif command in {"/new", "/session"}:
            if self.busy or self.queued_messages: raise ValueError("运行中或仍有未提交草稿时不能切换会话")
            if command == "/new":
                parser = argparse.ArgumentParser(add_help=False, exit_on_error=False)
                parser.add_argument("profile", nargs="?", default=DEFAULT_PROFILE, choices=list(PROFILES))
                parser.add_argument("--allow-root", action="append", default=[])
                parser.add_argument("--network", action="store_true")
                try: args, extra = parser.parse_known_args(shlex.split(rest))
                except (argparse.ArgumentError, SystemExit) as exc: raise ValueError("无效 /new 参数，参见 /help") from exc
                if extra: raise ValueError("未知 /new 参数: " + " ".join(extra))
                identifier = self.service.create_session(profile=args.profile, allowed_roots=args.allow_root, allow_network=args.network)["id"]
            else: identifier = rest
            self.acceptance = None
            self._load_session(identifier)
        elif command == "/sessions": self._write("会话", [{"id": s["id"], "profile": s["config"].get("profile")} for s in self.service.sessions()])
        elif command == "/tools": self._write("会话工具", self.service.tools(self.session_id))
        elif command in {"/enable", "/disable"}:
            method = self.service.enable_tool if command == "/enable" else self.service.disable_tool
            self._write("工具配置", method(self.session_id, rest))
        elif command == "/profile":
            self._write("Profile", self.service.set_profile(self.session_id, rest) if rest else self.service.session(self.session_id)["config"]["profile"])
        elif command == "/model":
            self._write("Model", self.service.set_model(self.session_id, rest) if rest else self.service.model_info(self.session_id))
        elif command == "/research":
            from .research.acceptance import requirements
            criteria = json.loads(rest) if rest else {"require_report": True, "require_evidence": True, "require_limitations": True}
            requirements(criteria)
            if not criteria: raise ValueError("严格模式需要非空验收条件；普通聊天使用 /chat")
            self.acceptance = criteria
            self._write("后续新任务验收", criteria)
        elif command == "/chat":
            self.acceptance = None
            self._write("模式", "后续新任务使用普通聊天；当前运行及已排队草稿的验收条件不变。")
        elif command == "/continue":
            run = self._outstanding()
            if run is None: raise ValueError("没有可恢复的未结束运行")
            self._launch(run["id"])
        elif command == "/answer":
            if self.busy: raise ValueError("请等当前调用返回真实澄清请求后再回答")
            run = self._outstanding()
            if run is None: raise ValueError("没有待回答运行")
            pending = self.service.pending_inputs(run["id"])
            request_id, _, answer = rest.partition(" ")
            if request_id not in {item["id"] for item in pending}:
                if len(pending) != 1: raise ValueError("请给出准确 REQUEST_ID 和答案")
                request_id, answer = pending[0]["id"], rest
            self.service.answer(run["id"], request_id, answer)
            self._write("你的澄清回答", answer)
            self._launch(run["id"])
        elif command in {"/approve", "/deny"}:
            if self.busy: raise ValueError("请等当前循环显示待审批状态")
            run = self._outstanding()
            if run is None: raise ValueError("没有待审批运行")
            pending = [a for a in self.service.approvals(self.session_id) if a["run_id"] == run["id"]]
            if rest: pending = [a for a in pending if a["call_id"] == rest]
            if len(pending) != 1: raise ValueError("请给出准确 CALL_ID；没有唯一匹配的待审批操作")
            self._write("审批决定", self.service.decide_approval(self.session_id, run["id"], pending[0]["call_id"], command == "/approve"))
            self._launch(run["id"])
        elif command == "/jobs": self._write("任务", self.service.jobs(self.session_id))
        elif command == "/cancel": self._cancel(rest or None)
        elif command == "/trace":
            if self._exporting: raise ValueError("已有导出进行中")
            self._exporting = True
            self._export(self.session_id, rest or None)
        else: raise ValueError("未知命令，使用 /help；命令不会作为模型指令发送")

    @work(thread=True, exit_on_error=False)
    def _export(self, session_id, destination):
        try: self.post_message(_ExportFinished(self.service.trace(session_id, destination)))
        except Exception as exc: self.post_message(_ExportFinished(error=str(exc)))

    def on__export_finished(self, message: _ExportFinished):
        self._exporting = False
        if not message.error: self.last_trace = message.result
        if not self._client_detaching: self._write("Trace 导出失败" if message.error else "Trace 已保存", message.error or message.result)

    def _cancel(self, run_id=None):
        if run_id is None:
            run = self._outstanding()
            if run is None: raise ValueError("没有未结束运行")
            run_id = run["id"]
        if self.service.run(run_id)["session_id"] != self.session_id: raise ValueError("只能取消当前会话的运行")
        result = self.service.cancel(run_id)
        self._write("取消请求已记录", f"{result['status']} · {run_id}\n已开始的远端操作可能继续/计费；/jobs 查看本地任务确认状态。")
        self._poll()

    def action_cancel_run(self):
        try: self._cancel()
        except Exception as exc: self._write("取消未执行", str(exc))

    def action_show_help(self): self._write("帮助", HELP)

    def _quit(self, discard=False):
        if self.queued_messages and not discard:
            self._write("还有未提交草稿", "先 /queue 查看；/queue clear 或 /quit discard 明确丢弃。已保存会话和独立任务不受影响。")
            return
        if discard: self.queued_messages.clear()
        self.stop_event.set()
        self._client_detaching = True
        self.exit(self.session_id)

    def action_request_quit(self): self._quit()
