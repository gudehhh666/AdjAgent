"""Diagnostic trusted LOCAL job handlers; these are NOT scientific backends."""

import json
import os
from pathlib import Path
import signal
import sys
import time

from adj_partner.contracts import ToolResult


def encoded_credential_payload(variant, value):
    """Diagnostic serialization variants whose full payload Store can redact."""
    assignment = 'password="' + value + '"'
    if variant == "nested":
        return json.dumps({"message": json.dumps({"message": assignment})})
    if variant == "nested_key":
        return json.dumps({"message": json.dumps({"password": value})})
    if variant in {"token", "unicode_key"}:
        payload = json.dumps({"token": value})
        return payload.replace('"token"', r'"t\u006fken"') if variant == "unicode_key" else payload
    return json.dumps({"message": assignment})


def exercise(context, mode="echo", text="hello", duration=0.0):
    """Use deterministic bounded diagnostics to exercise process lifecycle edges."""
    context.check_cancelled()
    marker = context.workspace / "invocations.txt"
    with marker.open("a") as output:
        output.write(f"{os.getpid()}\n")
    (context.workspace / "child.pid").write_text(str(os.getpid()))
    if mode == "crash":
        os._exit(23)
    if mode == "supervisor_crash":
        os.kill(os.getppid(), signal.SIGKILL)
        time.sleep(0.3)
        return ToolResult(data={"text": text})
    if mode == "cpu":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop = time.monotonic() + duration
        while time.monotonic() < stop:
            pass
    elif duration:
        time.sleep(duration)
    if mode == "malformed":
        return {"status": "ok", "data": "not ToolResult"}
    if mode == "nan":
        return ToolResult(data=float("nan"))
    if mode == "surrogate":
        return ToolResult(data="\ud800")
    if mode == "pending":
        return ToolResult(status="pending", job_id="nested")
    if mode == "bad_envelope":
        return ToolResult(status="ok", error={"code": "no", "message": "inconsistent"})
    if mode == "bad_schema":
        return ToolResult(data=17)
    if mode == "fake_artifact":
        return ToolResult(data={"text": text}, artifacts=[{"id": "artifact_not_real"}])
    if mode == "foreign_artifact":
        return ToolResult(data={"text": "foreign"}, artifacts=[{"id": text}])
    if mode == "metadata_expansion":
        from adj_partner.execution.manager import MAX_RESULT_BYTES
        artifact = context.store.put_artifact(
            b"genuine diagnostic bytes", "metadata-diagnostic.txt",
            provenance={"diagnostic_note": "x" * MAX_RESULT_BYTES},
            session_id=context.session_id, run_id=context.run_id)
        return ToolResult(data={"text": "small result"}, artifacts=[{"id": artifact["id"]}])
    if mode == "redaction_expansion":
        from adj_partner.execution.manager import MAX_RESULT_BYTES
        context.store.register_secrets(["qz"])
        return ToolResult(data="qz" * (MAX_RESULT_BYTES // 3))
    if mode == "truncated_assignment":
        for descriptor, label in ((1, "stdout"), (2, "stderr")):
            quote = b'"' if descriptor == 1 else b"'"
            os.write(descriptor, f"ordinary {label} prefix\n".encode() +
                     b"password=" + quote + b"QUALITY_SYNTHETIC_PASSWORD_VALUE_" +
                     b"x" * 40000 + quote + b"\n")
        return ToolResult(data={"text": "complete"})
    if mode == "escaped_assignment":
        payload = encoded_credential_payload(text, "QUALITY_SYNTHETIC_ESCAPED_VALUE_" + "x" * 40000)
        for descriptor, label in ((1, "stdout"), (2, "stderr")):
            os.write(descriptor, f"ordinary {label} prefix\n".encode() +
                     b'{"note": "completed ordinary record"}\n' + payload.encode())
        return ToolResult(data={"text": "complete"})
    if mode == "ordinary_truncated":
        for descriptor, label in ((1, "stdout"), (2, "stderr")):
            os.write(descriptor, f"ordinary {label} prefix\n".encode() + b"ordinary data " * 5000)
        return ToolResult(data={"text": "complete"})
    if mode == "read_path":
        source = context.check_path(text)
        try:
            context.check_path(text, write=True)
        except PermissionError:
            write_denied = True
        else:
            write_denied = False
        return ToolResult(data={"text": source.read_text(), "write_denied": write_denied})
    if mode == "exception":
        raise RuntimeError("API_KEY=sk-diagnosticsecret999")
    if mode == "env":
        return ToolResult(data={"env": dict(os.environ), "cwd": str(Path.cwd())})
    if mode == "paths":
        denied = []
        for path in (context.store.db_path, context.store.root / "jobs" / "job_other" / "file"):
            try:
                context.check_path(path, must_exist=False)
            except PermissionError:
                denied.append(str(path))
        own = context.check_path(context.workspace / "own.txt", write=True, must_exist=False)
        return ToolResult(data={"denied": denied, "own": str(own)})
    context.progress("diagnostic progress", note="Bearer diagnosticsecretvalue123")
    if mode == "log":
        os.write(1, b"API_KEY=sk-diagnosticsecret999\n" + b"x" * 100000)
        os.write(2, b"password=notpublic\n" + b"y" * 100000)
        for _ in range(150):
            context.progress("bounded" * 1000, details="z" * 20000)
    else:
        print("diagnostic stdout", flush=True)
        print("diagnostic stderr", file=sys.stderr, flush=True)
    artifact = context.store.put_artifact(
        b"exact diagnostic bytes", "diagnostic.txt", provenance={"job_id": context.job_id},
        session_id=context.session_id, run_id=context.run_id)
    return ToolResult(data={"text": text}, artifacts=[artifact])
