import os
from pathlib import Path
import json
import signal
import subprocess
import sys
import time

import pytest


requires_nested_seatbelt = pytest.mark.skipif(
    os.environ.get("CODEX_SANDBOX") == "seatbelt",
    reason="macOS forbids applying a nested Seatbelt profile inside the Codex Seatbelt",
)


def workspace(tmp_path):
    root = tmp_path / "work"
    for name in ("inputs", "outputs", "tmp"):
        (root / name).mkdir(parents=True)
    return root


def write_program(root, source):
    path = root / "program.py"
    path.write_text(source, encoding="utf-8")
    return path


def run_python(sandbox, root, source, timeout=20):
    program = write_program(root, source)
    return sandbox.run([sys.executable, "-I", str(program)], root, timeout_seconds=timeout)


def process_exists(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_file(path, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        time.sleep(.02)
    pytest.fail(f"Timed out waiting for {path.name}")


def wait_for_process_exit(pid, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not process_exists(pid):
            return
        time.sleep(.02)
    pytest.fail(f"Process {pid} survived analysis cleanup")


@requires_nested_seatbelt
@pytest.mark.real_sandbox
def test_real_sandbox_allows_reading_workspace_and_writing_outputs(tmp_path):
    from adj_partner.analysis.sandbox import select_sandbox

    root = workspace(tmp_path)
    (root / "inputs" / "data.txt").write_text("three", encoding="utf-8")
    result = run_python(select_sandbox(), root,
        "from pathlib import Path\n"
        "value=Path('inputs/data.txt').read_text()\n"
        "Path('outputs/result.txt').write_text(value)\n")

    assert result.returncode == 0, result.stderr
    assert (root / "outputs" / "result.txt").read_text() == "three"
    assert result.backend in {"macos-sandbox-exec", "linux-bubblewrap"}


@requires_nested_seatbelt
@pytest.mark.real_sandbox
def test_real_sandbox_denies_read_and_write_outside_workspace(tmp_path):
    from adj_partner.analysis.sandbox import select_sandbox

    root = workspace(tmp_path)
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("must-not-leak", encoding="utf-8")
    outside = tmp_path / "outside-write.txt"
    host_account_file = Path("/etc/passwd")
    source = (
        "from pathlib import Path\n"
        f"secret=Path({str(secret)!r})\n"
        f"outside=Path({str(outside)!r})\n"
        f"host_account_file=Path({str(host_account_file)!r})\n"
        "results=[]\n"
        "for operation in (lambda:secret.read_text(),lambda:outside.write_text('bad'),lambda:host_account_file.read_text()):\n"
        "    try: operation(); results.append('allowed')\n"
        "    except OSError: results.append('denied')\n"
        "Path('outputs/probe.txt').write_text(','.join(results))\n")

    result = run_python(select_sandbox(), root, source)

    assert result.returncode == 0, result.stderr
    assert (root / "outputs" / "probe.txt").read_text() == "denied,denied,denied"
    assert not outside.exists()


@requires_nested_seatbelt
@pytest.mark.real_sandbox
def test_real_sandbox_denies_network_and_drops_parent_secrets(tmp_path, monkeypatch):
    from adj_partner.analysis.sandbox import select_sandbox

    monkeypatch.setenv("ADJ_LLM_API_KEY", "synthetic-parent-secret")
    root = workspace(tmp_path)
    source = (
        "import os,socket\n"
        "from pathlib import Path\n"
        "visible='synthetic-parent-secret' in repr(dict(os.environ))\n"
        "try:\n"
        "    socket.socket().connect(('127.0.0.1',9)); network='allowed'\n"
        "except OSError:\n"
        "    network='denied'\n"
        "Path('outputs/probe.txt').write_text(f'{visible},{network}')\n")

    result = run_python(select_sandbox(), root, source)

    assert result.returncode == 0, result.stderr
    assert (root / "outputs" / "probe.txt").read_text() == "False,denied"
    assert "synthetic-parent-secret" not in result.stdout + result.stderr


@requires_nested_seatbelt
@pytest.mark.real_sandbox
def test_sandbox_times_out_and_bounds_captured_streams(tmp_path):
    from adj_partner.analysis.sandbox import select_sandbox

    root = workspace(tmp_path)
    result = run_python(select_sandbox(), root,
        "import sys,time\nprint('x'*100000)\nsys.stderr.write('y'*100000)\ntime.sleep(10)\n", timeout=1)

    assert result.timed_out is True
    assert len(result.stdout.encode("utf-8")) <= 32768
    assert len(result.stderr.encode("utf-8")) <= 32768


def test_macos_profile_does_not_grant_broad_host_ipc_or_signalling(tmp_path):
    from adj_partner.analysis.sandbox import MacOSSandbox

    profile = MacOSSandbox()._profile(workspace(tmp_path))

    assert "(allow signal)" not in profile
    assert "(allow mach-lookup)" not in profile
    assert "(allow ipc-posix*)" not in profile
    assert "(allow process*)" not in profile


def test_linux_metrics_follow_descendants_independent_of_process_groups(tmp_path):
    from adj_partner.analysis.sandbox import _linux_descendant_metrics

    proc = tmp_path / "proc"
    for pid, parent, children, pages in (
            (100, 1, "101", 2), (101, 100, "102", 3), (102, 101, "", 5)):
        entry = proc / str(pid)
        (entry / "task" / str(pid)).mkdir(parents=True)
        (entry / "task" / str(pid) / "children").write_text(children, encoding="utf-8")
        (entry / "statm").write_text(f"10 {pages}\n", encoding="utf-8")
        (entry / "stat").write_text(f"{pid} (worker) S {parent} {pid}\n", encoding="utf-8")

    count, resident = _linux_descendant_metrics(100, proc_root=proc, page_size=4096)

    assert count == 3
    assert resident == (2 + 3 + 5) * 4096


def test_linux_metrics_include_children_of_nonleader_threads_and_parent_snapshot(tmp_path):
    from adj_partner.analysis.sandbox import _linux_descendant_metrics

    proc = tmp_path / "proc"

    def process(pid, parent, pages, tasks):
        entry = proc / str(pid)
        for task_id, children in tasks.items():
            task = entry / "task" / str(task_id)
            task.mkdir(parents=True)
            (task / "children").write_text(children, encoding="utf-8")
        (entry / "statm").write_text(f"10 {pages}\n", encoding="utf-8")
        (entry / "stat").write_text(f"{pid} (worker) S {parent} {pid}\n", encoding="utf-8")

    process(100, 1, 2, {100: "", 150: "200"})
    process(200, 150, 3, {200: ""})
    # Simulate the per-task children file racing an additional child: the
    # complete PPID snapshot still links it to thread 150.
    process(300, 150, 7, {300: ""})

    task_count, resident = _linux_descendant_metrics(100, proc_root=proc, page_size=4096)

    assert task_count == 4  # root leader + root thread + two child processes
    assert resident == (2 + 3 + 7) * 4096


@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="Requires Linux /proc task ancestry")
def test_linux_monitor_detects_thread_forked_setsided_child(tmp_path, monkeypatch):
    import adj_partner.analysis.sandbox as module

    root = workspace(tmp_path)
    marker = root / "outputs" / "thread-child.json"
    program = write_program(root,
        "import json,os,threading,time\n"
        "from pathlib import Path\n"
        "def spawn():\n"
        "    pid=os.fork()\n"
        "    if pid == 0:\n"
        "        os.setsid(); time.sleep(60); os._exit(0)\n"
        "    Path('outputs/thread-child.json').write_text(json.dumps({'pid':pid}))\n"
        "    time.sleep(60)\n"
        "threading.Thread(target=spawn).start()\n"
        "time.sleep(60)\n")
    monkeypatch.setattr(module, "PROCESS_LIMIT", 2)
    child_pid = None
    try:
        result = module._bounded_process(
            [sys.executable, "-I", str(program)], root, 10, "linux-test")

        assert result.limit_reason == "process_limit"
    finally:
        # The monitor may stop the program before its thread records the escaped child.
        if marker.is_file():
            child_pid = json.loads(marker.read_text())["pid"]
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_bubblewrap_uses_pid_one_namespace_for_reparented_descendants(tmp_path, monkeypatch):
    import adj_partner.analysis.sandbox as module

    root = workspace(tmp_path)
    captured = {}

    class Boundary:
        closed = False

        def close(self):
            self.closed = True

    boundary = Boundary()

    def bounded(command, workspace_path, timeout_seconds, backend, cancel_check=None,
                hard_boundary=None):
        captured["command"] = command
        captured["boundary"] = hard_boundary
        return module.SandboxResult(backend, 0, "", "", False, 0)

    monkeypatch.setattr(module, "_bounded_process", bounded)
    monkeypatch.setattr(module.LinuxCgroupBoundary, "create", lambda: boundary)
    module.BubblewrapSandbox("/usr/bin/true").run(
        ["/bin/true"], root, timeout_seconds=5)

    assert "--unshare-all" in captured["command"]
    assert "--as-pid-1" in captured["command"]
    assert captured["boundary"] is boundary
    assert boundary.closed is True


def test_bubblewrap_fails_closed_without_a_hard_cgroup_boundary(tmp_path, monkeypatch):
    import adj_partner.analysis.sandbox as module

    root = workspace(tmp_path)
    monkeypatch.setattr(module.LinuxCgroupBoundary, "create",
        lambda: (_ for _ in ()).throw(module.SandboxUnavailable("no delegation")))
    monkeypatch.setattr(module, "_bounded_process",
        lambda *args, **kwargs: pytest.fail("process launched without a hard boundary"))

    with pytest.raises(module.SandboxUnavailable, match="delegation"):
        module.BubblewrapSandbox("/usr/bin/true").run(
            ["/bin/true"], root, timeout_seconds=5)


def test_cgroup_v2_membership_requires_one_safe_unified_entry(tmp_path):
    import adj_partner.analysis.sandbox as module

    membership = tmp_path / "cgroup"
    membership.write_text("0::/user.slice/delegated.scope\n", encoding="utf-8")

    assert module._cgroup_v2_membership(membership) == Path("user.slice/delegated.scope")
    membership.write_text("2:pids:/legacy\n", encoding="utf-8")
    with pytest.raises(module.SandboxUnavailable, match="membership"):
        module._cgroup_v2_membership(membership)


def test_hard_boundary_limit_events_are_classified(tmp_path):
    from adj_partner.analysis.sandbox import LinuxCgroupBoundary

    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "pids.events").write_text("max 1\n", encoding="ascii")
    (root / "memory.events").write_text("max 0\noom 0\noom_kill 0\n", encoding="ascii")
    (root / "pids.current").write_text("32\n", encoding="ascii")
    (root / "memory.current").write_text("4096\n", encoding="ascii")
    boundary = LinuxCgroupBoundary(root)

    assert boundary.limit_reason() == "process_limit"
    assert boundary.metrics() == (32, 4096)


def test_bounded_process_reads_terminal_cgroup_event_after_fast_exit(tmp_path):
    from adj_partner.analysis.sandbox import _bounded_process

    root = workspace(tmp_path)
    release = root / "tmp" / "release"

    class Boundary:
        def __init__(self):
            self.reads = 0

        def enter(self):
            return None

        def metrics(self):
            return 1, 4096

        def limit_reason(self):
            self.reads += 1
            if self.reads == 1:
                release.write_text("exit", encoding="utf-8")
                return None
            return "process_limit"

    program = write_program(root,
        "from pathlib import Path\n"
        "import time\n"
        "while not Path('tmp/release').exists(): time.sleep(.001)\n"
        "raise SystemExit(1)\n")
    boundary = Boundary()

    result = _bounded_process([sys.executable, "-I", str(program)], root, 5,
                              "linux-test", hard_boundary=boundary)

    assert result.returncode == 1
    assert result.limit_reason == "process_limit"
    assert boundary.reads >= 2


def test_linux_cgroup_creation_applies_hard_pids_and_memory_limits(tmp_path, monkeypatch):
    import adj_partner.analysis.sandbox as module

    mount = tmp_path / "cgroup2"
    base = mount / "delegated"
    base.mkdir(parents=True)
    (base / "cgroup.controllers").write_text("memory pids\n", encoding="ascii")
    (base / "cgroup.subtree_control").write_text("memory pids\n", encoding="ascii")
    membership = tmp_path / "membership"
    membership.write_text("0::/delegated\n", encoding="ascii")
    original_mkdir = Path.mkdir
    controls = ("pids.max", "memory.max", "memory.swap.max", "memory.oom.group",
                "cgroup.procs", "cgroup.events", "cgroup.kill", "pids.current",
                "pids.events", "memory.current", "memory.events")

    def virtual_cgroup_mkdir(path, *args, **kwargs):
        original_mkdir(path, *args, **kwargs)
        if path.name.startswith("adj-analysis-") and path.name.endswith("-test"):
            for name in controls:
                (path / name).write_text("0\n", encoding="ascii")

    monkeypatch.setattr(module.secrets, "token_hex", lambda count: "test")
    monkeypatch.setattr(Path, "mkdir", virtual_cgroup_mkdir)

    boundary = module.LinuxCgroupBoundary.create(mount, membership)

    assert (boundary.path / "pids.max").read_text() == str(module.PROCESS_LIMIT)
    assert (boundary.path / "memory.max").read_text() == str(module.MEMORY_LIMIT_BYTES)
    assert (boundary.path / "memory.swap.max").read_text() == "0"
    assert (boundary.path / "memory.oom.group").read_text() == "1"


def test_sandbox_rejects_state_nested_beneath_a_runtime_read_root(tmp_path):
    from adj_partner.analysis.sandbox import MacOSSandbox, SandboxUnavailable

    unsafe = Path(sys.prefix).resolve() / "analysis-state" / "jobs" / "job" / "work"

    with pytest.raises(SandboxUnavailable, match="outside every runtime"):
        MacOSSandbox()._profile(unsafe)


@requires_nested_seatbelt
@pytest.mark.real_sandbox
def test_real_sandbox_cannot_signal_an_unrelated_same_user_process(tmp_path):
    from adj_partner.analysis.sandbox import select_sandbox

    unrelated = subprocess.Popen([sys.executable, "-I", "-c", "import time;time.sleep(60)"],
                                 start_new_session=True)
    try:
        root = workspace(tmp_path)
        result = run_python(select_sandbox(), root,
            "import ctypes,os,signal\n"
            "from pathlib import Path\n"
            f"pid={unrelated.pid}\n"
            "try:\n"
            "    os.kill(pid, signal.SIGCONT); outcome='allowed'\n"
            "except OSError:\n"
            "    outcome='denied'\n"
            "lib=ctypes.CDLL('/usr/lib/libSystem.B.dylib')\n"
            "bootstrap=ctypes.c_uint32.in_dll(lib,'bootstrap_port').value\n"
            "service=ctypes.c_uint32()\n"
            "lookup=lib.bootstrap_look_up(bootstrap,b'com.apple.cfprefsd.daemon',ctypes.byref(service))\n"
            "mach='denied' if lookup != 0 else 'allowed'\n"
            "Path('outputs/signal.txt').write_text(outcome+','+mach)\n")

        assert result.returncode == 0, result.stderr
        assert (root / "outputs" / "signal.txt").read_text() == "denied,denied"
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_bounded_process_reaps_background_process_group_after_success(tmp_path):
    from adj_partner.analysis.sandbox import _bounded_process

    root = workspace(tmp_path)
    marker = root / "outputs" / "pids.json"
    child = write_program(root,
        "import json,os,subprocess,sys\n"
        "from pathlib import Path\n"
        "process=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)'])\n"
        "Path('outputs/pids.json').write_text(json.dumps({'root':os.getpid(),'child':process.pid}))\n")
    pids = None
    try:
        result = _bounded_process([sys.executable, "-I", str(child)], root, 2, "test")
        pids = json.loads(marker.read_text())

        assert result.returncode == 0
        assert result.timed_out is False
        wait_for_process_exit(pids["child"])
    finally:
        if pids is None and marker.is_file():
            pids = json.loads(marker.read_text())
        if pids is not None:
            try:
                os.killpg(pids["root"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_sigterm_of_handler_reaps_its_separate_analysis_process_group(tmp_path):
    root = workspace(tmp_path)
    marker = root / "outputs" / "pids.json"
    child = write_program(root,
        "import json,os,subprocess,sys,time\n"
        "from pathlib import Path\n"
        "process=subprocess.Popen([sys.executable,'-I','-c','import time;time.sleep(60)'])\n"
        "Path('outputs/pids.json').write_text(json.dumps({'root':os.getpid(),'child':process.pid}))\n"
        "time.sleep(60)\n")
    project = Path(__file__).resolve().parents[1]
    controller_source = (
        "import sys\n"
        f"sys.path.insert(0,{str(project)!r})\n"
        "from adj_partner.analysis.sandbox import _bounded_process\n"
        f"_bounded_process([{sys.executable!r},'-I',{str(child)!r}],{str(root)!r},60,'test')\n")
    controller = subprocess.Popen([sys.executable, "-I", "-c", controller_source],
                                  start_new_session=True)
    pids = None
    try:
        wait_for_file(marker)
        pids = json.loads(marker.read_text())
        os.kill(controller.pid, signal.SIGTERM)
        controller.wait(timeout=5)

        wait_for_process_exit(pids["root"])
        wait_for_process_exit(pids["child"])
    finally:
        if controller.poll() is None:
            os.killpg(controller.pid, signal.SIGKILL)
            controller.wait(timeout=3)
        if pids is not None:
            try:
                os.killpg(pids["root"], signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_bounded_process_enforces_aggregate_writable_tree_quota_during_execution(tmp_path):
    from adj_partner.analysis.sandbox import _bounded_process

    root = workspace(tmp_path)
    child = write_program(root,
        "from pathlib import Path\n"
        "import time\n"
        "for number in range(9):\n"
        "    with (Path('outputs')/f'{number}.bin').open('wb') as stream: stream.truncate(8*1024*1024)\n"
        "time.sleep(60)\n")

    result = _bounded_process([sys.executable, "-I", str(child)], root, 3, "test")

    assert result.timed_out is False
    assert result.limit_reason == "outputs_byte_limit"


def test_backend_selection_fails_closed_when_platform_support_is_missing(monkeypatch):
    import adj_partner.analysis.sandbox as module

    monkeypatch.setattr(module.sys, "platform", "unsupported")
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    with pytest.raises(module.SandboxUnavailable, match="supported analysis sandbox"):
        module.select_sandbox()


def test_backend_selection_rejects_an_installed_backend_that_fails_self_test(monkeypatch):
    import adj_partner.analysis.sandbox as module

    class BrokenBackend:
        name = "broken"

    monkeypatch.setattr(module, "_candidate_sandbox", lambda: BrokenBackend())
    monkeypatch.setattr(module, "_self_test_sandbox",
                        lambda backend: (_ for _ in ()).throw(RuntimeError("private setup detail")))

    with pytest.raises(module.SandboxUnavailable, match="failed its containment self-test") as caught:
        module.select_sandbox()

    assert "private setup detail" not in str(caught.value)
