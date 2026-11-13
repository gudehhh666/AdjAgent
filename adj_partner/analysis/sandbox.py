"""Fail-closed OS containment for model-authored local analysis code."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import resource
import selectors
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

from ..tools.policy import ToolCancelled


MAX_STREAM_BYTES = 32768
MEMORY_LIMIT_BYTES = 2 * 1024 * 1024 * 1024
FILE_LIMIT_BYTES = 32 * 1024 * 1024
OPEN_FILE_LIMIT = 256
PROCESS_LIMIT = 32
MAX_WRITABLE_TREE_ENTRIES = 512
MAX_OUTPUT_TREE_BYTES = 64 * 1024 * 1024
MAX_TEMP_TREE_BYTES = 64 * 1024 * 1024


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxResult:
    backend: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    limit_reason: str | None = None


def _clean_environment(workspace):
    temporary = Path(workspace) / "tmp"
    python_bin = str(Path(sys.executable).resolve().parent)
    return {
        "HOME": str(temporary),
        "LANG": "C",
        "LC_ALL": "C",
        "MPLCONFIGDIR": str(temporary),
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PATH": os.pathsep.join((python_bin, "/usr/bin", "/bin", "/usr/sbin", "/sbin")),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": str(temporary),
        "VECLIB_MAXIMUM_THREADS": "1",
        "XDG_CACHE_HOME": str(temporary),
    }


def _resource_limiter(timeout_seconds):
    cpu_seconds = max(1, int(math.ceil(timeout_seconds)))

    def limit():
        limits = (
            (resource.RLIMIT_CORE, 0),
            (resource.RLIMIT_CPU, cpu_seconds),
            (resource.RLIMIT_FSIZE, FILE_LIMIT_BYTES),
            (resource.RLIMIT_NOFILE, OPEN_FILE_LIMIT),
        )
        # Darwin's RLIMIT_AS is not a usable finite limit for ordinary Python
        # processes (setrlimit rejects values below the already mapped address
        # space). The trusted monitor below enforces resident memory there.
        if hasattr(resource, "RLIMIT_AS") and sys.platform != "darwin":
            limits += ((resource.RLIMIT_AS, MEMORY_LIMIT_BYTES),)
        for kind, value in limits:
            hard = resource.getrlimit(kind)[1]
            selected = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(kind, (selected, selected))

    return limit


def _tree_usage(path, *, maximum_entries, maximum_bytes):
    entries = total = 0
    stack = [Path(path)]
    while stack:
        current = stack.pop()
        try:
            children = os.scandir(current)
        except FileNotFoundError:
            continue
        except OSError:
            return "monitor_error"
        with children:
            for child in children:
                entries += 1
                if entries > maximum_entries:
                    return "entry_limit"
                try:
                    info = child.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue
                except OSError:
                    return "monitor_error"
                if stat.S_ISDIR(info.st_mode):
                    stack.append(Path(child.path))
                elif stat.S_ISREG(info.st_mode):
                    total += info.st_size
                    if total > maximum_bytes:
                        return "byte_limit"
                else:
                    return "special_file"
    return None


def _workspace_limit_reason(workspace):
    for name, maximum in (("outputs", MAX_OUTPUT_TREE_BYTES), ("tmp", MAX_TEMP_TREE_BYTES)):
        reason = _tree_usage(Path(workspace) / name,
            maximum_entries=MAX_WRITABLE_TREE_ENTRIES, maximum_bytes=maximum)
        if reason:
            return f"{name}_{reason}"
    return None


def _linux_descendant_metrics(root_pid, proc_root=Path("/proc"), page_size=None):
    """Return metrics for the complete Bubblewrap PID-namespace tree.

    Read every thread's ``children`` file and independently build a parent map
    from every visible process. The two views close each other's races and
    include children forked by non-leader threads, plus setsid()/setpgid()
    descendants. Bubblewrap's private PID namespace reparents orphans to the
    namespace root, so they remain in this ancestry closure.
    """
    if page_size is None:
        page_size = os.sysconf("SC_PAGE_SIZE")

    snapshot = {}
    children_by_parent = {}
    try:
        entries = list(Path(proc_root).iterdir())
    except (FileNotFoundError, PermissionError, OSError):
        return 1, 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        parent = None
        try:
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
            parent = int(fields[1])
        except (FileNotFoundError, PermissionError, IndexError, ValueError, OSError):
            pass
        try:
            resident = int((entry / "statm").read_text().split()[1]) * page_size
        except (FileNotFoundError, PermissionError, IndexError, ValueError, OSError):
            resident = 0
        task_ids = {pid}
        task_children = set()
        try:
            tasks = list((entry / "task").iterdir())
        except (FileNotFoundError, PermissionError, OSError):
            tasks = []
        for task in tasks:
            if not task.name.isdigit():
                continue
            task_ids.add(int(task.name))
            try:
                values = (task / "children").read_text().split()
            except (FileNotFoundError, PermissionError, OSError):
                continue
            for value in values:
                try:
                    task_children.add(int(value))
                except ValueError:
                    continue
        snapshot[pid] = (resident, task_ids, task_children)
        if parent is not None:
            children_by_parent.setdefault(parent, set()).add(pid)

    pending = [int(root_pid)]
    seen_processes = set()
    seen_tasks = set()
    resident_bytes = 0
    while pending:
        pid = pending.pop()
        if pid in seen_processes:
            continue
        seen_processes.add(pid)
        process_resident, task_ids, task_children = snapshot.get(pid, (0, {pid}, set()))
        resident_bytes += process_resident
        seen_tasks.update(task_ids)
        pending.extend(task_children)
        for task_id in task_ids:
            pending.extend(children_by_parent.get(task_id, ()))
    return len(seen_tasks), resident_bytes


def _owned_process_metrics(root_pid):
    """Return (processes, resident bytes) for the backend-owned process tree."""
    if sys.platform == "darwin":
        import ctypes

        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        capacity = 128
        values = (ctypes.c_int * capacity)()
        count = library.proc_listpgrppids(root_pid, values, ctypes.sizeof(values))
        if count < 0:
            raise OSError(ctypes.get_errno(), "proc_listpgrppids failed")
        pids = [pid for pid in values[:count] if pid > 0]

        class Usage(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_ubyte * 16),
                        *[(name, ctypes.c_uint64) for name in (
                            "user", "system", "idle", "interrupt", "pageins", "wired",
                            "resident", "footprint", "start", "exit")]]

        resident = 0
        for pid in pids:
            usage = Usage()
            if library.proc_pid_rusage(pid, 0, ctypes.byref(usage)) == 0:
                resident += usage.resident
        return len(pids), resident
    if sys.platform.startswith("linux"):
        return _linux_descendant_metrics(root_pid)
    return 1, 0


def _cgroup_v2_membership(path=Path("/proc/self/cgroup")):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SandboxUnavailable("Linux cgroup v2 membership is unavailable") from exc
    matches = [line.split("::", 1)[1] for line in lines if line.startswith("0::")]
    if len(matches) != 1:
        raise SandboxUnavailable("Linux cgroup v2 membership is unavailable")
    relative = Path(matches[0].lstrip("/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise SandboxUnavailable("Linux cgroup v2 membership is unsafe")
    return relative


class LinuxCgroupBoundary:
    """Hard per-job pids/memory boundary in a delegated cgroup-v2 subtree."""

    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def create(cls, mount_root=Path("/sys/fs/cgroup"), membership_file=Path("/proc/self/cgroup")):
        mount_root = Path(mount_root).resolve()
        base = (mount_root / _cgroup_v2_membership(membership_file)).resolve()
        if not base.is_relative_to(mount_root) or not (base / "cgroup.controllers").is_file():
            raise SandboxUnavailable("A delegated cgroup v2 boundary is required for Linux analysis")
        try:
            enabled = set((base / "cgroup.subtree_control").read_text(encoding="utf-8").split())
        except OSError as exc:
            raise SandboxUnavailable("A delegated cgroup v2 boundary is required for Linux analysis") from exc
        if not {"memory", "pids"} <= enabled:
            raise SandboxUnavailable("Linux analysis requires delegated memory and pids controllers")

        path = base / f"adj-analysis-{os.getpid()}-{secrets.token_hex(8)}"
        try:
            path.mkdir(mode=0o700)
            required = {
                "pids.max": str(PROCESS_LIMIT),
                "memory.max": str(MEMORY_LIMIT_BYTES),
                "memory.swap.max": "0",
                "memory.oom.group": "1",
            }
            for name, value in required.items():
                control = path / name
                if not control.is_file():
                    raise OSError(f"missing {name}")
                control.write_text(value, encoding="ascii")
                if control.read_text(encoding="ascii").strip() != value:
                    raise OSError(f"failed to apply {name}")
            for name in ("cgroup.procs", "cgroup.events", "cgroup.kill",
                         "pids.current", "pids.events", "memory.current", "memory.events"):
                if not (path / name).is_file():
                    raise OSError(f"missing {name}")
        except OSError as exc:
            try:
                path.rmdir()
            except OSError:
                pass
            raise SandboxUnavailable("The host could not establish hard Linux analysis limits") from exc
        return cls(path)

    def enter(self):
        descriptor = os.open(self.path / "cgroup.procs", os.O_WRONLY | os.O_CLOEXEC)
        try:
            os.write(descriptor, b"0")
        finally:
            os.close(descriptor)

    @staticmethod
    def _events(path):
        values = {}
        for line in Path(path).read_text(encoding="ascii").splitlines():
            name, value = line.split()
            values[name] = int(value)
        return values

    def metrics(self):
        return (int((self.path / "pids.current").read_text(encoding="ascii").strip()),
                int((self.path / "memory.current").read_text(encoding="ascii").strip()))

    def limit_reason(self):
        if self._events(self.path / "pids.events").get("max", 0):
            return "process_limit"
        memory = self._events(self.path / "memory.events")
        if any(memory.get(name, 0) for name in ("max", "oom", "oom_kill", "oom_group_kill")):
            return "memory_limit"
        return None

    def close(self):
        try:
            populated = self._events(self.path / "cgroup.events").get("populated", 0)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise SandboxUnavailable("The host could not clean the Linux analysis boundary") from exc
        if populated:
            try:
                (self.path / "cgroup.kill").write_text("1", encoding="ascii")
            except OSError as exc:
                raise SandboxUnavailable("The host could not clean the Linux analysis boundary") from exc
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                if self._events(self.path / "cgroup.events").get("populated", 0) == 0:
                    break
            except FileNotFoundError:
                return
            time.sleep(.01)
        else:
            raise SandboxUnavailable("The Linux analysis boundary did not become empty")
        try:
            self.path.rmdir()
        except OSError as exc:
            raise SandboxUnavailable("The host could not remove the Linux analysis boundary") from exc


def _group_exists(group_id):
    try:
        os.killpg(group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process):
    """Terminate only the process group created by this exact Popen object."""
    group_id = process.pid
    if process.poll() is None:
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        os.killpg(group_id, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + .2
    while _group_exists(group_id) and time.monotonic() < deadline:
        time.sleep(.01)
    if _group_exists(group_id):
        try:
            os.killpg(group_id, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        try:
            os.kill(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)


def _drain_streams(selector, captured, timeout=.25):
    deadline = time.monotonic() + timeout
    while selector.get_map() and time.monotonic() < deadline:
        events = selector.select(timeout=min(.02, max(0, deadline - time.monotonic())))
        if not events:
            continue
        for key, _ in events:
            try:
                chunk = os.read(key.fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            target = captured[key.data]
            remaining = MAX_STREAM_BYTES - len(target)
            if remaining > 0:
                target.extend(chunk[:remaining])


def _bounded_process(command, workspace, timeout_seconds, backend, cancel_check=None,
                     hard_boundary=None):
    if threading.current_thread() is not threading.main_thread():
        raise SandboxUnavailable("Analysis sandbox supervision requires the worker main thread")
    started = time.monotonic()
    limiter = _resource_limiter(timeout_seconds)

    def prepare_child():
        limiter()
        if hard_boundary is not None:
            hard_boundary.enter()

    try:
        process = subprocess.Popen(command, cwd=workspace, env=_clean_environment(workspace),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, close_fds=True, preexec_fn=prepare_child)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise SandboxUnavailable("The host could not establish required analysis resource limits") from exc
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, "stdout")
    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
    os.set_blocking(process.stdout.fileno(), False)
    os.set_blocking(process.stderr.fileno(), False)
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    timed_out = False
    limit_reason = None
    pending_error = None
    previous_handlers = {}

    def interrupted(signum, frame):
        raise ToolCancelled("Analysis execution interrupted by outer job cancellation")

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        while process.poll() is None:
            if cancel_check is not None:
                cancel_check()
            if time.monotonic() - started >= timeout_seconds:
                timed_out = True
                break
            limit_reason = _workspace_limit_reason(workspace)
            if limit_reason:
                break
            if hard_boundary is not None:
                try:
                    limit_reason = hard_boundary.limit_reason()
                    hard_count, hard_memory = hard_boundary.metrics()
                except OSError as exc:
                    raise SandboxUnavailable("The hard analysis resource boundary became unavailable") from exc
                if limit_reason:
                    break
            else:
                hard_count = hard_memory = 0
            process_count, resident_bytes = _owned_process_metrics(process.pid)
            process_count = max(process_count, hard_count)
            resident_bytes = max(resident_bytes, hard_memory)
            if process_count > PROCESS_LIMIT:
                limit_reason = "process_limit"
                break
            if resident_bytes > MEMORY_LIMIT_BYTES:
                limit_reason = "memory_limit"
                break
            for key, _ in selector.select(timeout=.05):
                try:
                    chunk = os.read(key.fd, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = captured[key.data]
                remaining = MAX_STREAM_BYTES - len(target)
                if remaining > 0:
                    target.extend(chunk[:remaining])
    except BaseException as exc:
        pending_error = exc
    finally:
        _terminate_process_group(process)
        _drain_streams(selector, captured)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        selector.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
    # A hard pids/memory event can terminate the launcher between monitor
    # samples. Read the still-live cgroup once more before its owner removes it
    # so fast limit exits retain the stable resource-limit classification.
    if (pending_error is None and not timed_out and limit_reason is None
            and hard_boundary is not None):
        try:
            limit_reason = hard_boundary.limit_reason()
        except OSError as exc:
            raise SandboxUnavailable("The hard analysis resource boundary became unavailable") from exc
    if pending_error is not None:
        raise pending_error
    return SandboxResult(backend=backend, returncode=process.returncode,
        stdout=bytes(captured["stdout"]).decode("utf-8", errors="replace"),
        stderr=bytes(captured["stderr"]).decode("utf-8", errors="replace"),
        timed_out=timed_out, duration_seconds=time.monotonic() - started,
        limit_reason=limit_reason)


def _quote_profile(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _runtime_read_roots():
    candidates = [
        Path("/System"), Path("/usr"), Path("/bin"), Path("/sbin"),
        Path("/Library/Apple"), Path("/private/var/db/dyld"),
        Path("/private/var/db/timezone"),
        Path(sys.prefix), Path(sys.base_prefix), Path(sys.executable).resolve().parent,
    ]
    roots = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if candidate.exists() and resolved not in roots:
            roots.append(resolved)
    return roots


def _runtime_read_files():
    # Python's stdlib mimetypes loader probes these fixed system databases.
    # Permit only the files that actually exist, never the containing /etc tree.
    candidates = (
        "/etc/mime.types", "/etc/httpd/mime.types", "/etc/httpd/conf/mime.types",
        "/etc/apache/mime.types", "/etc/apache2/mime.types",
    )
    files = []
    for candidate in candidates:
        path = Path(candidate)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if path.is_file() and resolved not in files:
            files.append(resolved)
    return files


def _validate_runtime_separation(workspace):
    workspace = Path(workspace).resolve()
    if any(workspace.is_relative_to(root) for root in _runtime_read_roots()):
        raise SandboxUnavailable("Analysis state must be outside every runtime read root")


class MacOSSandbox:
    name = "macos-sandbox-exec"

    def _profile(self, workspace):
        workspace = Path(workspace).resolve()
        _validate_runtime_separation(workspace)
        reads = [*_runtime_read_roots(), workspace]
        read_rules = " ".join([*(f"(subpath {_quote_profile(path)})" for path in reads),
                               *(f"(literal {_quote_profile(path)})" for path in _runtime_read_files())])
        write_rules = " ".join(f"(subpath {_quote_profile(workspace / name)})" for name in ("outputs", "tmp"))
        devices = " ".join(f"(literal {_quote_profile(path)})" for path in ("/dev/null", "/dev/random", "/dev/urandom"))
        return "\n".join((
            "(version 1)",
            "(deny default)",
            "(import \"dyld-support.sb\")",
            "(allow mach-bootstrap)",
            "(deny network*)",
            "(deny file-read* (subpath \"/private/etc\"))",
            "(deny process-fork)",
            "(allow signal (target self))",
            "(deny mach-lookup)",
            "(deny ipc-posix*)",
            "(allow process-exec)",
            "(allow sysctl-read)",
            f"(allow file-read* {read_rules} {devices})",
            f"(allow file-write* {write_rules} (literal \"/dev/null\"))",
        )) + "\n"

    def run(self, command, workspace, *, timeout_seconds, cancel_check=None):
        workspace = _validate_workspace(workspace)
        _validate_runtime_separation(workspace)
        profile = workspace / "tmp" / "analysis.sb"
        profile.write_text(self._profile(workspace), encoding="utf-8")
        wrapped = ["/usr/bin/sandbox-exec", "-f", str(profile), *map(str, command)]
        return _bounded_process(wrapped, workspace, timeout_seconds, self.name, cancel_check)


class BubblewrapSandbox:
    name = "linux-bubblewrap"

    def __init__(self, executable):
        self.executable = str(Path(executable).resolve())

    def run(self, command, workspace, *, timeout_seconds, cancel_check=None):
        workspace = _validate_workspace(workspace)
        _validate_runtime_separation(workspace)
        wrapped = [self.executable, "--die-with-parent", "--new-session", "--unshare-all", "--as-pid-1",
                   "--proc", "/proc", "--dev", "/dev"]
        roots = [Path("/usr"), Path("/bin"), Path("/sbin"), Path("/lib"), Path("/lib64"),
                 Path(sys.prefix), Path(sys.base_prefix)]
        seen = set()
        for root in roots:
            if root.exists():
                root = root.resolve()
                if root not in seen:
                    wrapped.extend(("--ro-bind", str(root), str(root)))
                    seen.add(root)
        wrapped.extend(("--ro-bind", str(workspace), str(workspace),
                        "--bind", str(workspace / "outputs"), str(workspace / "outputs"),
                        "--bind", str(workspace / "tmp"), str(workspace / "tmp"),
                        "--chdir", str(workspace), "--", *map(str, command)))
        boundary = LinuxCgroupBoundary.create()
        try:
            return _bounded_process(wrapped, workspace, timeout_seconds, self.name, cancel_check,
                                    hard_boundary=boundary)
        finally:
            boundary.close()


def _validate_workspace(workspace):
    workspace = Path(workspace)
    if workspace.is_symlink() or workspace.resolve() != workspace:
        raise ValueError("Analysis workspace cannot be a symlink")
    for name in ("inputs", "outputs", "tmp"):
        child = workspace / name
        if child.is_symlink() or not child.is_dir() or child.resolve().parent != workspace:
            raise ValueError("Analysis workspace directories are missing or unsafe")
    return workspace


def _candidate_sandbox():
    if sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file():
        return MacOSSandbox()
    executable = shutil.which("bwrap") if sys.platform.startswith("linux") else None
    if executable:
        return BubblewrapSandbox(executable)
    raise SandboxUnavailable("No supported analysis sandbox is installed")


def _self_test_sandbox(sandbox):
    with tempfile.TemporaryDirectory(prefix="adj-analysis-self-test-") as temporary:
        base = Path(temporary).resolve()
        workspace = base / "work"
        for name in ("inputs", "outputs", "tmp"):
            (workspace / name).mkdir(parents=True)
        outside = base / "outside.txt"
        outside.write_text("private", encoding="utf-8")
        probes = [
            "from pathlib import Path",
            "import json,os,socket",
            "result={}",
            "try: Path('outputs/ok.txt').write_text('ok'); result['write']='ok'",
            "except OSError: result['write']='denied'",
            f"try: Path({str(outside)!r}).read_text(); result['outside']='allowed'",
            "except OSError: result['outside']='denied'",
            "try: socket.socket().connect(('127.0.0.1',9)); result['network']='allowed'",
            "except OSError: result['network']='denied'",
        ]
        if sandbox.name == "macos-sandbox-exec":
            probes.extend([
                "import ctypes,signal",
                "try: os.kill(os.getppid(),signal.SIGCONT); result['signal']='allowed'",
                "except OSError: result['signal']='denied'",
                "try: pid=os.fork(); result['fork']='allowed'",
                "except OSError: result['fork']='denied'",
                "libc=ctypes.CDLL(None,use_errno=True)",
                "name=('/adj_analysis_'+str(os.getpid())).encode()",
                "descriptor=libc.shm_open(name,os.O_CREAT|os.O_EXCL|os.O_RDWR,0o600)",
                "result['ipc']='allowed' if descriptor >= 0 else 'denied'",
                "descriptor < 0 or os.close(descriptor)",
                "descriptor < 0 or libc.shm_unlink(name)",
                "libsystem=ctypes.CDLL('/usr/lib/libSystem.B.dylib')",
                "bootstrap=ctypes.c_uint32.in_dll(libsystem,'bootstrap_port').value",
                "service=ctypes.c_uint32()",
                "lookup=libsystem.bootstrap_look_up(bootstrap,b'com.apple.cfprefsd.daemon',ctypes.byref(service))",
                "result['mach']='denied' if lookup != 0 else 'allowed'",
            ])
        probes.append("Path('outputs/self_test.json').write_text(json.dumps(result,sort_keys=True))")
        program = workspace / "self_test.py"
        program.write_text("\n".join(probes) + "\n", encoding="utf-8")
        result = sandbox.run([sys.executable, "-I", str(program)], workspace,
                             timeout_seconds=5)
        if result.returncode != 0 or result.timed_out or result.limit_reason:
            raise RuntimeError("sandbox probe did not complete")
        values = __import__("json").loads((workspace / "outputs" / "self_test.json").read_text())
        expected = {"write": "ok", "outside": "denied", "network": "denied"}
        if sandbox.name == "macos-sandbox-exec":
            expected.update({"signal": "denied", "fork": "denied", "ipc": "denied", "mach": "denied"})
        if values != expected:
            raise RuntimeError("sandbox probe did not enforce the expected boundary")


def select_sandbox():
    sandbox = _candidate_sandbox()
    try:
        _self_test_sandbox(sandbox)
    except SandboxUnavailable:
        raise
    except BaseException as exc:
        raise SandboxUnavailable("The installed analysis sandbox failed its containment self-test") from exc
    return sandbox
