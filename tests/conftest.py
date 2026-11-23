"""Make the Adj_Partner project root importable when running pytest from anywhere."""
import os
import re
import sys
from pathlib import Path

import pytest

# Test runs must never write fixture papers into the project's literature library.
os.environ["ADJ_LITERATURE_LIBRARY"] = "off"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGES = ("adj_partner", "eval", "discovery_completion_fence")
CANONICAL = tuple(str(ROOT / name) for name in ("adj_partner", "eval", "tests"))
# The public checkout (no experiments/) omits the evaluation runners and data scripts; tests that need them skip.
PUBLIC_CHECKOUT = not (ROOT / "experiments").is_dir()
HELD_BACK = ("eval", "scripts", "experiments", "legacy")
# Private data, local work areas, the trace dashboard UI and held-back eval, script, config and record files.
HELD_BACK_FILES = (*HELD_BACK, "data", "workspace", "outputs", "config", "docs", "trace_dashboard")


def _held_back_module(error):
    if isinstance(error, ModuleNotFoundError):
        name = error.name or ""
    else:
        match = re.match(r"cannot import name '(\w+)' from '(\w+)'", str(error))
        name = f"{match[2]}.{match[1]}" if match else ""
    path = ROOT.joinpath(*name.split("."))
    held_back = (name.split(".")[0] in HELD_BACK and not path.with_suffix(".py").exists()
                 and not (path / "__init__.py").exists())
    return name if held_back else None


def _held_back_file(error):
    if not getattr(error, "filename", None):
        return None
    path = Path(error.filename).resolve()
    if not path.is_relative_to(ROOT.resolve()) or path.exists():
        return None
    relative = path.relative_to(ROOT.resolve())
    return str(relative) if relative.parts and relative.parts[0] in HELD_BACK_FILES else None


def pytest_configure(config):
    config.addinivalue_line("markers", "real_sandbox: runs code in the platform analysis sandbox "
                                       "(macOS sandbox-exec or Linux bwrap)")


def _sandbox_installed():
    from adj_partner.analysis import sandbox
    try:
        sandbox._candidate_sandbox()
    except sandbox.SandboxUnavailable:
        return False
    return True


def pytest_collection_modifyitems(config, items):
    # Only a missing backend skips; an installed backend that fails its self-test still fails the test.
    marked = [item for item in items if item.get_closest_marker("real_sandbox")]
    if marked and not _sandbox_installed():
        skip = pytest.mark.skip(reason="no analysis sandbox backend is installed (macOS sandbox-exec or Linux bwrap)")
        for item in marked:
            item.add_marker(skip)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not PUBLIC_CHECKOUT or call.excinfo is None:
        return
    error = call.excinfo.value
    name = (_held_back_module(error) if isinstance(error, ImportError)
            else _held_back_file(error) if isinstance(error, FileNotFoundError) else None)
    if name:
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0,
                           f"Skipped: needs {name}, which is not part of this checkout")


def _origin(module):
    path = getattr(module, "__file__", None)
    return str(Path(path).resolve()) if path else ""


def _project(name, module):
    # The virtualenv may live under ROOT; evicting its packages (e.g. textual) splits class identities.
    origin = _origin(module)
    if origin.startswith(str(Path(sys.prefix).resolve())):
        return False
    return name.split(".")[0] in PACKAGES or origin.startswith(str(ROOT))


@pytest.fixture(autouse=True)
def _restore_import_state():
    # Frozen-round runners import adj_partner from copied source trees; undo that per test.
    path = list(sys.path)
    before = {name: module for name, module in sys.modules.items() if _project(name, module)}
    yield
    sys.path[:] = path
    # Canonical modules imported against a cleared graph bind to classes that `before` replaces.
    rebuilt = any(sys.modules.get(name) is not module for name, module in before.items())
    for name, module in list(sys.modules.items()):
        if name in before or not _project(name, module):
            continue
        if rebuilt or not _origin(module).startswith(CANONICAL) or name.split(".")[0] not in before:
            sys.modules.pop(name, None)
    sys.modules.update(before)
