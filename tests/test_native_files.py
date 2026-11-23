"""Actual filesystem boundary tests for shared native readers, not a sandbox claim."""

import hashlib
import importlib.util
import os
from pathlib import Path

import pytest

from adj_partner.contracts import Budget, SessionConfig
from adj_partner.storage import Store
from adj_partner.tools.policy import ToolContext


def reader():
    assert importlib.util.find_spec("adj_partner.native"), "native shared capability package must exist"
    from adj_partner.native.safe_files import read_input
    return read_input


def context(tmp_path):
    store = Store(tmp_path / "private-state")
    session = store.create_session(SessionConfig(allowed_roots=[str(tmp_path)]).to_dict())
    task = store.create_task(session["id"], "Synthetic file inspection")
    run = store.create_run(session["id"], task["id"], Budget().to_dict())
    return ToolContext(store, session["id"], run["id"], 0, call_id="fixture-read")


def test_reads_exact_bounded_regular_file_with_input_hash(tmp_path):
    ctx = context(tmp_path)
    content = b"exact bytes\x00\xff"
    path = tmp_path / "中文 input.bin"
    path.write_bytes(content)
    result = reader()(ctx, path, maximum_bytes=len(content))
    assert result.content == content and result.size == len(content)
    assert result.sha256 == hashlib.sha256(content).hexdigest()
    assert result.path == str(path.resolve())
    with pytest.raises(ValueError, match="[Bb]yte|[Ll]imit|[Ll]arge|[Ss]ize"):
        reader()(ctx, path, maximum_bytes=len(content)-1)


@pytest.mark.parametrize("name", [".env", ".env.local", "credentials.json", "secrets.yaml", "key.pem", ".ssh/id_rsa"])
def test_credential_paths_are_denied_even_under_an_allowed_root(tmp_path, name):
    ctx = context(tmp_path)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic fixture only")
    with pytest.raises(PermissionError):
        reader()(ctx, path)


@pytest.mark.parametrize("variant", ["final_inside", "final_outside", "parent"])
def test_symlink_components_are_denied_not_merely_resolved(tmp_path, variant):
    ctx = context(tmp_path)
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "data.txt").write_text("fixture")
    link = tmp_path / "linked"
    if variant == "parent":
        link.symlink_to(actual, target_is_directory=True)
        requested = link / "data.txt"
    else:
        link.symlink_to(actual / "data.txt" if variant == "final_inside" else Path("/etc/passwd"))
        requested = link
    with pytest.raises((PermissionError, OSError)):
        reader()(ctx, requested)


def test_traversal_cross_root_and_private_store_are_denied(tmp_path):
    ctx = context(tmp_path)
    good = tmp_path / "data.txt"
    good.write_text("fixture")
    requests = [str(tmp_path / "child") + "/../data.txt", Path("/etc/passwd"), ctx.store.root / "state.sqlite3"]
    for requested in requests:
        with pytest.raises(PermissionError):
            reader()(ctx, requested)


def test_fifo_and_directory_are_rejected_without_reading_them(tmp_path):
    ctx = context(tmp_path)
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    for requested in (fifo, tmp_path):
        with pytest.raises((ValueError, PermissionError, OSError)):
            reader()(ctx, requested)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_resource_limits_rejected(tmp_path, limit):
    ctx = context(tmp_path)
    path = tmp_path / "data.txt"
    path.write_text("fixture")
    with pytest.raises(ValueError):
        reader()(ctx, path, maximum_bytes=limit)


def test_symlink_swap_after_path_check_cannot_redirect_read(tmp_path, monkeypatch):
    ctx = context(tmp_path)
    path = tmp_path / "data.txt"
    secret = tmp_path / "credentials.json"
    path.write_text("ordinary fixture")
    secret.write_text("synthetic private marker")
    original_check = ctx.check_path

    def swap(*args, **kwargs):
        checked = original_check(*args, **kwargs)
        path.unlink()
        path.symlink_to(secret)
        return checked

    monkeypatch.setattr(ctx, "check_path", swap)
    with pytest.raises((PermissionError, OSError)):
        reader()(ctx, path)


def test_persisted_cancellation_denies_read(tmp_path):
    ctx = context(tmp_path)
    path = tmp_path / "data.txt"
    path.write_text("fixture")
    run = ctx.store.get_run(ctx.run_id)
    ctx.store.update_run(ctx.run_id, {"status": "cancelled"}, run["revision"])
    from adj_partner.tools.policy import ToolCancelled
    with pytest.raises(ToolCancelled):
        reader()(ctx, path)


def test_hardlink_cannot_alias_private_store_into_an_allowed_input_name(tmp_path):
    ctx = context(tmp_path)
    alias = tmp_path / "apparently_public.db"
    os.link(ctx.store.db_path, alias)
    with pytest.raises(PermissionError):
        reader()(ctx, alias)
