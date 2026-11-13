"""Record-level sequence scope and fail-closed native inventory contract."""

import bz2
import hashlib
import importlib
import importlib.util
import io
import json
import lzma
import os
import stat
import tarfile
import threading
import zipfile

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor
from adj_partner.tools.policy import ToolCancelled, ToolContext


def test_sequence_inventory_module_is_available():
    assert importlib.util.find_spec("adj_partner.native.sequence_inventory"), "sequence inventory must exist"


@pytest.fixture
def env(tmp_path):
    assert importlib.util.find_spec("adj_partner.native.sequence_inventory"), "sequence inventory must exist"
    module = importlib.import_module("adj_partner.native.sequence_inventory")
    catalog = ToolCatalog()
    module.register_sequence_inventory(catalog)
    store = Store(tmp_path / "state")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    sid = store.create_session(SessionConfig(allowed_roots=[str(inputs)],
        allowed_tools=catalog.names(), enabled_tools=catalog.names()).to_dict())["id"]
    task = store.create_task(sid, "sequence scope audit", {})["id"]
    rid = store.create_run(sid, task, Budget().to_dict())["id"]
    return module, store, inputs, catalog, ToolExecutor(store, catalog), sid, rid


def run(env, input_paths, **kwargs):
    _, _, _, _, executor, sid, rid = env
    return executor.execute(ToolCall("inventory", "sequence_inventory", {
        "paths": [str(path) for path in input_paths], "member_glob": "*",
        "scope_note": "User selected protein FASTA inputs; no species inference.", **kwargs}), sid, rid)


def archive(env, members):
    path = env[2] / "proteins.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name, content in members:
            handle.writestr(name, content)
    return path


def counts(files, records, residues, gaps=0, stops=0):
    return dict(files=files, records=records, residues=residues, gaps=gaps, stops=stops)


def assert_failed(env, result):
    assert result.status in {"error", "denied"}, result.to_dict()
    assert env[1].artifacts(run_id=env[6]) == []


def test_record_counts_selection_duplicates_and_provenance(env):
    multi = ">same first\nAX-z.*\n>same second\nbuO\n"
    path = archive(env, [("single_copy/OG1.faa", multi),
        ("single_copy/OG2.faa", ">other\nQQ\n"), ("duplicated/OG1.faa", ">same\nAAAA\n")])
    result = run(env, [path], member_glob="single_copy/*", file_stems=["OG1"],
                 unresolved_scope=["Species identity was not supplied."])
    assert result.status == "ok", result.to_dict()
    assert result.data["counts"] == {"before_filters": counts(3, 4, 12, 2, 1),
        "after_member_filter": counts(2, 3, 8, 2, 1), "after_stem_filter": counts(1, 2, 6, 2, 1)}
    assert result.data["inputs"][0]["counts"] == result.data["counts"]
    assert result.data["duplicate_record_ids"] == 1
    assert result.data["records_with_duplicate_ids"] == 3
    output = json.loads(env[1].read_artifact(result.data["artifact_id"]))
    assert output["scope"]["unresolved_scope"] == ["Species identity was not supplied."]
    source = output["inputs"][0]
    assert source["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    member = source["members"][0]
    assert member["sha256"] == hashlib.sha256(multi.encode()).hexdigest()
    assert [r["ordinal"] for r in member["records"]] == [1, 2]
    assert [r["record_id"] for r in member["records"]] == ["same", "same"]
    assert all(r["selected"] and r["duplicate_record_id"] for r in member["records"])
    assert output["inputs"][0]["members"][2]["records"][0]["selected"] is False
    assert result.artifacts[0]["run_id"] == env[6]
    assert "members" not in result.data["inputs"][0]


@pytest.mark.parametrize("selection,expected", [(None, counts(2, 3, 6)), ([], counts(0, 0, 0)),
    (["OG1"], counts(1, 2, 4)), (["og1"], counts(0, 0, 0)), (["OG*"], counts(0, 0, 0))])
def test_explicit_empty_stems_and_exact_case_sensitive_matching(env, selection, expected):
    path = archive(env, [("OG1.faa", ">a\nAA\n>b\nCC\n"), ("OG2.faa", ">c\nDD\n")])
    result = run(env, [path], **({} if selection is None else {"file_stems": selection}))
    assert result.status == "ok", result.to_dict()
    assert result.data["counts"]["after_stem_filter"] == expected


def test_plain_files_use_basename_and_keep_per_input_counts(env):
    a, b = env[2] / "one.fa", env[2] / "two.FASTA"
    a.write_text(">x\nA C\n>x\nBZXJUO\n")
    b.write_text(">y\nMM\n")
    result = run(env, [a, b], member_glob="one.*")
    assert result.status == "ok", result.to_dict()
    assert result.data["counts"]["after_member_filter"] == counts(1, 2, 8)
    assert result.data["inputs"][1]["counts"]["after_member_filter"] == counts(0, 0, 0)


@pytest.mark.parametrize("content", [b"", b"ACD\n", b">\nACD", b">a\n", b">a\n>b\nA", b">a\nAC1",
    b">a\nAC?", b">a\nA\xff", b">a\nA>Q"])
def test_malformed_fasta_rejected_even_if_excluded(env, content):
    path = archive(env, [("selected.faa", ">good\nAC\n"), ("excluded.faa", content)])
    assert_failed(env, run(env, [path], member_glob="selected.faa"))


@pytest.mark.parametrize("name", ["../bad.faa", "/bad.faa", "a/../bad.faa", "a\\bad.faa",
    "C:/bad.faa", "./bad.faa", "a//bad.faa", "nested.zip", "nested.tar.gz"])
def test_unsafe_or_unsupported_members_rejected(env, name):
    path = archive(env, [(name, ">a\nAC\n")])
    assert_failed(env, run(env, [path], member_glob="unmatched/*"))


def test_duplicate_casefold_paths_rejected(env):
    path = archive(env, [("A.faa", ">a\nA"), ("a.faa", ">b\nB")])
    assert_failed(env, run(env, [path]))


def test_zip_symlink_rejected_and_directory_allowed(env):
    link = zipfile.ZipInfo("alias.faa")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    assert_failed(env, run(env, [archive(env, [(link, "target.faa")])]))


def test_safe_zip_directory_entries_do_not_count_as_files(env):
    path = archive(env, [("group/", ""), ("group/a.faa", ">a\nA")])
    result = run(env, [path])
    assert result.status == "ok", result.to_dict()
    assert result.data["counts"]["before_filters"] == counts(1, 1, 1)


def test_non_fasta_metadata_is_validated_and_explicitly_excluded(env):
    path = archive(env, [("a.faa", ">a\nA"), ("summary.txt", "metadata"), ("report.json", "{}")])
    result = run(env, [path])
    assert result.status == "ok", result.to_dict()
    assert result.data["excluded_non_fasta_files"] == 2
    assert result.data["counts"]["before_filters"] == counts(1, 1, 1)
    output = json.loads(env[1].read_artifact(result.data["artifact_id"]))
    assert [m["kind"] for m in output["inputs"][0]["members"]] == ["fasta", "excluded_non_fasta", "excluded_non_fasta"]


@pytest.mark.parametrize("encoding", ["bzip2", "xz", "zstd", "ustar", "v7_tar"])
def test_renamed_nested_archives_are_rejected_before_publication(env, encoding):
    payload = b">nested\nACD\n"
    if encoding == "bzip2":
        content = bz2.compress(payload)
    elif encoding == "xz":
        content = lzma.compress(payload)
    elif encoding == "zstd":
        # A valid single-segment frame with one final raw (uncompressed) block.
        content = b"\x28\xb5\x2f\xfd\x20" + bytes([len(payload)])
        content += ((len(payload) << 3) | 1).to_bytes(3, "little") + payload
    else:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as nested:
            member = tarfile.TarInfo("nested.faa")
            member.size = len(payload)
            nested.addfile(member, io.BytesIO(payload))
        content = buffer.getvalue()
        if encoding == "v7_tar":
            header = bytearray(content[:512])
            header[257:265] = b"\0" * 8
            header[148:156] = b" " * 8
            header[148:156] = ("%06o\0 " % sum(header)).encode("ascii")
            content = bytes(header) + content[512:]
        # Verify the TAR fixture itself, including the legacy checksum variant.
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as nested:
            assert nested.extractfile("nested.faa").read() == payload
    path = archive(env, [("selected.faa", ">selected\nAC"), ("hidden.dat", content)])
    assert_failed(env, run(env, [path], member_glob="selected.faa"))


@pytest.mark.parametrize("kind", ["unauthorized", "symlink", "hardlink", "duplicate", "store_private", "fifo", "bad_zip"])
def test_input_authority_and_regular_snapshot_contract(env, tmp_path, kind):
    path = env[2] / "a.faa"
    path.write_text(">a\nA")
    paths = [path]
    if kind == "unauthorized":
        path = tmp_path / "outside.faa"
        path.write_text(">a\nA")
        paths = [path]
    elif kind in {"symlink", "hardlink"}:
        alias = env[2] / "alias.faa"
        os.symlink(path, alias) if kind == "symlink" else os.link(path, alias)
        paths = [alias]
    elif kind == "duplicate":
        paths *= 2
    elif kind == "store_private":
        paths = [env[1].root / "private.faa"]
        paths[0].write_text(">a\nA")
    elif kind == "fifo":
        paths = [env[2] / "pipe.faa"]
        os.mkfifo(paths[0])
    elif kind == "bad_zip":
        paths = [env[2] / "bad.zip"]
        paths[0].write_bytes(b"PK malformed")
    assert_failed(env, run(env, paths))


@pytest.mark.parametrize("limit,value", [("MAX_INPUT_BYTES", 5), ("MAX_TOTAL_INPUT_BYTES", 5),
    ("MAX_EXPANDED_BYTES", 5), ("MAX_MEMBERS", 1), ("MAX_RECORDS", 1)])
def test_resource_bounds_are_fail_closed(env, monkeypatch, limit, value):
    path = archive(env, [("a.faa", ">a\nAAAA"), ("b.faa", ">b\nBBBB")])
    monkeypatch.setattr(env[0], limit, value)
    assert_failed(env, run(env, [path]))


def test_crc_failure_is_not_hidden_by_selection(env):
    path = env[2] / "crc.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as handle:
        handle.writestr("excluded.faa", ">a\nAAAA")
    path.write_bytes(path.read_bytes().replace(b">a\nAAAA", b">a\nBAAA"))
    assert_failed(env, run(env, [path], file_stems=[]))


@pytest.mark.parametrize("arguments", [{"member_glob": ""}, {"scope_note": "   "},
    {"paths": []}, {"file_stems": ["a", "a"]}, {"unresolved_scope": "unknown"}])
def test_explicit_bounded_scope_schema(env, arguments):
    path = env[2] / "a.faa"
    path.write_text(">a\nA")
    assert_failed(env, run(env, [path], **arguments))


def test_cancellation_before_publication_leaves_no_artifact(env):
    event = threading.Event()
    event.set()
    context = ToolContext(env[1], env[5], env[6], 1, event)
    with pytest.raises(ToolCancelled):
        env[3].get("sequence_inventory").handler(context, paths=[str(env[2] / "none.faa")],
            member_glob="*", scope_note="all records")
    assert env[1].artifacts(run_id=env[6]) == []


def test_registered_in_native_catalog(env):
    from adj_partner.native.capabilities import NativeCapabilities
    catalog = ToolCatalog()
    NativeCapabilities(env[1]).register(catalog)
    assert "sequence_inventory" in catalog.names()
