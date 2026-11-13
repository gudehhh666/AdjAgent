"""Bounded FASTA record inventory; selection describes scope, never biology."""

from collections import Counter
from fnmatch import fnmatchcase
import hashlib
import io
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import zipfile

from ..contracts import ToolResult
from ..tools import ToolSpec
from ..tools.catalog import canonical_json, validator
from .safe_files import read_input


MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_INPUT_BYTES = 128 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_MEMBERS = 10000
MAX_RECORDS = 100000
FASTA_SUFFIXES = {".fa", ".faa", ".fasta"}
ARCHIVE_SUFFIXES = {".zip", ".gz", ".bz2", ".xz", ".tar", ".tgz", ".tbz", ".tbz2", ".txz", ".7z", ".rar", ".zst"}
STAGES = ("before_filters", "after_member_filter", "after_stem_filter")
LIMITATIONS = [
    "Counts describe supplied files and explicit filters only; paths do not establish species, orthology or function.",
    "Every FASTA record is retained, including repeated IDs. Alphabetic symbols count as residues; '-' and '.' are gaps; '*' is a stop.",
    "Non-FASTA ZIP members are validated and excluded. Nested archives are unsupported.",
    "Nested-archive detection uses known filename extensions, content signatures and valid TAR headers; arbitrary encoded or unrecognized compressed bytes cannot all be detected.",
]
SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["paths", "member_glob", "scope_note"],
    "properties": {
        "paths": {"type": "array", "minItems": 1, "maxItems": 8, "uniqueItems": True,
                  "items": {"type": "string", "minLength": 1, "maxLength": 4096}},
        "member_glob": {"type": "string", "minLength": 1, "maxLength": 1024,
                        "description": "Explicit case-sensitive fnmatch of relative member paths; plain files use basename. '*' also matches '/'."},
        "file_stems": {"type": "array", "maxItems": 10000, "uniqueItems": True,
                       "items": {"type": "string", "minLength": 1, "maxLength": 1024},
                       "description": "Exact case-sensitive final filename stems. Omit for all; [] selects none."},
        "scope_note": {"type": "string", "minLength": 1, "maxLength": 2000, "pattern": "\\S"},
        "unresolved_scope": {"type": "array", "maxItems": 20,
                             "items": {"type": "string", "minLength": 1, "maxLength": 500}},
    },
}


def register_sequence_inventory(catalog):
    catalog.register(ToolSpec("sequence_inventory",
        "Inventory records in authorized FASTA files/ZIPs before and after explicit path/stem filters. "
        "Report files, records, residues separately with a full hash-linked artifact; "
        "duplicate IDs are retained. State scope and unresolved conditions; no biological scope is inferred.",
        SCHEMA), sequence_inventory)
    from .sequence_query import register_sequence_query
    register_sequence_query(catalog)


def _counts():
    return dict(files=0, records=0, residues=0, gaps=0, stops=0)


def _stages():
    return {stage: _counts() for stage in STAGES}


def _add(target, source):
    for key in target:
        target[key] += source[key]


def _archive_content(content):
    """Recognize common renamed archives without decompressing or extracting them."""
    signatures = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"\x1f\x8b",
                  b"BZh", b"\xfd7zXZ\x00", b"\x28\xb5\x2f\xfd",
                  b"7z\xbc\xaf\x27\x1c", b"Rar!")
    if content.startswith(signatures):
        return True
    # Zstandard skippable frames use the little-endian magic range 0x184D2A50..5F.
    if len(content) >= 4 and 0x184D2A50 <= int.from_bytes(content[:4], "little") <= 0x184D2A5F:
        return True
    if len(content) >= tarfile.BLOCKSIZE:
        try:
            # frombuf verifies the header checksum, including legacy TAR without
            # USTAR magic, and does not read payloads or follow archive paths.
            tarfile.TarInfo.frombuf(content[:tarfile.BLOCKSIZE], "utf-8", "surrogateescape")
        except tarfile.HeaderError:
            pass
        else:
            return True
    return False


def _member_path(info, seen):
    name = info.orig_filename
    path = name[:-1] if info.is_dir() else name
    parts = path.split("/")
    if (not path or len(name) > 4096 or "\\" in name or ":" in name
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or any(part in {"", ".", ".."} for part in parts)):
        raise ValueError("Unsafe ZIP member path")
    folded = path.casefold()
    if folded in seen:
        raise ValueError("Duplicate or casefold-colliding ZIP member path")
    seen.add(folded)
    mode = info.external_attr >> 16
    kind = stat.S_IFMT(mode)
    if kind not in {0, stat.S_IFREG, stat.S_IFDIR} or (kind == stat.S_IFDIR) != info.is_dir() and kind != 0:
        raise ValueError("ZIP members must be regular files or directories; symlinks are denied")
    if info.flag_bits & 1:
        raise ValueError("Encrypted ZIP members are unsupported")
    return name


def _members(context, item, budget):
    """Read and CRC-check every entry, even excluded metadata, without extraction."""
    if Path(item.path).suffix.lower() != ".zip":
        budget["members"] += 1
        budget["expanded"] += item.size
        _check_budget(budget)
        yield Path(item.path).name, item.content, False
        return
    with zipfile.ZipFile(io.BytesIO(item.content)) as archive:
        infos = archive.infolist()
        budget["members"] += len(infos)
        budget["expanded"] += sum(info.file_size for info in infos)
        _check_budget(budget)
        seen = set()
        for info in infos:
            context.check_cancelled()
            name = _member_path(info, seen)
            if info.is_dir() and info.file_size:
                raise ValueError("ZIP directory entries cannot contain data")
            chunks, length = [], 0
            with archive.open(info) as stream:
                while True:
                    context.check_cancelled()
                    chunk = stream.read(min(65536, info.file_size + 1 - length))
                    if not chunk:
                        break
                    length += len(chunk)
                    if length > info.file_size or length > MAX_EXPANDED_BYTES:
                        raise ValueError("Expanded ZIP entry exceeds byte limit")
                    chunks.append(chunk)
            if length != info.file_size:
                raise ValueError("ZIP entry length does not match directory metadata")
            yield name, b"".join(chunks), info.is_dir()


def _check_budget(budget):
    if budget["members"] > MAX_MEMBERS:
        raise ValueError("Inventory exceeds member limit")
    if budget["expanded"] > MAX_EXPANDED_BYTES:
        raise ValueError("Inventory exceeds expanded byte limit")


def _records(context, content, selected, budget, ids):
    try:
        text = content.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ValueError("FASTA must be UTF-8 text with ASCII sequence symbols") from exc
    records, current = [], None

    def finish():
        if current is None:
            return
        if current["residues"] + current["gaps"] + current["stops"] == 0:
            raise ValueError("Every FASTA header must have a nonempty sequence")
        records.append(current)
        ids[current["record_id"]] += 1

    for line_number, raw in enumerate(io.StringIO(text)):
        if line_number % 256 == 0:
            context.check_cancelled()
        line = raw.strip(" \t\r\n")
        if not line:
            continue
        if line.startswith(">"):
            finish()
            header = line[1:].strip()
            if not header or len(header) > 4096 or any(ord(char) < 32 and char != "\t" for char in header):
                raise ValueError("FASTA requires a nonempty header of at most 4096 characters")
            budget["records"] += 1
            if budget["records"] > MAX_RECORDS:
                raise ValueError("Inventory exceeds record limit")
            current = dict(ordinal=len(records) + 1, record_id=header.split()[0],
                           selected=selected, residues=0, gaps=0, stops=0)
        else:
            if current is None or re.fullmatch(r"[A-Za-z.* \t-]+", line) is None:
                raise ValueError("Invalid FASTA sequence: header required and only alphabetic residues, gaps and stops allowed")
            sequence = line.replace(" ", "").replace("\t", "")
            gaps = sequence.count("-") + sequence.count(".")
            stops = sequence.count("*")
            current["residues"] += len(sequence) - gaps - stops
            current["gaps"] += gaps
            current["stops"] += stops
    finish()
    if not records:
        raise ValueError("FASTA must contain at least one record")
    return records


def sequence_inventory(context, paths, member_glob, scope_note, file_stems=None, unresolved_scope=None):
    context.check_cancelled()
    arguments = dict(paths=paths, member_glob=member_glob, scope_note=scope_note)
    if file_stems is not None:
        arguments["file_stems"] = file_stems
    if unresolved_scope is not None:
        arguments["unresolved_scope"] = unresolved_scope
    validator(SCHEMA).validate(arguments)
    snapshots, seen_paths, total_bytes = [], set(), 0
    for path in paths:
        context.check_cancelled()
        if Path(path).suffix.lower() not in FASTA_SUFFIXES | {".zip"}:
            raise ValueError("Inputs must be .fa, .faa, .fasta or .zip files")
        item = read_input(context, path, maximum_bytes=MAX_INPUT_BYTES)
        if item.path in seen_paths:
            raise ValueError("Input files must be unique")
        seen_paths.add(item.path)
        total_bytes += item.size
        if total_bytes > MAX_TOTAL_INPUT_BYTES:
            raise ValueError("Inventory exceeds total input byte limit")
        snapshots.append(item)

    stems = None if file_stems is None else set(file_stems)
    budget = dict(members=0, expanded=0, records=0)
    inputs, ids, totals = [], Counter(), _stages()
    excluded_total = 0
    for source_index, item in enumerate(snapshots):
        summary = {"source_index": source_index, **item.manifest(), "counts": _stages(),
                   "excluded_non_fasta_files": 0, "members": []}
        for name, content, directory in _members(context, item, budget):
            context.check_cancelled()
            suffix = PurePosixPath(name).suffix.lower()
            if not directory and (suffix in ARCHIVE_SUFFIXES or _archive_content(content)):
                raise ValueError("Nested archives are unsupported")
            kind = "directory" if directory else "fasta" if suffix in FASTA_SUFFIXES else "excluded_non_fasta"
            member = dict(path=name, sha256=hashlib.sha256(content).hexdigest(), size=len(content), kind=kind)
            summary["members"].append(member)
            if kind != "fasta":
                if kind == "excluded_non_fasta":
                    summary["excluded_non_fasta_files"] += 1
                    excluded_total += 1
                continue
            member_match = fnmatchcase(name, member_glob)
            stem_match = stems is None or PurePosixPath(name).stem in stems
            selected = member_match and stem_match
            records = _records(context, content, selected, budget, ids)
            member.update(member_filter_match=member_match, stem_filter_match=stem_match,
                          selected=selected, records=records)
            count = dict(files=1, records=len(records), **{
                key: sum(record[key] for record in records) for key in ("residues", "gaps", "stops")})
            for stage, include in zip(STAGES, (True, member_match, selected)):
                if include:
                    _add(summary["counts"][stage], count)
                    _add(totals[stage], count)
        inputs.append(summary)
    for source in inputs:
        for member in source["members"]:
            context.check_cancelled()
            for record in member.get("records", []):
                record["duplicate_record_id"] = ids[record["record_id"]] > 1
    duplicates = {"duplicate_record_ids": sum(count > 1 for count in ids.values()),
                  "records_with_duplicate_ids": sum(count for count in ids.values() if count > 1)}
    scope = dict(member_glob=member_glob, file_stems=file_stems, scope_note=scope_note,
                 unresolved_scope=unresolved_scope or [])
    document = dict(format_version=1, counts=totals, scope=scope, inputs=inputs, **duplicates,
        excluded_non_fasta_files=excluded_total, input_bytes=total_bytes,
        expanded_bytes=budget["expanded"], inspected_members=budget["members"], limitations=LIMITATIONS)
    serialized = canonical_json(document)
    context.check_cancelled()
    artifact = context.store.put_artifact(serialized, "sequence-inventory.json", "application/json",
        {"provider": "sequence_inventory", "format_version": 1, "inputs": [item.manifest() for item in snapshots],
         "limitations": LIMITATIONS}, session_id=context.session_id, run_id=context.run_id)
    # Put aggregate counts first and keep the immediate response below routine context-preview limits.
    compact_inputs = [{key: source[key] for key in ("source_index", "sha256", "counts", "excluded_non_fasta_files")}
                      for source in inputs[:2]]
    from .sequence_query import compact_result, directory_facets
    facets = directory_facets(inputs)
    return compact_result(dict(counts=totals, artifact_id=artifact["id"], **duplicates,
        directories=facets[:6], directories_omitted=max(0, len(facets) - 6),
        excluded_non_fasta_files=excluded_total, input_count=len(inputs), inputs=compact_inputs,
        inputs_omitted=max(0, len(inputs) - len(compact_inputs)),
        unresolved_scope_count=len(scope["unresolved_scope"]),
        scope_note="Full scope, all inputs, every member and record are in the artifact; no biological inference."),
        artifact)
