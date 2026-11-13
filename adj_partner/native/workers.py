"""Trusted importable job handlers; scientific computation runs in child processes."""

import hashlib
from pathlib import Path

from ..contracts import ToolResult
from ..tools.catalog import canonical_json
from . import protein
from .safe_files import read_input


def code_hashes():
    return {"adj_partner/native/" + path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(protein.__file__))}


def publish_computed(context, title, data, inputs, parameters):
    context.check_cancelled()
    provenance = {"kind": "computed", "text_availability": "computed", "job_id": context.job_id,
                  "inputs": inputs, "parameters": parameters, "method": data["method"],
                  "package_versions": data["package_versions"], "code_hashes": code_hashes(),
                  "limitations": data["limitations"]}
    data = context.store.redact(data)
    artifact = context.store.put_artifact(canonical_json(data), "computed-result.json", "application/json",
                 provenance, session_id=context.session_id, run_id=context.run_id)
    descriptor = {"artifact_id": artifact["id"], "kind": "computed", "title": title, "provenance": provenance}
    # Full values live in the immutable JSON artifact, not duplicated in job IPC.
    summary = {key: value for key, value in data.items() if key not in {"records", "composition", "parser_warnings"}}
    if "records" in data:
        summary["record_lengths"] = [{"id": r["id"], "length": r["properties"]["length"]} for r in data["records"]]
    return ToolResult(data={"source": descriptor, "summary": summary}, artifacts=[artifact])


def sequence_job(context, sequence):
    context.progress("Computing theoretical canonical-sequence properties")
    data = protein.sequence_properties(sequence)
    return publish_computed(context, "Computed protein sequence properties", data,
                            [{"sequence_sha256": data["sequence_sha256"], "length": data["length"]}],
                            {"residue_policy": "canonical20", "monoisotopic": False})


def fasta_job(context, path):
    context.progress("Reading permitted FASTA input")
    item = read_input(context, path, maximum_bytes=2 * 1024 * 1024)
    records = protein.parse_fasta(item.content)
    data = {"records": records, "method": "Bio.SeqIO.parse + Bio.SeqUtils.ProtParam.ProteinAnalysis",
            "package_versions": {"biopython": protein.Bio.__version__},
            "limitations": list(protein.SEQUENCE_LIMITATIONS)}
    return publish_computed(context, "Computed FASTA sequence properties", data, [item.manifest()],
                            {"maximum_records": 16, "maximum_total_residues": 200000})


def structure_job(context, path, format=None):
    context.progress("Inspecting permitted structure file; no prediction is performed")
    item = read_input(context, path, maximum_bytes=8 * 1024 * 1024)
    if format is None:
        suffix = Path(item.path).suffix.lower()
        if suffix not in {".pdb", ".cif", ".mmcif"}:
            raise ValueError("Specify pdb/mmcif format for an unrecognized structure extension")
        format = "pdb" if suffix == ".pdb" else "mmcif"
    data = protein.inspect_structure(item.content, format)
    return publish_computed(context, "Parsed structure-file counts", data, [item.manifest()], {"format": format})
