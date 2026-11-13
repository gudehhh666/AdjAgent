"""Numerical/parsing controls, deliberately not protein function prediction."""

import importlib
import importlib.util
import io
import json
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def api():
    assert importlib.util.find_spec("adj_partner.native"), "native shared capability package must exist"
    return importlib.import_module("adj_partner.native.protein")


def test_sequence_properties_match_recorded_independent_biopython_baseline():
    baseline = json.loads((PROJECT / "docs/refactor_2026-08-31/protein-baseline.json").read_text())
    result = api().sequence_properties("  mkwvtfisllflfssays\n")
    for key in ("length", "composition"):
        assert result[key] == baseline[key]
    for key in ("molecular_weight_da", "theoretical_isoelectric_point", "gravy", "aromaticity"):
        assert result[key] == pytest.approx(baseline[key], rel=1e-12)
    assert result["method"] == "Bio.SeqUtils.ProtParam.ProteinAnalysis"
    assert result["package_versions"]["biopython"]
    assert "theoretical" in " ".join(result["limitations"]).lower()
    assert "function" in " ".join(result["limitations"]).lower()


@pytest.mark.parametrize("sequence", ["", "  ", "MKW*", "ABCD", "AX", "AZ", "AJ", "AU", "AO", "MKW1", None])
def test_ambiguous_nonstandard_stop_and_malformed_sequences_fail_explicitly(sequence):
    with pytest.raises(ValueError):
        api().sequence_properties(sequence)


def test_sequence_resource_limit_is_explicit():
    with pytest.raises(ValueError, match="[Ll]imit|[Ll]ong|[Ll]arge|[Rr]esidue"):
        api().sequence_properties("A" * 100001)


def test_actual_fasta_and_multiple_record_policy():
    content = (PROJECT / "tests/fixtures/protein_example.fasta").read_bytes()
    records = api().parse_fasta(content)
    assert len(records) == 1 and records[0]["properties"]["length"] == 18
    multiple = api().parse_fasta(b">one\nACDE\n>two\nMKWV\n")
    assert [record["id"] for record in multiple] == ["one", "two"]
    assert [record["properties"]["length"] for record in multiple] == [4, 4]


@pytest.mark.parametrize("content", [b"ACDE", b"", b">missing\n", b">bad\nABCD\n", b">duplicate\nAA\n>duplicate\nCC\n"])
def test_invalid_fasta_is_not_silently_a_success(content):
    with pytest.raises(ValueError):
        api().parse_fasta(content)


def test_actual_pdb_fixture_is_inspection_not_prediction():
    content = (PROJECT / "tests/fixtures/structure_example.pdb").read_bytes()
    result = api().inspect_structure(content, "pdb")
    assert {key: result[key] for key in ("models", "chains", "residues", "atoms")} == {
        "models": 1, "chains": 1, "residues": 2, "atoms": 4}
    assert isinstance(result["parser_warnings"], list)
    assert "not" in " ".join(result["limitations"]).lower()
    assert "prediction" in " ".join(result["limitations"]).lower()


def test_actual_mmcif_parser_matches_the_same_synthetic_structure():
    from Bio.PDB import MMCIFIO, PDBParser
    structure = PDBParser(QUIET=True).get_structure("synthetic", PROJECT / "tests/fixtures/structure_example.pdb")
    output = io.StringIO()
    writer = MMCIFIO()
    writer.set_structure(structure)
    writer.save(output)
    result = api().inspect_structure(output.getvalue().encode(), "mmcif")
    assert result["models"] == 1 and result["chains"] == 1
    assert result["residues"] == 2 and result["atoms"] == 4


@pytest.mark.parametrize("content,format", [(b"nothing", "pdb"), (b"", "mmcif"), (b"nothing", "pickle")])
def test_invalid_or_unsupported_structure_is_not_a_placeholder(content, format):
    with pytest.raises(ValueError):
        api().inspect_structure(content, format)


def noisy_pdb(records):
    atom = next(line for line in (PROJECT / "tests/fixtures/structure_example.pdb").read_bytes().splitlines() if line.startswith(b"ATOM"))
    return atom + b"\n" + b"X\n" * records + b"END\n"


def test_concurrent_structure_warning_counts_stay_with_their_actual_inputs():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier = Barrier(2)
    def parse(count):
        content = noisy_pdb(count)
        barrier.wait()
        return api().inspect_structure(content, "pdb")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(parse, (20000, 30000)))
    assert [result["parser_warning_count"] for result in results] == [20001, 30001]
    assert all(len(result["parser_warnings"]) == 50 and result["warnings_truncated"] for result in results)


def test_warning_samples_are_bounded_during_collection_not_only_on_output():
    import tracemalloc
    # Warm imports separately; this measures parser/collector allocations.
    inspector = api().inspect_structure
    content = noisy_pdb(20000)
    tracemalloc.start()
    try:
        result = inspector(content, "pdb")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result["parser_warning_count"] == 20001
    assert len(result["parser_warnings"]) == 50
    assert peak < 5 * 1024 * 1024, f"Warning capture unexpectedly retained {peak} bytes"
