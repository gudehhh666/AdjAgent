"""Deterministic DEG integration against the recorded real local baseline."""

import hashlib
import importlib
import json
import os
from pathlib import Path
from pathlib import Path

import pytest

from adj_partner.contracts import ToolCall
from adj_partner.native.capabilities import NativeCapabilities
from adj_partner.tools import ToolCatalog, ToolExecutor
from test_native_files import context
from test_native_integration import setup, complete, PROJECT


DATASET = Path(os.environ.get("ADJ_TEST_DATASET_DIR", str(PROJECT / "data/gene_diff"))).expanduser().resolve()


def test_legacy_decoder_only_accepts_the_reviewed_original_cache_bytes():
    from adj_partner.native import deg_worker
    assert hasattr(deg_worker, "verify_cache_manifest")
    records = json.loads((PROJECT / "adj_partner/native/deg-cache-manifest.json").read_text())
    deg_worker.verify_cache_manifest(records)
    changed = [dict(item) for item in records]
    changed[0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reviewed"):
        deg_worker.verify_cache_manifest(changed)
    with pytest.raises(ValueError, match="reviewed"):
        deg_worker.verify_cache_manifest(records[:-1])
    with pytest.raises(ValueError, match="reviewed"):
        deg_worker.verify_cache_manifest(records + [records[0]])


@pytest.mark.parametrize("change", ["compound_control", "old_control", "vehicle_counts", "compound_count"])
def test_legacy_design_assumptions_are_explicitly_verified(change):
    import pandas as pd
    from types import SimpleNamespace
    from adj_partner.native.deg_worker import _validate_cache, AGEING, SOLVENT
    columns = ["gene_id", "gene_name", "fc", "log2(fc)", "pval", "qval", "regulation", "significant"]
    base = pd.DataFrame({"gene_id": ["a", "b"], "gene_name": ["A", "B"], "fc": [1., 2.],
        "log2(fc)": [0., 1.], "pval": [.1, .2], "qval": [.1, .2], "regulation": ["up", "up"],
        "significant": ["no", "no"]})
    frames = {AGEING: base.assign(**{"FPKM.Young_ACT": [2., 3.], "FPKM.Old_ACT": [1., 2.]}),
              SOLVENT: base.assign(**{"FPKM.DMSO": [3., 4.], "FPKM.Old_ACT": [1., 2.]})}
    frames[SOLVENT]["significant"] = ["yes", "no"]
    for index in range(26):
        frames[f"c{index}VSDMSO"] = base.assign(**{f"FPKM.c{index}": [2., 3.], "FPKM.DMSO": [3., 4.]})
    if change == "compound_control": frames["c0VSDMSO"]["FPKM.DMSO"] = [9., 9.]
    if change == "old_control": frames[SOLVENT]["FPKM.Old_ACT"] = [9., 9.]
    if change == "vehicle_counts": frames[SOLVENT]["significant"] = ["no", "no"]
    if change == "compound_count": frames.pop("c0VSDMSO")
    loader = SimpleNamespace(DEG_COLS=columns, load_comparison=frames.__getitem__,
        list_comparisons=lambda: list(frames), fpkm_columns=lambda f: [c for c in f if c.startswith("FPKM")],
        load_annotation=lambda: base.assign(GO="", KEGG="", Reactome=""))
    with pytest.raises(ValueError, match="supported experiment"):
        _validate_cache(SimpleNamespace(check_cancelled=lambda: None), loader)


def test_safe_directory_listing_is_bounded_and_rejects_symlink_entries(tmp_path):
    from adj_partner.native import safe_files
    assert hasattr(safe_files, "list_inputs"), "bounded descriptor directory listing must exist"
    ctx = context(tmp_path)
    folder = tmp_path / "cache"
    folder.mkdir()
    (folder / "a.parquet").write_bytes(b"synthetic")
    assert safe_files.list_inputs(ctx, folder, suffix=".parquet", maximum_files=2) == [str(folder / "a.parquet")]
    (folder / "b.parquet").symlink_to(folder / "a.parquet")
    with pytest.raises(PermissionError):
        safe_files.list_inputs(ctx, folder, suffix=".parquet", maximum_files=2)
    (folder / "b.parquet").unlink()
    (folder / "b.parquet").write_bytes(b"synthetic")
    with pytest.raises(ValueError):
        safe_files.list_inputs(ctx, folder, suffix=".parquet", maximum_files=1)


def test_missing_cache_fails_explicitly_without_creating_source_cache(tmp_path):
    assert importlib.util.find_spec("adj_partner.native.deg_worker"), "native DEG job handler must exist"
    store, research, _, _, sid, rid = setup(tmp_path)
    dataset = tmp_path / "xlsx_only"
    dataset.mkdir()
    native = NativeCapabilities(store, research=research, dataset_dir=dataset)
    catalog = ToolCatalog()
    native.register(catalog)
    result = ToolExecutor(store, catalog).execute(ToolCall("missing", "deg_analyze", {}), sid, rid)
    final = complete(native, sid, rid, result.job_id)
    assert final.status == "error"
    assert not (dataset / "cache").exists() and research.list_sources(sid, rid) == []


def test_duplicate_deg_intent_in_one_run_reuses_the_same_durable_job(tmp_path):
    store, research, _, _, sid, rid = setup(tmp_path)
    dataset = tmp_path / "empty-dataset"
    dataset.mkdir()
    native = NativeCapabilities(store, research=research, dataset_dir=dataset)
    catalog = ToolCatalog()
    native.register(catalog)
    executor = ToolExecutor(store, catalog)
    arguments = {"top_n": 250, "n_permutations": 500}
    first = executor.execute(ToolCall("deg-first", "deg_analyze", arguments), sid, rid)
    second = executor.execute(ToolCall("deg-second", "deg_analyze", arguments), sid, rid)
    complete(native, sid, rid, first.job_id)
    if second.job_id != first.job_id:
        complete(native, sid, rid, second.job_id)
    assert second.job_id == first.job_id


@pytest.mark.skipif(not (DATASET / "cache").is_dir(), reason="Real private DEG cache not distributed with source")
def test_real_native_deg_job_matches_baseline_and_leaves_all_58_source_files_unchanged(tmp_path):
    assert importlib.util.find_spec("adj_partner.native.deg_worker"), "native DEG job handler must exist"
    store, research, _, _, sid, rid = setup(tmp_path)
    session = store.get_session(sid)
    config = {**session["config"], "allowed_roots": [str(tmp_path), str(DATASET)]}
    store.update_session(sid, config, session["revision"])
    native = NativeCapabilities(store, research=research, dataset_dir=DATASET)
    catalog = ToolCatalog()
    native.register(catalog)
    executor = ToolExecutor(store, catalog)
    arguments = {"top_n": 250, "n_permutations": 500}
    pending = executor.execute(ToolCall("real-deg", "deg_analyze", arguments), sid, rid)
    assert pending.status == "pending", pending.to_dict()
    result = complete(native, sid, rid, pending.job_id, timeout=120)
    assert result.status == "ok", result.to_dict()
    assert result.data["summary"]["ranking_metric"] == "centered_reversal_r_descending"
    assert [row["compound"] for row in result.data["summary"]["top_centered"]] == [
        "Zincgluconate", "Saxagliptin", "Flunarizine", "Ezetimibe", "Betaine"]
    assert "immune" in result.data["summary"]["selection_guidance"].lower()
    from adj_partner.runtime.context import ContextView
    message = ContextView(store).tool_message(sid, rid, "real-deg", result.to_dict())
    projected = json.loads(message["content"])
    assert projected["data"]["summary"]["top_centered"][0]["compound"] == "Zincgluconate"
    assert len(message["content"].encode("utf-8")) <= 8000
    source = research.get_source(sid, rid, result.data["source_id"])
    output = json.loads(store.read_artifact(source["artifact_id"]))
    assert output["design"]["n_compounds"] == 26 and output["design"]["n_genes"] == 34899
    assert output["summary"]["top_centered"][0]["compound"] == "Zincgluconate"
    assert output["design"]["replicates_per_group"] == 1 and output["design"]["shared_dmso_denominator"] is True
    expected = [("Zincgluconate", .35670675702367227), ("Saxagliptin", .3484536206089971),
                ("Flunarizine", .24535323759342373), ("Ezetimibe", .2371169774976519), ("Betaine", .22765660544763183)]
    table = output["tables"]["panel_centered_reversal"]
    for row, (compound, score) in zip(table[:5], expected):
        assert row["compound"] == compound and row["centered_reversal_r"] == pytest.approx(score, abs=1e-12)
    assert output["controls"]["Old_ACT"] == pytest.approx(.522717588574134, abs=1e-12)
    assert output["statistics"]["n_significant_compounds_two_sided"] == 17
    assert "n_significant_genes" not in output["statistics"]
    assert all(row["centered_reversal_r"] > 0 for row in output["positive_reversers"])
    assert all(row["centered_reversal_r"] < 0 for row in output["negative_associations"])
    assert {key: len(value) for key, value in output["tables"].items()} == {
        "panel_centered_reversal": 26, "raw_reversal": 26, "immune_modules": 26}
    assert len(source["provenance"]["inputs"]) == 30  # 29 caches + method resource
    assert len([a for a in result.artifacts if a["media_type"] == "text/csv"]) == 3
    for line in (PROJECT / "docs/refactor_2026-08-31/dataset-before.sha256").read_text().splitlines():
        digest, path = line.split("  ", 1)
        original_prefix = "../ADJ/gene_diff/"
        selected = DATASET / path[len(original_prefix):] if path.startswith(original_prefix) else PROJECT / path
        assert hashlib.sha256(selected.read_bytes()).hexdigest() == digest
    assert native.recover_action(sid, rid, ToolCall("real-deg", "deg_analyze", arguments)).data["source_id"] == source["id"]


@pytest.mark.skipif(not (DATASET / "cache").is_dir(), reason="Real private DEG cache not distributed with source")
def test_real_deg_explore_returns_bounded_compound_genes_and_pathways(tmp_path):
    store, research, _, _, sid, rid = setup(tmp_path)
    session = store.get_session(sid)
    store.update_session(sid, {**session["config"], "allowed_roots": [str(tmp_path), str(DATASET)]}, session["revision"])
    native = NativeCapabilities(store, research=research, dataset_dir=DATASET)
    catalog = ToolCatalog()
    native.register(catalog)
    executor = ToolExecutor(store, catalog)
    pending = executor.execute(ToolCall("explore-zinc", "deg_explore", {
        "compound": "Zincgluconate", "direction": "up", "annotation": "KEGG",
        "top_genes": 8, "top_terms": 5}), sid, rid)
    assert pending.status == "pending", pending.to_dict()
    result = complete(native, sid, rid, pending.job_id, timeout=120)
    assert result.status == "ok", result.to_dict()
    summary = result.data["summary"]
    assert summary["compound"] == "Zincgluconate" and summary["comparison"] == "ZincgluconateVSDMSO"
    assert summary["direction"] == "up" and summary["annotation"] == "KEGG"
    assert 0 < len(summary["top_genes"]) <= 8 and len(summary["enrichment_terms"]) <= 5
    assert all({"gene_id", "gene_name", "log2fc", "qval", "regulation"} <= row.keys()
               for row in summary["top_genes"])
    assert summary["limitations"] and result.data["source_id"]
    from adj_partner.runtime.context import ContextView
    message = ContextView(store).tool_message(sid, rid, "explore-zinc", result.to_dict())
    assert "Zincgluconate" in message["content"] and len(message["content"].encode("utf-8")) <= 8000
