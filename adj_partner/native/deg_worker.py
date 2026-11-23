"""Adapter of unchanged deterministic legacy DEG math over a private snapshot.

Only cached parquet layout is supported. No legacy bootstrap, coordinator,
generated Python, source-cache creation or Excel conversion is invoked.
"""

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow
import scipy

from ..contracts import ToolResult
from ..tools.catalog import canonical_json
from .safe_files import list_inputs, read_input


PROJECT = Path(__file__).resolve().parents[2]
AGEING = "000_Young_ACTVSOld_ACT"
SOLVENT = "001_DMSOVSOld_ACT"
MANIFEST = Path(__file__).with_name("deg-cache-manifest.json")


def verify_cache_manifest(records):
    """This legacy adapter accepts only one reviewed experiment, not arbitrary parquet.

    Byte verification precedes ALL parquet decoding, including metadata parsing.
    A new experiment needs a separately reviewed adapter/manifest version.
    """
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    actual = [{"name": item.get("name", Path(item.get("path", "")).name),
               "size": item["size"], "sha256": item["sha256"]} for item in records]
    if sorted(actual, key=lambda item: item["name"]) != expected:
        raise ValueError("DEG adapter accepts only the reviewed original experiment cache bytes")


def _snapshot(context, dataset_dir):
    original_cache = Path(dataset_dir) / "cache"
    paths = list_inputs(context, original_cache, suffix=".parquet", maximum_files=64)
    names = {Path(path).name for path in paths}
    required = {AGEING + ".parquet", SOLVENT + ".parquet", "_annotation.parquet"}
    if not required <= names or not any(name.endswith("VSDMSO.parquet") for name in names):
        raise ValueError("DEG cache requires ageing/vehicle contrasts, annotation and compound VSDMSO tables")
    if any(not re.fullmatch(r"[A-Za-z0-9_]+\.parquet", name) for name in names):
        raise ValueError("Unsupported DEG cache filename")
    if any(name not in required and not name.endswith("VSDMSO.parquet") for name in names):
        raise ValueError("Unsupported extra contrasts in configured DEG cache")
    private = context.check_path(context.workspace / "inputs" / "cache", write=True, must_exist=False)
    private.mkdir(mode=0o700, parents=True, exist_ok=False)
    inputs, total = [], 0
    for number, path in enumerate(paths, 1):
        item = read_input(context, path, maximum_bytes=16 * 1024 * 1024)
        total += item.size
        if total > 96 * 1024 * 1024:
            raise ValueError("DEG inputs exceed 96 MiB snapshot limit")
        with (private / Path(path).name).open("xb") as target:
            target.write(item.content)
        inputs.append(item.manifest())
        context.progress("Copied verified dataset input", completed=number, total=len(paths))
    verify_cache_manifest(inputs)
    # A fixed packaged method definition, not a user-selected path or setting.
    module_path = PROJECT / "environment/knowhow/immune_modules.json"
    module_bytes = module_path.read_bytes()
    if len(module_bytes) > 1024 * 1024:
        raise ValueError("Packaged immune-module definition exceeds size limit")
    modules = json.loads(module_bytes)["modules"]
    canonical_json(modules)
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Packaged immune-module definition is malformed")
    for definition in modules.values():
        if (not isinstance(definition, dict) or not isinstance(definition.get("keywords"), list)
                or not all(isinstance(item, str) for item in definition["keywords"])):
            raise ValueError("Immune module keywords must be strings")
    with (private.parent / "immune_modules.json").open("xb") as target:
        target.write(module_bytes)
    inputs.append({"path": str(module_path), "sha256": hashlib.sha256(module_bytes).hexdigest(), "size": len(module_bytes),
                   "role": "packaged_method_definition"})
    return private, inputs, modules


def _validate_cache(context, loader):
    ageing = loader.load_comparison(AGEING)
    needed = set(loader.DEG_COLS)
    if "gene_id" not in ageing or ageing["gene_id"].isna().any() or ageing["gene_id"].duplicated().any():
        raise ValueError("Ageing contrast requires unique non-null gene identifiers")
    genes = set(ageing["gene_id"])
    if not 1 <= len(genes) <= 100000:
        raise ValueError("Supported DEG cache has 1..100000 genes")
    for name in loader.list_comparisons():
        context.check_cancelled()
        frame = loader.load_comparison(name)
        if not needed <= set(frame.columns) or frame["gene_id"].duplicated().any() or set(frame["gene_id"]) != genes:
            raise ValueError("DEG contrasts require complete aligned unique gene identifiers and expected columns")
        fpkm = loader.fpkm_columns(frame)
        expected = ({"FPKM.Young_ACT", "FPKM.Old_ACT"} if name == AGEING else
                    {"FPKM.DMSO", "FPKM.Old_ACT"} if name == SOLVENT else None)
        if len(fpkm) != 2 or (expected is not None and set(fpkm) != expected) or (expected is None and "FPKM.DMSO" not in fpkm):
            raise ValueError("Unsupported sample/replicate layout in DEG cache")
        for column in ["fc", "log2(fc)", "pval", "qval", *fpkm]:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                raise ValueError("DEG numeric columns must have numeric types")
        if not np.isfinite(frame[fpkm].to_numpy()).all() or (frame[fpkm].to_numpy() < 0).any():
            raise ValueError("Expression values must be finite and nonnegative")
    annotation = loader.load_annotation()
    if not {"gene_id", "gene_name", "GO", "KEGG", "Reactome"} <= set(annotation.columns):
        raise ValueError("Annotation cache is missing required columns")
    if annotation["gene_id"].duplicated().any() or set(annotation["gene_id"]) != genes:
        raise ValueError("Annotation gene identifiers do not align with contrasts")
    comparisons = loader.list_comparisons()
    compounds = [name for name in comparisons if name.endswith("VSDMSO")]
    solvent = loader.load_comparison(SOLVENT).set_index("gene_id").reindex(ageing["gene_id"])
    if (len(compounds) != 26 or len(comparisons) != 28
            or not np.array_equal(ageing["FPKM.Old_ACT"].to_numpy(), solvent["FPKM.Old_ACT"].to_numpy())
            or int((solvent["significant"] == "yes").sum()) <= int((ageing["significant"] == "yes").sum())):
        raise ValueError("Cache differs from the supported experiment's compound, background or vehicle-count design")
    for name in compounds:
        dmso = loader.load_comparison(name).set_index("gene_id").reindex(ageing["gene_id"])["FPKM.DMSO"]
        if not np.array_equal(dmso.to_numpy(), solvent["FPKM.DMSO"].to_numpy()):
            raise ValueError("Cache differs from the supported experiment's shared DMSO control")


def _finite_json(value):
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def deg_job(context, dataset_dir, top_n=250, n_permutations=2000):
    context.progress("Preparing a private verified parquet snapshot; source data remains read-only")
    private, inputs, modules = _snapshot(context, dataset_dir)
    # Imported only in this independent child. Their registered legacy tools
    # are never used as an executor; only deterministic functions are invoked.
    from environment.tools.data import deg_loader
    from environment.tools.analysis import deg_bio, adjuvant_scoring
    deg_loader.cache_dir = lambda: str(private)
    deg_loader.deg_dir = lambda: str(private.parent)
    deg_loader.ensure_cache = lambda *args, **kwargs: len(inputs) - 2
    deg_bio.load_immune_modules = lambda: modules
    _validate_cache(context, deg_loader)
    context.progress("Running unchanged deterministic reversal, controls and immune-module calculations")
    pack = deg_bio.evidence_pack(top_n=top_n, n_permutations=n_permutations)
    context.check_cancelled()
    statistics = dict(pack["stats"])
    statistics["n_significant_compounds_two_sided"] = statistics.pop("n_significant_genes")
    centered = pack["tables"]["panel_centered_reversal"]
    significant = centered[centered["perm_q"] < .05]
    limitations = [*pack["design"]["caveats"],
        "Exploratory in-vitro association/ranking only; mechanism, reproducibility and in-vivo vaccine benefit remain hypotheses.",
        "Permutation FDR is two-sided across compounds, not a count of significant genes or only positive reversers.",
        "No-compound controls qualify raw reversal scores; panel centering is a relative comparison among this compound panel.",
        "Immune-module scores are transcript-based associations, not verified pathway activation, cell targeting or safety.",
        "Any undefined nonfinite legacy output is represented as JSON null, not an invented zero.",
        "Legacy interpretive prose is retained as unassessed attributed narrative, not causal evidence."]
    parameters = {"top_n": top_n, "n_permutations": n_permutations, "seed": 0,
                  "raw_reversal_permutations": max(500, n_permutations // 4)}
    versions = {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "pyarrow": pyarrow.__version__}
    method = "environment.tools.analysis.deg_bio.evidence_pack (unchanged deterministic math)"
    code_paths = [Path(__file__), Path(deg_bio.__file__), Path(deg_loader.__file__), Path(adjuvant_scoring.__file__)]
    provenance = {"job_id": context.job_id, "text_availability": "computed", "inputs": inputs,
                  "dataset_adapter": "recorded-aged-splenocyte-26-v1",
                  "approved_cache_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                  "method": method, "parameters": parameters, "package_versions": versions, "limitations": limitations,
                  "code_hashes": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
                  "input_manifest_sha256": hashlib.sha256(canonical_json(inputs).encode()).hexdigest()}
    artifacts = []
    for name, table in pack["tables"].items():
        artifacts.append(context.store.put_artifact(table.to_csv(index=False, lineterminator="\n"), name + ".csv", "text/csv",
            {**provenance, "table": name}, session_id=context.session_id, run_id=context.run_id))
    output = _finite_json({"method": method, "parameters": parameters, "package_versions": versions,
        "statistics": statistics, "design": pack["design"], "controls": pack["controls"],
        "tables": {name: table.to_dict(orient="records") for name, table in pack["tables"].items()},
        "positive_reversers": significant[significant["centered_reversal_r"] > 0].to_dict(orient="records"),
        "negative_associations": significant[significant["centered_reversal_r"] < 0].to_dict(orient="records"),
        "shared_core": pack["shared_core"], "limitations": limitations,
        "legacy_interpretations": {"status": "unassessed_attributed_narrative", "records": pack["evidence"]},
        "table_artifact_ids": {name: artifact["id"] for name, artifact in zip(pack["tables"], artifacts)}})
    summary = _finite_json({
        "ranking_metric": "centered_reversal_r_descending",
        "selection_guidance": "Rank candidates by top_centered/positive_reversers. Immune-module scores are descriptive associations and must not replace the panel-centered reversal ranking.",
        "recommended_exploration_tool": "deg_explore",
        "statistics": statistics, "design": pack["design"],
        "controls": pack["controls"], "top_centered": output["tables"]["panel_centered_reversal"][:5],
        "positive_reversers": output["positive_reversers"], "negative_associations": output["negative_associations"],
        "limitations": limitations})
    # Keep the exact model-facing decision summary addressable through the
    # registered immutable source, so evidence_select('/summary/...') has the
    # same meaning as the tool result shown to the model.
    output["summary"] = summary
    artifact = context.store.put_artifact(canonical_json(context.store.redact(output)), "deg-computed-result.json", "application/json",
                                         provenance, session_id=context.session_id, run_id=context.run_id)
    artifacts.insert(0, artifact)
    descriptor = {"artifact_id": artifact["id"], "kind": "computed", "title": "Exploratory aged-splenocyte DEG analysis", "provenance": provenance}
    return ToolResult(data={"source": descriptor, "summary": summary}, artifacts=artifacts)


def deg_explore_job(context, dataset_dir, compound, direction="both", annotation="KEGG",
                    top_genes=10, top_terms=8):
    """Deterministically inspect one ranked compound without arbitrary model code."""
    context.progress("Preparing a private verified parquet snapshot; source data remains read-only")
    private, inputs, _ = _snapshot(context, dataset_dir)
    from environment.tools.data import deg_loader
    from environment.tools.analysis import deg_bio
    deg_loader.cache_dir = lambda: str(private)
    deg_loader.deg_dir = lambda: str(private.parent)
    deg_loader.ensure_cache = lambda *args, **kwargs: len(inputs) - 2
    _validate_cache(context, deg_loader)
    compounds = {deg_loader.compound_name(name): name for name in deg_loader.compound_comparisons()}
    if compound not in compounds:
        raise ValueError("Compound is not part of the reviewed 26-compound experiment")
    if direction not in {"up", "down", "both"}:
        raise ValueError("direction must be up, down or both")
    if annotation not in {"GO", "KEGG", "Reactome"}:
        raise ValueError("annotation must be GO, KEGG or Reactome")
    comparison = compounds[compound]
    frame = deg_loader.load_comparison(comparison)
    significant = frame[frame["significant"] == "yes"].copy()
    selected = significant if direction == "both" else significant[significant["regulation"] == direction]
    selected["_abs_log2fc"] = selected["log2(fc)"].abs()
    gene_columns = ["gene_id", "gene_name", "log2(fc)", "qval", "regulation"]
    genes = selected.sort_values(["_abs_log2fc", "qval"], ascending=[False, True]).head(top_genes)[gene_columns]
    genes = genes.rename(columns={"log2(fc)": "log2fc"})
    enrichment = deg_bio.enrich_comparison(comparison, None if direction == "both" else direction,
                                            annotation, top=top_terms)
    terms = enrichment["terms"]
    profile = deg_loader.profile_deg_dataset(comparison)
    limitations = [
        "Exploratory in-vitro transcriptomic association only; this result does not establish mechanism, vaccine benefit or safety.",
        "There is one library per group, so vendor p/q values reflect within-library sampling rather than biological reproducibility.",
        "All compound contrasts share one DMSO denominator and the vehicle effect is larger than the ageing DEG count.",
        "Top genes are vendor-significant genes ordered by absolute log2 fold change; pathway terms are over-representation results, not causal pathway activation.",
    ]
    parameters = {"compound": compound, "comparison": comparison, "direction": direction,
                  "annotation": annotation, "top_genes": top_genes, "top_terms": top_terms}
    code_paths = [Path(__file__), Path(deg_bio.__file__), Path(deg_loader.__file__)]
    provenance = {"job_id": context.job_id, "text_availability": "computed", "inputs": inputs,
        "dataset_adapter": "recorded-aged-splenocyte-26-v1",
        "approved_cache_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "method": "reviewed DEG table profiling plus environment.tools.analysis.deg_bio.enrich_comparison",
        "parameters": parameters, "limitations": limitations,
        "code_hashes": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths},
        "input_manifest_sha256": hashlib.sha256(canonical_json(inputs).encode()).hexdigest()}
    gene_artifact = context.store.put_artifact(genes.to_csv(index=False, lineterminator="\n"),
        f"{compound}-top-genes.csv", "text/csv", {**provenance, "table": "top_genes"},
        session_id=context.session_id, run_id=context.run_id)
    term_frame = pd.DataFrame(terms)
    term_artifact = context.store.put_artifact(term_frame.to_csv(index=False, lineterminator="\n"),
        f"{compound}-{annotation}-{direction}-enrichment.csv", "text/csv",
        {**provenance, "table": "enrichment_terms"}, session_id=context.session_id, run_id=context.run_id)
    summary = _finite_json({"compound": compound, "comparison": comparison, "direction": direction,
        "annotation": annotation, "profile": profile, "top_genes": genes.to_dict(orient="records"),
        "enrichment_query_genes": enrichment["n_query_genes"], "enrichment_terms": terms,
        "limitations": limitations,
        "interpretation": "Candidate-specific deterministic exploration; use results to form testable proposals, not validated claims."})
    output = {"summary": summary, "parameters": parameters,
              "artifact_ids": {"top_genes": gene_artifact["id"], "enrichment_terms": term_artifact["id"]},
              "provenance": provenance}
    result_artifact = context.store.put_artifact(canonical_json(context.store.redact(output)),
        f"{compound}-deg-exploration.json", "application/json", provenance,
        session_id=context.session_id, run_id=context.run_id)
    artifacts = [result_artifact, gene_artifact, term_artifact]
    descriptor = {"artifact_id": result_artifact["id"], "kind": "computed",
                  "title": f"Exploratory DEG details for {compound}", "provenance": provenance}
    return ToolResult(data={"source": descriptor, "summary": summary}, artifacts=artifacts)
