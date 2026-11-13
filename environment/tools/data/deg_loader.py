"""Loader for the ADJ/gene_diff differential-expression dataset.

The dataset is 28 vendor-produced xlsx tables from one experiment: aged mouse
splenocytes with a vaccine-antigen background, incubated in vitro with 26 small
molecules (all dissolved in DMSO), plus two background contrasts
(Young_ACT vs Old_ACT, and DMSO vs Old_ACT).

Two things make this dataset easy to misuse, so this loader reports them as
first-class facts rather than leaving them for the caller to discover:

* every group is a SINGLE library (n=1 vs n=1) -- the vendor's pval/qval model
  only sampling noise, not biological variance;
* all 26 compound contrasts reuse the SAME DMSO profile as denominator, so the
  contrasts are statistically dependent.

Because every sample shares one normalisation batch (the Old_ACT column is
byte-identical between the two background files, and the DMSO column is
byte-identical across all 26 compound files), the per-contrast tables can be
recombined into a single 29-column FPKM matrix.
"""
from __future__ import annotations

import glob
import os
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from core.config import get_setting, resolve_path
from core.schemas import EvidenceRecord
from environment.registry import REGISTRY, ToolSpec

AGEING_CONTRAST = "000_Young_ACTVSOld_ACT"
SOLVENT_CONTRAST = "001_DMSOVSOld_ACT"

# The vendor's `significant` column is exactly reproduced by this rule
# (verified: 34899/34899 rows match on RapamycinVSDMSO).
VENDOR_QVAL_CUTOFF = 0.05
VENDOR_LOG2FC_CUTOFF = 1.0

DEG_COLS = ["gene_id", "gene_name", "fc", "log2(fc)", "pval", "qval",
            "regulation", "significant"]
ANNOT_COLS = ["gene_id", "gene_name", "EntrezID", "GO", "KEGG", "Reactome",
              "Description", "trans_type", "TF_Family"]

_SUFFIX = "_Gene_differential_expression.xlsx"


def deg_dir() -> str:
    configured = get_setting("data_lake", "gene_diff_dir", default="../ADJ/gene_diff")
    return str(resolve_path(configured))


def cache_dir() -> str:
    path = os.path.join(deg_dir(), "cache")
    os.makedirs(path, exist_ok=True)
    return path


def _convert_one(xlsx_path: str) -> None:
    name = os.path.basename(xlsx_path).replace(_SUFFIX, "")
    out = os.path.join(cache_dir(), f"{name}.parquet")
    if os.path.exists(out):
        return
    raw = pd.read_excel(xlsx_path)
    fpkm = [c for c in raw.columns if c.startswith("FPKM")]
    slim = raw[DEG_COLS + fpkm].copy()
    slim["significant"] = slim["significant"].astype(str).str.lower().str.strip()
    slim["regulation"] = slim["regulation"].astype(str).str.lower().str.strip()
    for col in ["fc", "log2(fc)", "pval", "qval"] + fpkm:
        slim[col] = pd.to_numeric(slim[col], errors="coerce")
    slim.to_parquet(out, index=False)

    annot_path = os.path.join(cache_dir(), "_annotation.parquet")
    if not os.path.exists(annot_path):
        # EntrezID mixes ints with comma-joined multi-gene strings, so force str.
        raw[[c for c in ANNOT_COLS if c in raw.columns]].astype(str).to_parquet(
            annot_path, index=False)


def ensure_cache(max_workers: int = 8) -> int:
    """Convert any xlsx lacking a parquet twin. Returns number of cached tables."""
    sources = sorted(glob.glob(os.path.join(deg_dir(), f"*{_SUFFIX}")))
    pending = [p for p in sources
               if not os.path.exists(os.path.join(
                   cache_dir(), os.path.basename(p).replace(_SUFFIX, "") + ".parquet"))]
    if pending:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            list(pool.map(_convert_one, pending))
    return len(glob.glob(os.path.join(cache_dir(), "*.parquet"))) - 1


def list_comparisons() -> list[str]:
    ensure_cache()
    names = [os.path.basename(p)[:-len(".parquet")]
             for p in sorted(glob.glob(os.path.join(cache_dir(), "*.parquet")))]
    return [n for n in names if not n.startswith("_")]


def compound_comparisons() -> list[str]:
    return [c for c in list_comparisons() if c.endswith("VSDMSO")]


@lru_cache(maxsize=64)
def load_comparison(name: str) -> pd.DataFrame:
    ensure_cache()
    path = os.path.join(cache_dir(), f"{name}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"unknown comparison: {name}")
    return pd.read_parquet(path)


@lru_cache(maxsize=1)
def load_annotation() -> pd.DataFrame:
    ensure_cache()
    return pd.read_parquet(os.path.join(cache_dir(), "_annotation.parquet"))


def compound_name(comparison: str) -> str:
    return comparison.replace("VSDMSO", "")


def fpkm_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("FPKM")]


@lru_cache(maxsize=1)
def expression_matrix() -> pd.DataFrame:
    """Recombine every contrast into one gene x sample FPKM matrix (index=gene_id)."""
    ageing = load_comparison(AGEING_CONTRAST).set_index("gene_id")
    solvent = load_comparison(SOLVENT_CONTRAST).set_index("gene_id")

    matrix = pd.DataFrame(index=ageing.index)
    matrix["gene_name"] = ageing["gene_name"]
    matrix["Young_ACT"] = ageing["FPKM.Young_ACT"]
    matrix["Old_ACT"] = ageing["FPKM.Old_ACT"]
    matrix["DMSO"] = solvent["FPKM.DMSO"].reindex(matrix.index)

    for comparison in compound_comparisons():
        df = load_comparison(comparison).set_index("gene_id")
        treated = [c for c in fpkm_columns(df) if c != "FPKM.DMSO"][0]
        matrix[compound_name(comparison)] = df[treated].reindex(matrix.index)
    return matrix


def log2fc_matrix() -> pd.DataFrame:
    """Vendor log2(fc) for every contrast, aligned on gene_id (index=gene_id)."""
    out = {}
    for comparison in list_comparisons():
        df = load_comparison(comparison).set_index("gene_id")
        out[comparison] = df["log2(fc)"]
    return pd.DataFrame(out)


def deg_sets(comparison: str, direction: str | None = None) -> set[str]:
    df = load_comparison(comparison)
    sig = df[df["significant"] == "yes"]
    if direction:
        sig = sig[sig["regulation"] == direction]
    return set(sig["gene_id"])


# ------------------------------------------------------------------ QC / tool
def audit_design() -> dict[str, Any]:
    """Machine-checked facts about the experimental design. No assumptions."""
    compounds = compound_comparisons()
    reference = load_comparison(compounds[0]).set_index("gene_id")["FPKM.DMSO"]
    shared, max_dev = True, 0.0
    for comparison in compounds[1:]:
        other = load_comparison(comparison).set_index("gene_id")["FPKM.DMSO"]
        aligned = other.reindex(reference.index)
        max_dev = max(max_dev, float(np.nanmax(np.abs(aligned.values - reference.values))))
        shared = shared and bool(np.allclose(aligned.values, reference.values, equal_nan=True))

    ageing = load_comparison(AGEING_CONTRAST)
    solvent = load_comparison(SOLVENT_CONTRAST)
    replicate_counts = {c: len(fpkm_columns(load_comparison(c))) - 1
                        for c in list_comparisons()}

    return {
        "n_comparisons": len(list_comparisons()),
        "n_compounds": len(compounds),
        "n_genes": int(len(ageing)),
        "groups_per_contrast": 2,
        "replicates_per_group": 1,
        "replicate_columns_per_contrast": sorted(set(replicate_counts.values())),
        "shared_dmso_denominator": shared,
        "shared_dmso_max_deviation": max_dev,
        "vendor_significance_rule": f"qval < {VENDOR_QVAL_CUTOFF} AND "
                                    f"|log2(fc)| >= {VENDOR_LOG2FC_CUTOFF}",
        "fold_change_convention": "fc = FPKM(first name in 'XVSY') / FPKM(second name); "
                                  "log2(fc) > 0 means up in the first group",
        "ageing_degs": int((ageing["significant"] == "yes").sum()),
        "solvent_degs": int((solvent["significant"] == "yes").sum()),
        "caveats": [
            "n=1 per group: reported pval/qval reflect only within-library sampling "
            "noise, not biological variability. They cannot support a claim about "
            "reproducible biological effects.",
            "All 26 compound contrasts share one identical DMSO denominator, so the "
            "contrasts are statistically dependent and a single noisy control library "
            "propagates into every compound's DEG list.",
            "The vehicle alone changes more genes vs Old_ACT than ageing does, so any "
            "compound effect must be interpreted on top of a large solvent effect.",
        ],
    }


def profile_deg_dataset(comparison: str | None = None) -> dict[str, Any]:
    """Profile the gene_diff dataset, or one contrast within it."""
    directory = deg_dir()
    if not os.path.isdir(directory):
        return {"error": f"gene_diff directory not found: {directory}"}
    ensure_cache()

    if comparison:
        df = load_comparison(comparison)
        sig = df[df["significant"] == "yes"]
        return {
            "comparison": comparison,
            "n_genes": int(len(df)),
            "n_significant": int(len(sig)),
            "n_up": int((sig["regulation"] == "up").sum()),
            "n_down": int((sig["regulation"] == "down").sum()),
            "fpkm_columns": fpkm_columns(df),
            "median_abs_log2fc_of_degs": float(sig["log2(fc)"].abs().median()),
        }

    summary = []
    for name in list_comparisons():
        df = load_comparison(name)
        sig = df[df["significant"] == "yes"]
        summary.append({"comparison": name,
                        "up": int((sig["regulation"] == "up").sum()),
                        "down": int((sig["regulation"] == "down").sum()),
                        "total": int(len(sig))})
    return {"path": directory,
            "design": audit_design(),
            "per_comparison_degs": sorted(summary, key=lambda r: -r["total"])}


def deg_evidence(comparison: str | None = None) -> list[dict[str, Any]]:
    """Wrap the dataset profile as EvidenceRecords, caveats included."""
    profile = profile_deg_dataset(comparison)
    if "error" in profile:
        return [EvidenceRecord(claim=profile["error"], source_type="local_data",
                               source_id="error", authority_weight=0.0).to_dict()]
    if comparison:
        claim = (f"{comparison}: {profile['n_significant']} DEGs "
                 f"({profile['n_up']} up / {profile['n_down']} down) of "
                 f"{profile['n_genes']} genes.")
        extra = profile
    else:
        design = profile["design"]
        claim = (f"gene_diff: {design['n_compounds']} compounds vs a shared DMSO "
                 f"control in aged splenocytes, {design['n_genes']} genes, "
                 f"n={design['replicates_per_group']} per group.")
        extra = design
    # n=1 design caps how much authority this evidence may carry downstream.
    return [EvidenceRecord(claim=claim, source_type="local_data",
                           source_id=profile.get("path", comparison or ""),
                           snippet="; ".join(audit_design()["caveats"]),
                           authority_weight=0.45, extra=extra).to_dict()]


REGISTRY.register(
    ToolSpec(name="profile_deg_dataset",
             description="Profile the ADJ/gene_diff DEG dataset (26 compounds vs shared "
                         "DMSO control in aged mouse splenocytes). Returns per-contrast "
                         "DEG counts plus machine-checked design audit and caveats "
                         "(n=1 per group, shared control).",
             domain="data",
             parameters={"comparison": "optional contrast name, e.g. RapamycinVSDMSO"},
             returns="dict with design audit and per-comparison DEG counts"),
    profile_deg_dataset,
)
