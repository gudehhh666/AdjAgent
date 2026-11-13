"""OTG transcriptomics loader/profiler.

Describes a dataset (shape, columns, dtypes, preview) so the data specialist can
write targeted analysis code. Handles very large CSVs via chunked reading — the
main OTG file is multi-GB, so we never load it whole for profiling.
"""
from __future__ import annotations

import os
from typing import Any

import pandas as pd

from core.config import get_setting, resolve_path
from core.schemas import EvidenceRecord
from environment.registry import REGISTRY, ToolSpec


def _resolve(path: str | None) -> str:
    if path:
        p = resolve_path(path) if not os.path.isabs(path) else path
        return str(p)
    # default to the first small dev dataset if configured
    small = get_setting("data_lake", "otg_dataset_small", default=[]) or []
    if small:
        return str(resolve_path(small[0]))
    return str(resolve_path(get_setting("data_lake", "otg_dataset", default="")))


def profile_dataset(path: str | None = None, preview_rows: int = 5) -> dict[str, Any]:
    fpath = _resolve(path)
    if not fpath or not os.path.exists(fpath):
        return {"error": f"dataset not found: {fpath}"}

    size_mb = round(os.path.getsize(fpath) / 1e6, 1)
    # header + small preview only (safe for multi-GB files)
    head = pd.read_csv(fpath, nrows=preview_rows)
    n_cols = head.shape[1]

    # count rows cheaply for large files by streaming lines
    n_rows = 0
    with open(fpath, "r", errors="ignore") as fh:
        for _ in fh:
            n_rows += 1
    n_rows = max(0, n_rows - 1)

    cols = list(head.columns)
    return {
        "path": fpath,
        "size_mb": size_mb,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns_sample": cols[:20],
        "preview": head.head(preview_rows).to_dict(orient="records"),
        "hint": "Use run_python to load in chunks (pd.read_csv(..., chunksize=...)) "
                "and import environment.tools.analysis.adjuvant_scoring for stats.",
    }


def dataset_evidence(path: str | None = None, note: str = "") -> list[dict[str, Any]]:
    """Wrap a dataset profile as an EvidenceRecord (local_data source)."""
    prof = profile_dataset(path)
    if "error" in prof:
        return [EvidenceRecord(claim=prof["error"], source_type="local_data",
                source_id="error", authority_weight=0.0).to_dict()]
    rec = EvidenceRecord(
        claim=f"OTG dataset {os.path.basename(prof['path'])}: "
              f"{prof['n_rows']} rows x {prof['n_cols']} cols. {note}",
        source_type="local_data",
        source_id=prof["path"],
        snippet=str(prof["columns_sample"]),
        authority_weight=0.7,
        extra={"size_mb": prof["size_mb"]},
    )
    return [rec.to_dict()]


REGISTRY.register(
    ToolSpec(name="profile_dataset",
             description="Profile an OTG transcriptomics CSV (shape, columns, preview); "
                         "chunk-safe for multi-GB files.",
             domain="data",
             parameters={"path": "csv path (optional; defaults to dev dataset)",
                         "preview_rows": "int"},
             returns="dict with shape, columns_sample, preview"),
    profile_dataset,
)
