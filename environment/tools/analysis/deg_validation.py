"""In-silico validation of a hypothesis against the gene_diff dataset.

The reviewer can only judge a hypothesis on plausibility and rigour; it cannot
tell whether the hypothesis actually agrees with the measurements it claims to
rest on. This module closes that gap: an LLM extracts the hypothesis's
data-checkable predictions into a fixed schema, and each prediction is then
tested by deterministic code against the measured tables.

A prediction can come back three ways, and the distinction matters:

* ``supported`` / ``contradicted`` -- the dataset can decide it, and did;
* ``not_testable`` -- the claim is about something this experiment never measured
  (germinal centres, antibody titres, in-vivo protection). Counting these as
  failures would punish a hypothesis for proposing wet-lab work, so they are
  reported separately and excluded from the support rate.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

from core.utils import extract_json
from environment.registry import REGISTRY, ToolSpec
from environment.tools.analysis import deg_bio
from environment.tools.data import deg_loader

_EXTRACT_SYS = (
    "You extract the machine-checkable predictions from an adjuvant hypothesis so "
    "they can be tested against a transcriptomic dataset.\n"
    "The dataset measures, for each of 26 named compounds versus a DMSO control, "
    "the log2 fold change of every mouse gene in aged splenocytes, plus KEGG/GO/"
    "Reactome annotations and five immune module scores (Type_I_IFN, Type_II_NFkB, "
    "Type_III_NLRP3, Type_IV_Stress, Toxicity_Death).\n"
    "For each prediction the hypothesis makes, emit one object:\n"
    '{"claim": "<short restatement>", "type": "<one of: compound_reversal, '
    'pathway_enrichment, gene_direction, module_activation, not_measurable>", '
    '"compound": "<compound name or empty>", "target": "<pathway name, gene symbol, '
    'or module name; empty for compound_reversal>", "direction": "<up|down>"}\n'
    "Rules: use type 'not_measurable' for anything this in-vitro transcriptome "
    "cannot address (antibody titres, germinal centres, survival, particle size, "
    "dosing, in-vivo protection). Use 'gene_direction' only for a specific gene "
    "symbol. Use 'compound_reversal' when the claim is that a compound shifts aged "
    "cells toward the young profile (direction 'up') or away (direction 'down').\n"
    'Return STRICT JSON: {"predictions": [ ... ]}. Extract at most 12.'
)

_MODULES = ("Type_I_IFN", "Type_II_NFkB", "Type_III_NLRP3", "Type_IV_Stress",
            "Toxicity_Death")


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _resolve_compound(name: str) -> str | None:
    """Map a free-text compound name onto a dataset column."""
    if not name:
        return None
    known = {_normalise(deg_loader.compound_name(c)): deg_loader.compound_name(c)
             for c in deg_loader.compound_comparisons()}
    target = _normalise(name)
    if target in known:
        return known[target]
    for key, value in known.items():          # "zinc gluconate" -> "Zincgluconate"
        if key in target or target in key:
            return value
    return None


class _Cache:
    """Tables are reused across every prediction and hypothesis in a run."""

    def __init__(self) -> None:
        self._centered: pd.DataFrame | None = None
        self._modules: pd.DataFrame | None = None
        self._enrich: dict[tuple[str, str], pd.DataFrame] = {}

    def centered(self) -> pd.DataFrame:
        if self._centered is None:
            self._centered = deg_bio.panel_centered_reversal(n_permutations=1000)
        return self._centered

    def modules(self) -> pd.DataFrame:
        if self._modules is None:
            self._modules = deg_bio.immune_module_scores(n_permutations=1000)
        return self._modules

    def enrichment(self, compound: str, direction: str) -> pd.DataFrame:
        key = (compound, direction)
        if key not in self._enrich:
            genes = deg_loader.deg_sets(f"{compound}VSDMSO", direction=direction)
            frames = [deg_bio.enrich(genes, column=col, top=0)
                      for col in ("KEGG", "GO", "Reactome")]
            frames = [f for f in frames if not f.empty]
            self._enrich[key] = (pd.concat(frames, ignore_index=True) if frames
                                 else pd.DataFrame())
        return self._enrich[key]


_CACHE = _Cache()


def _test_compound_reversal(compound: str, direction: str) -> dict[str, Any]:
    table = _CACHE.centered()
    row = table[table["compound"] == compound]
    if row.empty:
        return {"status": "not_testable", "detail": f"{compound} not in panel"}
    r = float(row["centered_reversal_r"].iloc[0])
    q = float(row["perm_q"].iloc[0])
    expected_positive = direction == "up"
    agrees = (r > 0) if expected_positive else (r < 0)
    return {
        "status": "supported" if agrees else "contradicted",
        "detail": f"panel-centred reversal r={r:+.3f} (perm q={q:.3f}); "
                  f"predicted {'toward' if expected_positive else 'away from'} young",
        "measured": r, "q": q,
    }


def _find_term(table: pd.DataFrame, target: str) -> pd.Series | None:
    if table.empty:
        return None
    needle = _normalise(target)
    hits = table[[bool(needle in _normalise(t) or _normalise(t) in needle)
                  for t in table["term"]]]
    return None if hits.empty else hits.sort_values("qval").iloc[0]


def _test_pathway(compound: str, target: str, direction: str) -> dict[str, Any]:
    opposite = "down" if direction == "up" else "up"
    predicted = _find_term(_CACHE.enrichment(compound, direction), target)
    counter = _find_term(_CACHE.enrichment(compound, opposite), target)

    if predicted is not None and predicted["qval"] < 0.05:
        return {"status": "supported",
                "detail": f"'{predicted['term']}' enriched among {compound} "
                          f"{direction}-regulated genes: q={predicted['qval']:.2e}, "
                          f"fold-enrichment={predicted['fold_enrichment']:.1f}",
                "measured": float(predicted["fold_enrichment"]),
                "q": float(predicted["qval"])}

    # Enriched in the opposite direction instead: the data says the reverse.
    if counter is not None and counter["qval"] < 0.05:
        return {"status": "contradicted",
                "detail": f"'{counter['term']}' is instead enriched among {compound} "
                          f"{opposite}-regulated genes (q={counter['qval']:.2e}), "
                          f"the opposite of the predicted {direction}",
                "q": float(counter["qval"])}

    if predicted is not None:
        return {"status": "unsupported",
                "detail": f"'{predicted['term']}' present among {compound} "
                          f"{direction}-regulated genes but not significant "
                          f"(q={predicted['qval']:.2f})",
                "q": float(predicted["qval"])}

    return {"status": "unsupported",
            "detail": f"'{target}' not enriched in either direction for {compound}"}


def _test_gene(compound: str, gene: str, direction: str) -> dict[str, Any]:
    df = deg_loader.load_comparison(f"{compound}VSDMSO")
    match = df[df["gene_name"].astype(str).str.lower() == gene.strip().lower()]
    if match.empty:
        return {"status": "not_testable", "detail": f"gene {gene} not in dataset"}
    log2fc = float(match["log2(fc)"].iloc[0])
    qval = float(match["qval"].iloc[0])
    significant = str(match["significant"].iloc[0]) == "yes"
    agrees = (log2fc > 0) if direction == "up" else (log2fc < 0)

    if agrees and significant:
        status = "supported"
    elif not agrees and significant:
        status = "contradicted"
    else:
        # Direction may match, but without significance there is no evidence either way.
        status = "unsupported"
    return {
        "status": status,
        "detail": f"{gene} in {compound}: log2fc={log2fc:+.2f}, q={qval:.2e}, "
                  f"vendor-significant={significant}; predicted {direction}",
        "measured": log2fc, "q": qval,
    }


def _test_module(compound: str, module: str, direction: str) -> dict[str, Any]:
    resolved = next((m for m in _MODULES if _normalise(module) in _normalise(m)
                     or _normalise(m) in _normalise(module)), None)
    if resolved is None:
        return {"status": "not_testable", "detail": f"module '{module}' not defined"}
    table = _CACHE.modules()
    row = table[table["compound"] == compound]
    if row.empty:
        return {"status": "not_testable", "detail": f"{compound} not in panel"}
    z = float(row[resolved].iloc[0])
    # The no-compound control also scores positive on these modules, so a compound
    # only counts as an activator if it exceeds the panel median.
    median = float(table[resolved].median())
    agrees = (z > median) if direction == "up" else (z < median)
    return {
        "status": "supported" if agrees else "contradicted",
        "detail": f"{resolved} z={z:+.2f} vs panel median {median:+.2f} "
                  f"(predicted {direction}); absolute z is inflated by the shared "
                  f"control so only the comparison to the panel is used",
        "measured": z,
    }


def test_prediction(pred: dict[str, Any]) -> dict[str, Any]:
    kind = str(pred.get("type", "")).strip()
    direction = str(pred.get("direction", "up")).strip().lower()
    target = str(pred.get("target", "")).strip()
    result = {"claim": pred.get("claim", ""), "type": kind,
              "compound": pred.get("compound", ""), "target": target,
              "direction": direction}

    if kind == "not_measurable":
        result.update({"status": "not_testable",
                       "detail": "requires an experiment this dataset does not contain"})
        return result

    compound = _resolve_compound(str(pred.get("compound", "")))
    if kind != "not_measurable" and compound is None:
        result.update({"status": "not_testable",
                       "detail": f"compound '{pred.get('compound')}' not in the panel"})
        return result
    result["compound"] = compound

    try:
        if kind == "compound_reversal":
            result.update(_test_compound_reversal(compound, direction))
        elif kind == "pathway_enrichment":
            result.update(_test_pathway(compound, target, direction))
        elif kind == "gene_direction":
            result.update(_test_gene(compound, target, direction))
        elif kind == "module_activation":
            result.update(_test_module(compound, target, direction))
        else:
            result.update({"status": "not_testable",
                           "detail": f"unknown prediction type '{kind}'"})
    except Exception as exc:  # noqa: BLE001
        result.update({"status": "not_testable", "detail": f"test error: {exc}"})
    return result


def validate_hypothesis(hypothesis: str, llm: Any = None) -> dict[str, Any]:
    """Extract and test a hypothesis's predictions against the measured data."""
    if llm is None:
        return {"error": "validate_hypothesis needs an llm to extract predictions",
                "predictions": [], "support_rate": None}
    try:
        # Reasoning models spend this budget on thinking before emitting JSON, so it
        # must comfortably exceed the output size or the response comes back empty.
        raw = llm.ask(f"HYPOTHESIS:\n{hypothesis}", system=_EXTRACT_SYS,
                      max_completion_tokens=8000)
        predictions = extract_json(raw).get("predictions", [])
    except Exception as exc:  # noqa: BLE001
        return {"error": f"prediction extraction failed: {exc}",
                "predictions": [], "support_rate": None,
                "summary": f"VALIDATION FAILED: {exc}"}

    if not predictions:
        # Distinguish a failed extraction from a hypothesis with nothing checkable;
        # reporting the former as "nothing decidable" would let an unvalidated
        # hypothesis look cleanly validated.
        return {"error": "extraction returned no predictions",
                "n_predictions": 0, "predictions": [], "support_rate": None,
                "summary": "VALIDATION FAILED: could not extract any prediction from "
                           "this hypothesis, so it is UNVALIDATED (not 'clean')"}

    tested = [test_prediction(p) for p in predictions if isinstance(p, dict)]
    supported = [t for t in tested if t["status"] == "supported"]
    contradicted = [t for t in tested if t["status"] == "contradicted"]
    unsupported = [t for t in tested if t["status"] == "unsupported"]
    not_testable = [t for t in tested if t["status"] == "not_testable"]
    decidable = len(supported) + len(contradicted) + len(unsupported)

    return {
        "n_predictions": len(tested),
        "n_supported": len(supported),
        "n_contradicted": len(contradicted),
        "n_unsupported": len(unsupported),
        "n_not_testable": len(not_testable),
        "support_rate": (len(supported) / decidable) if decidable else None,
        "predictions": tested,
        "summary": (f"{len(supported)}/{decidable} data-checkable predictions supported, "
                    f"{len(contradicted)} contradicted, {len(unsupported)} without "
                    f"statistical support" if decidable
                    else "no prediction could be decided by this dataset")
                   + f"; {len(not_testable)} require new experiments",
    }


REGISTRY.register(
    ToolSpec(name="validate_hypothesis_against_deg",
             description="In-silico validation: extract a hypothesis's data-checkable "
                         "predictions and test each against the gene_diff measurements. "
                         "Separates contradicted claims from claims this experiment "
                         "cannot address.",
             domain="reliability",
             parameters={"hypothesis": "hypothesis text", "llm": "LLM instance"},
             returns="dict with per-prediction status and support_rate"),
    validate_hypothesis,
)
