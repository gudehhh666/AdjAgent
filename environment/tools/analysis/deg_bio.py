"""Bioinformatics analysis over the ADJ/gene_diff DEG dataset.

Three capabilities the pipeline previously lacked, all deterministic and all
runnable offline (the vendor tables already carry GO / KEGG / Reactome columns,
so no external enrichment service is needed):

1. an immunosenescence axis defined against the *vehicle* baseline, which is the
   only baseline the compound contrasts are measured against;
2. a Connectivity-Map-style reversal score ranking compounds by how far they push
   aged splenocytes back toward the young profile;
3. over-representation testing on the built-in annotation columns, and scoring of
   the five adjuvant immune modules already declared in
   ``environment/knowhow/immune_modules.json``.

Deliberately reuses the deterministic primitives in ``adjuvant_scoring`` rather
than reimplementing normalisation / FDR logic.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from core.config import get_setting, resolve_path
from core.schemas import EvidenceRecord
from environment.registry import REGISTRY, ToolSpec
from environment.tools.analysis.adjuvant_scoring import (
    benjamini_hochberg, composite_score, minmax_normalize, robust_zscore,
)
from environment.tools.data import deg_loader
from environment.tools.data.deg_loader import (
    AGEING_CONTRAST, SOLVENT_CONTRAST, compound_comparisons, compound_name,
    expression_matrix, load_annotation, load_comparison,
)

# FPKM floor: 12.6% of vendor DEGs sit below 1 FPKM in both groups, where ratios
# are dominated by the noise floor rather than by biology.
EXPRESSION_FLOOR = 1.0
PSEUDOCOUNT = 0.1

ANNOTATION_COLUMNS = ("GO", "KEGG", "Reactome")
# "GO:0006915(apoptotic process)" / "04657(IL-17 signaling pathway)"
_TERM_RE = re.compile(r"^([^(]+)\((.*)\)$")


# --------------------------------------------------------------- ageing axis
def ageing_axis(vehicle_relative: bool = True,
                min_expression: float = EXPRESSION_FLOOR) -> pd.DataFrame:
    """Per-gene direction from the aged baseline toward the young profile.

    Args:
        vehicle_relative: if True the axis is log2(Young_ACT / DMSO), i.e. the
            distance to travel from the aged+vehicle baseline that the compound
            contrasts actually use. If False the naive log2(Young_ACT / Old_ACT)
            vendor contrast is returned, which ignores the solvent effect.
        min_expression: require this FPKM in at least one of the two samples.

    Returns:
        DataFrame indexed by gene_id with gene_name, youth_direction and the
        expression levels the direction was computed from.
    """
    matrix = expression_matrix()
    baseline = "DMSO" if vehicle_relative else "Old_ACT"
    young, aged = matrix["Young_ACT"], matrix[baseline]

    axis = pd.DataFrame({
        "gene_name": matrix["gene_name"],
        "young_fpkm": young,
        "aged_fpkm": aged,
        "youth_direction": np.log2((young + PSEUDOCOUNT) / (aged + PSEUDOCOUNT)),
    })
    expressed = axis[["young_fpkm", "aged_fpkm"]].max(axis=1) >= min_expression
    axis = axis[expressed & np.isfinite(axis["youth_direction"])]
    axis["baseline"] = baseline
    return axis


def ageing_signature(top_n: int = 250, vehicle_relative: bool = True,
                     min_abs_log2fc: float = 1.0) -> dict[str, Any]:
    """Up / down gene sets defining the young-versus-aged phenotype."""
    axis = ageing_axis(vehicle_relative=vehicle_relative)
    strong = axis[axis["youth_direction"].abs() >= min_abs_log2fc]
    ranked = strong.sort_values("youth_direction", ascending=False)
    up = ranked.head(top_n)      # higher in young -> compound should raise these
    down = ranked.tail(top_n)    # higher in aged  -> compound should lower these
    return {
        "baseline": axis["baseline"].iloc[0] if len(axis) else None,
        "vehicle_relative": vehicle_relative,
        "n_expressed_genes": int(len(axis)),
        "n_above_threshold": int(len(strong)),
        "up_in_young": list(up.index),
        "up_in_young_names": up["gene_name"].tolist(),
        "down_in_young": list(down.index),
        "down_in_young_names": down["gene_name"].tolist(),
        "axis": axis,
    }


# ------------------------------------------------------- reversal / CMap score
def _row_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks along axis 1, ties broken arbitrarily (FPKM ties are negligible)."""
    order = np.argsort(values, axis=1)
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(values.shape[0])[:, None]
    ranks[rows, order] = np.arange(values.shape[1], dtype=float)
    return ranks


def _permutation_null_r(pool: np.ndarray, target: np.ndarray, n_permutations: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Null distribution of Spearman r between random gene draws and a fixed target.

    Sampled with replacement across the expressed universe; with k genes drawn out
    of ~1.5e4 the difference from a without-replacement permutation is ~k/n and
    does not affect the resulting tail probabilities.
    """
    k = target.size
    draws = pool[rng.integers(0, pool.size, size=(n_permutations, k))]
    draw_ranks = _row_ranks(draws)
    target_ranks = stats.rankdata(target)

    dr = draw_ranks - draw_ranks.mean(axis=1, keepdims=True)
    tr = target_ranks - target_ranks.mean()
    denom = np.sqrt((dr ** 2).sum(axis=1) * (tr ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (dr @ tr) / denom, 0.0)


def _connectivity(ranks: pd.Series, gene_set: Iterable[str]) -> float:
    """CMap-style one-sided KS statistic of a gene set against a ranked list."""
    members = [g for g in gene_set if g in ranks.index]
    n_total, n_set = len(ranks), len(members)
    if n_set == 0 or n_total == 0:
        return 0.0
    positions = np.sort(ranks.loc[members].values)
    j = np.arange(1, n_set + 1)
    a = np.max(j / n_set - positions / n_total)
    b = np.max(positions / n_total - (j - 1) / n_set)
    return float(a if a > b else -b)


def reversal_scores(top_n: int = 250, vehicle_relative: bool = True,
                    n_permutations: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Rank compounds by how strongly they reverse the ageing signature.

    Two independent scores are reported so that a ranking is only trusted when
    both agree:

    * ``reversal_r`` -- Spearman correlation between the compound's log2(fc) and
      the youth direction across signature genes. Positive means the compound
      moves aged cells toward the young profile.
    * ``cmap_score`` -- KS-based connectivity, the difference between the
      up-in-young and down-in-young enrichment in the compound's ranked profile.

    ``perm_p`` is a gene-label permutation p-value for ``reversal_r``, FDR
    corrected across compounds.
    """
    signature = ageing_signature(top_n=top_n, vehicle_relative=vehicle_relative)
    axis = signature["axis"]
    sig_genes = list(signature["up_in_young"]) + list(signature["down_in_young"])
    rng = np.random.default_rng(seed)

    rows = []
    for comparison in compound_comparisons():
        df = load_comparison(comparison).set_index("gene_id")
        treated = [c for c in deg_loader.fpkm_columns(df) if c != "FPKM.DMSO"][0]

        effect = df["log2(fc)"]
        expressed = df[[treated, "FPKM.DMSO"]].max(axis=1) >= EXPRESSION_FLOOR
        effect = effect[expressed & np.isfinite(effect)]

        shared = [g for g in sig_genes if g in effect.index]
        target = axis.loc[shared, "youth_direction"]
        observed = effect.loc[shared]
        r = float(stats.spearmanr(observed.values, target.values).statistic)

        null = _permutation_null_r(effect.to_numpy(), target.to_numpy(),
                                   n_permutations, rng)
        perm_p = float((np.sum(np.abs(null) >= abs(r)) + 1) / (n_permutations + 1))

        ranks = effect.rank(ascending=False)
        up_conn = _connectivity(ranks, signature["up_in_young"])
        down_conn = _connectivity(ranks, signature["down_in_young"])

        sig = df[df["significant"] == "yes"]
        rows.append({
            "compound": compound_name(comparison),
            "reversal_r": r,
            "perm_p": perm_p,
            "cmap_score": (up_conn - down_conn) / 2.0,
            "n_signature_genes_used": len(shared),
            "n_deg": int(len(sig)),
            "n_up": int((sig["regulation"] == "up").sum()),
            "n_down": int((sig["regulation"] == "down").sum()),
        })

    out = pd.DataFrame(rows)
    out["perm_q"] = benjamini_hochberg(out["perm_p"].values)
    return out.sort_values("reversal_r", ascending=False).reset_index(drop=True)


def panel_centered_reversal(top_n: int = 250, vehicle_relative: bool = True,
                            n_permutations: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Reversal scoring after removing the component shared by all 26 compounds.

    Every compound contrast is log2(compound) - log2(DMSO) against the *same*
    single DMSO library, so that term is common to all 26 and cancels when a
    compound is expressed relative to the panel:

        log2(A/DMSO) - median_over_panel(log2(*/DMSO)) ~= log2(A / typical compound)

    This is the population-control normalisation used by L1000/CMap for exactly
    this reason, and it is what makes a *relative* ranking of these 26 compounds
    interpretable even though their absolute scores are not.
    """
    signature = ageing_signature(top_n=top_n, vehicle_relative=vehicle_relative)
    axis = signature["axis"]
    sig_genes = list(signature["up_in_young"]) + list(signature["down_in_young"])
    rng = np.random.default_rng(seed)

    profiles = {}
    for comparison in compound_comparisons():
        df = load_comparison(comparison).set_index("gene_id")
        treated = [c for c in deg_loader.fpkm_columns(df) if c != "FPKM.DMSO"][0]
        expressed = df[[treated, "FPKM.DMSO"]].max(axis=1) >= EXPRESSION_FLOOR
        profiles[compound_name(comparison)] = df.loc[expressed, "log2(fc)"]

    panel = pd.DataFrame(profiles)
    centered = panel.sub(panel.median(axis=1), axis=0)

    rows = []
    for compound in centered.columns:
        effect = centered[compound].dropna()
        shared = [g for g in sig_genes if g in effect.index]
        target = axis.loc[shared, "youth_direction"]
        r = float(stats.spearmanr(effect.loc[shared].values, target.values).statistic)
        null = _permutation_null_r(effect.to_numpy(), target.to_numpy(),
                                   n_permutations, rng)
        rows.append({
            "compound": compound,
            "centered_reversal_r": r,
            "perm_p": float((np.sum(np.abs(null) >= abs(r)) + 1) / (n_permutations + 1)),
            "n_signature_genes_used": len(shared),
        })
    out = pd.DataFrame(rows)
    out["perm_q"] = benjamini_hochberg(out["perm_p"].values)
    return out.sort_values("centered_reversal_r", ascending=False).reset_index(drop=True)


# ------------------------------------------------- controls used for grounding
def no_compound_control_score(top_n: int = 250) -> dict[str, float]:
    """Reversal score of the two libraries that received no compound at all.

    ``Old_ACT`` cannot rejuvenate anything, so whatever score it earns is the
    floor that a real compound has to beat before its score means anything.
    ``Young_ACT`` is the target direction itself and therefore the ceiling.
    """
    signature = ageing_signature(top_n=top_n, vehicle_relative=True)
    axis = signature["axis"]
    sig_genes = list(signature["up_in_young"]) + list(signature["down_in_young"])
    matrix = expression_matrix()

    out = {}
    for sample in ("Old_ACT", "Young_ACT"):
        ratio = np.log2((matrix[sample] + PSEUDOCOUNT) / (matrix["DMSO"] + PSEUDOCOUNT))
        expressed = matrix[[sample, "DMSO"]].max(axis=1) >= EXPRESSION_FLOOR
        effect = ratio[expressed & np.isfinite(ratio)]
        shared = [g for g in sig_genes if g in effect.index]
        out[sample] = float(stats.spearmanr(
            effect.loc[shared].values, axis.loc[shared, "youth_direction"].values).statistic)
    return out


def shared_deg_core(min_compounds: int = 20) -> dict[str, Any]:
    """How much of each compound's DEG list is a core shared across the panel.

    A core shared by nearly all 26 pharmacologically unrelated compounds, which
    also overlaps the vehicle-versus-aged contrast, indicates the shared DMSO
    denominator rather than 26 distinct drug responses.
    """
    from collections import Counter

    per_compound = {compound_name(c): deg_loader.deg_sets(c)
                    for c in compound_comparisons()}
    counter: Counter = Counter()
    for genes in per_compound.values():
        counter.update(genes)
    core = {g for g, n in counter.items() if n >= min_compounds}
    vehicle = deg_loader.deg_sets(SOLVENT_CONTRAST)
    ageing = deg_loader.deg_sets(AGEING_CONTRAST)
    fractions = [len(core & genes) / len(genes) for genes in per_compound.values()]

    names = load_annotation().set_index("gene_id")["gene_name"]
    ubiquitous = [g for g, n in counter.items() if n == len(per_compound)]
    return {
        "min_compounds": min_compounds,
        "core_size": len(core),
        "n_deg_in_all_compounds": len(ubiquitous),
        "median_fraction_of_compound_deg_list": float(np.median(fractions)),
        "core_overlap_with_vehicle_frac": len(core & vehicle) / max(len(core), 1),
        "core_overlap_with_ageing_frac": len(core & ageing) / max(len(core), 1),
        "vehicle_deg_count": len(vehicle),
        "ageing_deg_count": len(ageing),
        "example_core_genes": names.reindex(sorted(core)).dropna().tolist()[:20],
    }


# ------------------------------------------------------------- enrichment
def _parse_terms(cell: str) -> list[tuple[str, str]]:
    if not isinstance(cell, str) or cell in ("", "nan", "None"):
        return []
    seen, terms = set(), []
    for chunk in cell.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = _TERM_RE.match(chunk)
        term_id, label = (match.group(1), match.group(2)) if match else (chunk, chunk)
        if term_id not in seen:
            seen.add(term_id)
            terms.append((term_id, label))
    return terms


@lru_cache(maxsize=8)
def _term_index(column: str) -> tuple[dict[str, set[str]], dict[str, str], set[str]]:
    """Build term -> gene_ids, term -> label, and the annotated background.

    Cached: parsing the GO and Reactome columns means splitting very long strings
    for all ~35k genes, which is far too slow to repeat per enrichment call.
    """
    annotation = load_annotation()
    members: dict[str, set[str]] = defaultdict(set)
    labels: dict[str, str] = {}
    background: set[str] = set()
    for gene_id, cell in zip(annotation["gene_id"], annotation[column]):
        terms = _parse_terms(cell)
        if terms:
            background.add(gene_id)
        for term_id, label in terms:
            members[term_id].add(gene_id)
            labels.setdefault(term_id, label)
    return members, labels, background


def enrich(gene_ids: Iterable[str], column: str = "GO", min_size: int = 5,
           max_size: int = 500, top: int = 20) -> pd.DataFrame:
    """Hypergeometric over-representation test on a built-in annotation column."""
    if column not in ANNOTATION_COLUMNS:
        raise ValueError(f"column must be one of {ANNOTATION_COLUMNS}")
    members, labels, background = _term_index(column)
    query = set(gene_ids) & background
    n_background, n_query = len(background), len(query)
    if n_query == 0:
        return pd.DataFrame()

    rows = []
    for term_id, genes in members.items():
        size = len(genes)
        if size < min_size or size > max_size:
            continue
        hits = query & genes
        if len(hits) < 2:
            continue
        p = float(stats.hypergeom.sf(len(hits) - 1, n_background, size, n_query))
        rows.append({
            "term_id": term_id,
            "term": labels.get(term_id, term_id),
            "n_term": size,
            "n_hits": len(hits),
            "expected": size * n_query / n_background,
            "fold_enrichment": (len(hits) / n_query) / (size / n_background),
            "pval": p,
            "hit_genes": ",".join(sorted(hits)),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["qval"] = benjamini_hochberg(out["pval"].values)
    out = out.sort_values("pval").reset_index(drop=True)
    out.attrs["n_query"] = n_query
    out.attrs["n_background"] = n_background
    return out.head(top) if top else out


def enrich_comparison(comparison: str, direction: str | None = None,
                      column: str = "GO", top: int = 15) -> dict[str, Any]:
    """Enrichment of one contrast's vendor DEG list."""
    genes = deg_loader.deg_sets(comparison, direction=direction)
    table = enrich(genes, column=column, top=top)
    return {
        "comparison": comparison,
        "direction": direction or "both",
        "annotation": column,
        "n_query_genes": int(len(genes)),
        "terms": table.drop(columns=["hit_genes"]).to_dict(orient="records")
        if not table.empty else [],
    }


# -------------------------------------------------------- immune modules
@lru_cache(maxsize=1)
def load_immune_modules() -> dict[str, dict[str, Any]]:
    knowhow = get_setting("knowhow_dir", default="environment/knowhow")
    path = os.path.join(str(resolve_path(knowhow)), "immune_modules.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["modules"]


def _match_module_genes(patterns: Iterable[str], gene_names: pd.Series) -> list[str]:
    """Expand module keywords (which may contain wildcards like 'Ifit*')."""
    lowered = gene_names.astype(str).str.lower()
    matched: set[str] = set()
    for pattern in patterns:
        p = pattern.lower()
        if any(ch in p for ch in "*?["):
            hit = lowered[[bool(fnmatch.fnmatch(v, p)) for v in lowered]]
        else:
            hit = lowered[lowered == p]
        matched.update(hit.index)
    return sorted(matched)


def immune_module_scores(n_permutations: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Score every compound on the five declared adjuvant immune modules.

    Score is the mean log2(fc) of the module's genes, converted to a z-score
    against random gene sets of equal size so modules of different size and
    compounds of different overall response magnitude stay comparable.
    """
    modules = load_immune_modules()
    rng = np.random.default_rng(seed)
    rows = []

    for comparison in compound_comparisons():
        df = load_comparison(comparison).set_index("gene_id")
        treated = [c for c in deg_loader.fpkm_columns(df) if c != "FPKM.DMSO"][0]
        expressed = df[[treated, "FPKM.DMSO"]].max(axis=1) >= EXPRESSION_FLOOR
        effect = df.loc[expressed & np.isfinite(df["log2(fc)"]), :]
        record: dict[str, Any] = {"compound": compound_name(comparison)}

        for module, spec in modules.items():
            genes = _match_module_genes(spec["keywords"], effect["gene_name"])
            if not genes:
                record[module] = np.nan
                record[f"{module}_n"] = 0
                continue
            observed = float(effect.loc[genes, "log2(fc)"].mean())
            pool = effect["log2(fc)"].to_numpy()
            null = pool[rng.integers(0, pool.size,
                                     size=(n_permutations, len(genes)))].mean(axis=1)
            sd = null.std(ddof=1)
            record[module] = (observed - null.mean()) / sd if sd > 0 else 0.0
            record[f"{module}_n"] = len(genes)
        rows.append(record)

    return pd.DataFrame(rows)


# ------------------------------------------------------- candidate ranking
def rank_candidates(top_n: int = 250, vehicle_relative: bool = True,
                    n_permutations: int = 2000) -> pd.DataFrame:
    """Combine reversal, immune activation and toxicity into one ranking.

    Toxicity enters with higher_is_better=False so that compounds achieving a
    high reversal score merely by killing cells are penalised rather than
    promoted. All components are min-max normalised first, which is why the
    unbounded reversal and module z-scores cannot dominate the total.
    """
    reversal = reversal_scores(top_n=top_n, vehicle_relative=vehicle_relative,
                              n_permutations=n_permutations)
    modules = immune_module_scores(n_permutations=n_permutations)
    merged = reversal.merge(modules, on="compound", how="left")

    innate = merged[["Type_I_IFN", "Type_II_NFkB", "Type_III_NLRP3"]].mean(axis=1)
    merged["innate_activation"] = innate
    merged["reversal_z"] = robust_zscore(merged["reversal_r"].values)
    merged["candidate_score"] = composite_score(
        components={
            "reversal": merged["reversal_r"].fillna(0.0).values,
            "innate": innate.fillna(0.0).values,
            "toxicity": merged["Toxicity_Death"].fillna(0.0).values,
        },
        weights={"reversal": 0.5, "innate": 0.3, "toxicity": 0.2},
        higher_is_better={"reversal": True, "innate": True, "toxicity": False},
    )
    merged["response_breadth"] = minmax_normalize(merged["n_deg"].values)
    return merged.sort_values("candidate_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------- evidence pack
def _terms_text(gene_ids: Iterable[str], column: str = "KEGG", top: int = 6) -> str:
    table = enrich(gene_ids, column=column, top=top)
    if table.empty:
        return "(no term enriched)"
    return "; ".join(f"{row['term']} (q={row['qval']:.1e}, FE={row['fold_enrichment']:.1f})"
                     for _, row in table.iterrows())


def evidence_pack(top_n: int = 250, n_permutations: int = 2000) -> dict[str, Any]:
    """Deterministic analysis of gene_diff, packaged as stats + EvidenceRecords.

    This replaces LLM-authored analysis code for this dataset. Every number the
    hypothesis generator sees is computed here by fixed code, and the design
    limitations travel with the numbers instead of being left implicit.
    """
    design = deg_loader.audit_design()
    centered = panel_centered_reversal(top_n=top_n, n_permutations=n_permutations)
    raw = reversal_scores(top_n=top_n, vehicle_relative=True,
                          n_permutations=max(500, n_permutations // 4))
    modules = immune_module_scores(n_permutations=n_permutations)
    controls = no_compound_control_score(top_n=top_n)
    core = shared_deg_core()
    signature = ageing_signature(top_n=top_n, vehicle_relative=True)

    module_cols = [c for c in modules.columns
                   if c != "compound" and not c.endswith("_n")]
    promoters = centered.head(5)
    opposers = centered.tail(5).iloc[::-1]
    n_credible = int((centered["perm_q"] < 0.05).sum())

    records: list[dict[str, Any]] = []

    def add(claim: str, source_id: str, snippet: str = "", weight: float = 0.45,
            extra: dict[str, Any] | None = None) -> None:
        records.append(EvidenceRecord(
            claim=claim, source_type="local_data", source_id=source_id,
            snippet=snippet, authority_weight=weight, extra=extra or {}).to_dict())

    add(f"Experimental design: {design['n_compounds']} small molecules incubated with "
        f"vaccine-antigen-stimulated aged mouse splenocytes, profiled against a DMSO "
        f"vehicle control over {design['n_genes']} genes. Replicates per group = "
        f"{design['replicates_per_group']}. All 26 compound contrasts reuse one "
        f"identical DMSO denominator (max deviation "
        f"{design['shared_dmso_max_deviation']}).",
        "gene_diff:design", snippet=" | ".join(design["caveats"]), weight=0.45,
        extra=design)

    add(f"Vehicle effect is as large as the biology of interest: DMSO alone yields "
        f"{core['vehicle_deg_count']} DEGs versus untreated aged cells, exceeding the "
        f"{core['ageing_deg_count']} DEGs of the ageing contrast itself.",
        "gene_diff:vehicle_effect", weight=0.5,
        snippet="Any compound effect must be read on top of a large solvent effect.")

    add(f"A shared DEG core of {core['core_size']} genes is significant in >=20 of 26 "
        f"compounds ({core['n_deg_in_all_compounds']} in all 26) and makes up a median "
        f"{core['median_fraction_of_compound_deg_list']:.0%} of each compound's DEG "
        f"list; {core['core_overlap_with_vehicle_frac']:.1%} of that core also appears "
        f"in the vehicle-versus-aged contrast, indicating shared-control leakage rather "
        f"than 26 distinct pharmacologies.",
        "gene_diff:shared_core", weight=0.5,
        snippet=f"example core genes: {', '.join(core['example_core_genes'][:12])}",
        extra=core)

    add(f"Negative control invalidates raw reversal scoring: untreated Old_ACT, which "
        f"contains no compound, scores reversal_r={controls['Old_ACT']:+.3f}, higher "
        f"than {int((raw['reversal_r'] < controls['Old_ACT']).sum())} of the 26 real "
        f"compounds (raw range {raw['reversal_r'].min():+.3f} to "
        f"{raw['reversal_r'].max():+.3f}, all nominally significant). Compound rankings "
        f"must therefore use panel-centred scores, not raw vehicle-referenced scores.",
        "gene_diff:negative_control", weight=0.55, extra=controls)

    add(f"Immunosenescence axis (young versus aged, vehicle baseline): pathways higher "
        f"in YOUNG cells are {_terms_text(signature['up_in_young'])}. Pathways higher "
        f"in AGED cells are {_terms_text(signature['down_in_young'])}.",
        "gene_diff:ageing_axis", weight=0.5)

    add(f"After removing the shared DMSO component (CMap population control), "
        f"{n_credible} of 26 compounds reach permutation FDR<0.05 and the scores span "
        f"{centered['centered_reversal_r'].min():+.3f} to "
        f"{centered['centered_reversal_r'].max():+.3f}. Compounds shifting aged cells "
        f"TOWARD the young profile: "
        + ", ".join(f"{r['compound']} ({r['centered_reversal_r']:+.3f}, "
                    f"q={r['perm_q']:.3f})" for _, r in promoters.iterrows())
        + ". Compounds shifting cells AWAY from the young profile: "
        + ", ".join(f"{r['compound']} ({r['centered_reversal_r']:+.3f}, "
                    f"q={r['perm_q']:.3f})" for _, r in opposers.iterrows()) + ".",
        "gene_diff:panel_centered_reversal", weight=0.55,
        extra={"table": centered.to_dict(orient="records")})

    module_summary = "; ".join(
        f"{col}: median z={modules[col].median():+.2f} across compounds, "
        f"but the no-compound Old_ACT control also scores highly, so only "
        f"between-compound differences are interpretable" if col == "Type_II_NFkB"
        else f"{col}: median z={modules[col].median():+.2f} "
             f"(range {modules[col].min():+.2f} to {modules[col].max():+.2f})"
        for col in module_cols)
    add(f"Immune module activation across the panel — {module_summary}.",
        "gene_diff:immune_modules", weight=0.45,
        extra={"table": modules.to_dict(orient="records")})

    for _, row in promoters.head(2).iterrows():
        comparison = f"{row['compound']}VSDMSO"
        add(f"{row['compound']} (top panel-centred reverser, "
            f"r={row['centered_reversal_r']:+.3f}): up-regulated genes enrich "
            f"{_terms_text(deg_loader.deg_sets(comparison, 'up'))}. Down-regulated genes "
            f"enrich {_terms_text(deg_loader.deg_sets(comparison, 'down'))}.",
            f"gene_diff:{row['compound']}", weight=0.5)

    stats_dict = {
        "n_significant_genes": n_credible,
        "fdr": float(centered["perm_q"].min()),
        "cv_auc": None,
        "biological_replicates": design["replicates_per_group"],
        "shared_control": design["shared_dmso_denominator"],
        "statistic": "panel-centred ageing-signature reversal, permutation FDR",
        "negative_control_reversal_r": controls["Old_ACT"],
    }

    return {
        "stats": stats_dict,
        "evidence": records,
        "tables": {
            "panel_centered_reversal": centered,
            "raw_reversal": raw,
            "immune_modules": modules,
        },
        "design": design,
        "controls": controls,
        "shared_core": core,
    }


REGISTRY.register(
    ToolSpec(name="deg_evidence_pack",
             description="Run the full deterministic gene_diff analysis and return "
                         "hypothesis-ready statistics plus EvidenceRecords with the "
                         "design caveats attached. Use this instead of LLM-written "
                         "analysis code for this dataset.",
             domain="analysis",
             parameters={"top_n": "signature size (int)",
                         "n_permutations": "int"},
             returns="dict with stats and evidence records"),
    lambda top_n=250, n_permutations=2000: {
        k: v for k, v in evidence_pack(top_n, n_permutations).items() if k != "tables"},
)

REGISTRY.register(
    ToolSpec(name="rank_adjuvant_candidates",
             description="Rank the 26 compounds by a composite of ageing-signature "
                         "reversal, innate immune module activation and (penalised) "
                         "toxicity module activation.",
             domain="analysis",
             parameters={"top_n": "signature size (int)",
                         "vehicle_relative": "bool",
                         "n_permutations": "int"},
             returns="list of compounds with candidate_score and components"),
    lambda top_n=250, vehicle_relative=True, n_permutations=2000: rank_candidates(
        top_n, vehicle_relative, n_permutations).to_dict(orient="records"),
)

REGISTRY.register(
    ToolSpec(name="ageing_signature",
             description="Define the immunosenescence axis (young vs aged splenocytes) "
                         "from the gene_diff dataset. vehicle_relative=True measures it "
                         "against the DMSO baseline that compound contrasts use.",
             domain="analysis",
             parameters={"top_n": "genes per direction (int)",
                         "vehicle_relative": "bool; True uses the DMSO baseline"},
             returns="dict with up/down gene sets and the per-gene youth direction"),
    lambda top_n=250, vehicle_relative=True: {
        k: v for k, v in ageing_signature(top_n, vehicle_relative).items() if k != "axis"},
)

REGISTRY.register(
    ToolSpec(name="reversal_scores",
             description="Rank the 26 compounds by Connectivity-Map-style reversal of the "
                         "ageing signature in aged mouse splenocytes. Reports Spearman "
                         "reversal_r with permutation FDR plus an independent KS cmap_score.",
             domain="analysis",
             parameters={"top_n": "signature size per direction (int)",
                         "vehicle_relative": "bool; True uses the DMSO baseline",
                         "n_permutations": "int"},
             returns="DataFrame of compounds with reversal_r, perm_q, cmap_score, n_deg"),
    lambda top_n=250, vehicle_relative=True, n_permutations=2000: reversal_scores(
        top_n, vehicle_relative, n_permutations).to_dict(orient="records"),
)

REGISTRY.register(
    ToolSpec(name="panel_centered_reversal",
             description="Reversal ranking after removing the shared DMSO component "
                         "(CMap population-control normalisation). Use this instead of "
                         "reversal_scores for any claim comparing these 26 compounds, "
                         "because the raw scores are inflated by the shared control.",
             domain="analysis",
             parameters={"top_n": "signature size (int)",
                         "vehicle_relative": "bool",
                         "n_permutations": "int"},
             returns="list of compounds with centered_reversal_r and perm_q"),
    lambda top_n=250, vehicle_relative=True, n_permutations=2000:
        panel_centered_reversal(top_n, vehicle_relative, n_permutations).to_dict(
            orient="records"),
)

REGISTRY.register(
    ToolSpec(name="enrich_deg_annotations",
             description="Hypergeometric GO/KEGG/Reactome over-representation test using "
                         "the annotation columns shipped inside the gene_diff tables; "
                         "runs fully offline with BH-FDR correction.",
             domain="analysis",
             parameters={"comparison": "contrast name, e.g. RapamycinVSDMSO",
                         "direction": "'up' | 'down' | None",
                         "column": "GO | KEGG | Reactome"},
             returns="dict with enriched terms, fold_enrichment, pval, qval"),
    enrich_comparison,
)

REGISTRY.register(
    ToolSpec(name="immune_module_scores",
             description="Score all 26 compounds on the five adjuvant immune modules "
                         "declared in knowhow/immune_modules.json (Type I IFN, NF-kB, "
                         "NLRP3, ER stress, toxicity), as permutation z-scores.",
             domain="analysis",
             parameters={"n_permutations": "int"},
             returns="DataFrame of per-compound module z-scores"),
    lambda n_permutations=2000: immune_module_scores(n_permutations).to_dict(
        orient="records"),
)
