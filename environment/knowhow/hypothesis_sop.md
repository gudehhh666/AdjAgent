# Know-How: Adjuvant Hypothesis SOP

Procedural memory card. The coordinator reads relevant cards during planning.

## Data-driven signal (OTG transcriptomics)
- Never treat dose x time x replicate as independent samples (pseudo-replication).
  Aggregate replicates first (see adjuvant_scoring pseudo-replication handling).
- Use Welch t-test for group comparison; apply BH-FDR multiple-comparison
  correction. Report `n_significant_genes` at FDR<=0.05.
- Prefer deterministic `adjuvant_scoring` functions over ad-hoc LLM-written stats.
- Cross-validate discriminative signatures (report CV-AUC; >=0.7 is meaningful).

## Immune module priors
- Use `immune_modules.json` (Type I IFN / NF-kB / NLRP3 / ER-stress / Toxicity).
- A candidate adjuvant should activate an immune module WITHOUT dominating the
  Toxicity_Death module.

## Hypothesis quality gates
- Every claim must cite an EvidenceRecord (grounding).
- Cross-source corroboration raises confidence; conflicts must be surfaced.
- A hypothesis is not "promoted" until the independent reviewer passes it.
- Always attach a falsification experiment (what negative result disproves it).

## Adjuvant mechanism classes (for framing hypotheses)
- PRR agonism (TLR/RIG-I/STING/NLR) -> innate activation -> adaptive help.
- Depot / sustained antigen release.
- Inflammasome (NLRP3) -> IL-1b/IL-18 -> Th responses.
- Cell-stress / DAMP release -> immunogenic signaling.
