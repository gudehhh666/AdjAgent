# Frozen configuration

`harness_freeze.json` is the machine-readable record; this document explains it. The release is checked with `scripts/harness_freeze.py`:

```sh
.venv/bin/python scripts/harness_freeze.py verify   # lists files changed since the freeze; exit code 1 on drift
.venv/bin/python scripts/harness_freeze.py slots    # task slots and their pipelines
```

**What is pinned.** The SHA-256 of all 197 `.py` files under `adj_partner/` and `environment/` (tree digest `17b858d3c714`), and of the 24 non-Python files the harness reads at run time: the skills under `config/skills/`, the resource index and `config/*.yaml` (`host_files`). Seven resource-entry guides and the resource index describe the unreleased resource packs; they are listed in `host_files.held_back` and their hashes are kept for later verification.

**Relation to the reported experiments.** The experiments were run before the agent forms were registered as pipeline modules. The registered pipelines reproduce the frozen runs' instructions, tool lists, budgets and per-request limits exactly; the only other change since then is that the search-gateway URL is read from `ADJ_AIHUB_GATEWAY_URL` instead of a built-in default. No model-facing behavior changed, and nothing was rerun.

## 1. Shared settings

- Solver model `gpt-5.6-luna` through an OpenAI-compatible endpoint (`ADJ_LLM_BASE_URL`).
- Web search `serper_gateway`, then `serper`, then `mwmbl`; page reading through `jina_gateway`; PubMed with an NCBI key, 45 s timeout and one retry, falling back to Serper.
- Judge `gpt-4o-2024-11-20` throughout; models of the solver's family are never judges.
- Every comparison is paired on the same item: harness minus direct answer, item-level bootstrap 95% CI, with win/tie/loss counts and token usage.
- Harness file tools refuse evaluation sets and judge-only references (`adj_partner/tools/policy.py`).

## 2. Pipeline registry

`PIPELINES` in `adj_partner/pipelines/__init__.py` maps each form to an entry point. `pipelines.spec(name)` returns its full settings and `pipelines.digest(name)` their SHA-256 (first 12 characters below; full digests in `forms.*.pipelines`). Procedures and budgets are described in the [architecture document](docs/architecture/HARNESS_ARCHITECTURE.md#3-agent-forms).

| Form | Pipeline | Entry point | Digest |
|---|---|---|---|
| F0 | `direct` | `adj_partner/pipelines/direct.py` | `b50f50d0d0ed` |
| F1b | `qa-grounded-revision-v2` | registered workflow (`adj_partner/orchestration/registry.py`) | `e8acf2307457` |
| F2 | `evidence-bound-v3` | `adj_partner/pipelines/evidence_bound.py` | `b95fdd06eedc` |
| F3 | `research-report` | `adj_partner/pipelines/research_report.py`, `citation_binding.py` | `aeaf38a757ea` |
| F4 | `research-report-quality` | `adj_partner/pipelines/research_report.py` | `72f77774c8cf` |
| F5 | `synth-execution` | `adj_partner/pipelines/synth.py` | `103ee1dd2e9f` |
| F6 | `guided-evidence-bound` | `adj_partner/pipelines/guided.py` | `5be414be3aae` |
| F7 grounding | `mechanism-graph-v2-grounding` | `adj_partner/pipelines/mechanism.py` | `2510bbaf3e91` |
| F7 planning | `mechanism-graph-v3-planning` | `adj_partner/pipelines/mechanism.py` | `0ec9c8ad1739` |

## 3. Task slots

| Pillar | Task | Role | Incumbent | Pipeline |
|---|---|---|---|---|
| P1 | AdjResource | main | F2 | `evidence-bound-v3` |
| P1 | AdjData | main | F2 | `evidence-bound-v3` |
| P1 | AdjSynth | main | F5 | `synth-execution` |
| P1 | AdjNER | appendix | F2 (pilot before the freeze) | unregistered |
| P2 | AdjQA | main | F1b | `qa-grounded-revision-v2` |
| P2 | AdjDeepResearch | main | F3 with citation binding | `research-report` |
| P2 | AdjDeepResearch quality | appendix | F4 | `research-report-quality` |
| Motivation | AdjQA-Fact | introduction | F0 | `direct` |
| P3 | Mechanism retrospection | main | F7 grounding | `mechanism-graph-v2-grounding` |
| P3 | PLGA (controlled) | case | F6 | `guided-evidence-bound` |
| P3 | Mechanism decision | main | F7 planning | `mechanism-graph-v3-planning` |
| P3 | ADB rediscovery | pending | none yet | unregistered |

AdjNER must be rerun with `evidence-bound-v3` before it is cited beyond the appendix.

## 4. Replacement rules

1. Solver model, endpoint, judge, budget and incumbent stay fixed unless they are the variable under test.
2. An incumbent is replaced only for a real gain over the direct answer: a held-out paired-difference CI entirely above 0, or a strictly better exact or verifiability metric without loss of content quality.
3. Tune on the development split, confirm once on the held-out split, and report the held-out numbers.
4. Report paired differences against both the direct answer and the incumbent, with 95% CI, win/tie/loss and tokens.
5. After any change to `adj_partner/` or `environment/`, run `verify`, rerun both arms of the affected tasks on the new code, then record a new `snapshot`.
6. Guidance that reveals the target is allowed only in slots marked as controlled reproduction (currently PLGA) and must be disclosed.
7. On replacement, the previous form moves to `history` and the candidate becomes the new `incumbent`, with the date, reason and run directory recorded.
