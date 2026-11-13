"""Frozen pipeline catalog: the incumbent agent form behind each benchmark slot, as harness code.

Kinds: ``direct`` is one tool-free call; ``session`` is a recipe over ``AgentService``; ``controller`` is a
workflow registered in ``orchestration.registry`` and run by ``AgentFramework``; ``host`` is host-driven steps
around solver calls. ``digest(name)`` hashes a pipeline's texts and limits for the freeze record.
"""
from dataclasses import dataclass
import hashlib
from importlib import import_module
import json

KINDS = frozenset({'direct', 'session', 'controller', 'host'})


@dataclass(frozen=True)
class Pipeline:
    name: str
    form: str
    pillar: str
    slots: tuple
    kind: str
    entry: str
    summary: str
    setting: str = ''

    def __post_init__(self):
        if self.kind not in KINDS or not self.entry:
            raise ValueError(f'Invalid pipeline registration: {self.name}')

    def module(self):
        return import_module(self.entry) if self.kind != 'controller' else None


PIPELINES = {p.name: p for p in (
    Pipeline('direct', 'F0', 'baseline', ('AdjQA-Fact',), 'direct', f'{__name__}.direct',
             'Bare solver, one call; also the paired baseline arm of every slot.'),
    Pipeline('qa-grounded-revision-v2', 'F1b', 'P2', ('AdjQA',), 'controller', 'qa-grounded-revision-v2',
             'Tool-free baseline, host PubMed/PMC plus Scholar/web retrieval, REVISE or KEEP_DRAFT, one anchor '
             'self-check.'),
    Pipeline('evidence-bound-v3', 'F2', 'P1', ('AdjResource', 'AdjData'), 'session', f'{__name__}.evidence_bound',
             'One session over typed resource packs, the local literature library and sandboxed computation, '
             'with task-kind session instructions.'),
    Pipeline('research-report', 'F3', 'P2', ('AdjDeepResearch',), 'session', f'{__name__}.research_report',
             'Tool-free draft, evidence-first web research session, delivery fallbacks, then citation binding '
             'against the passages the run read.', 'report'),
    Pipeline('research-report-quality', 'F4', 'P2', ('AdjDeepResearch-quality',), 'session',
             f'{__name__}.research_report', 'The F3 run with the short-answer budget and no binding pass.', 'quality'),
    Pipeline('synth-execution', 'F5', 'P1', ('AdjSynth',), 'session', f'{__name__}.synth',
             'Adjuvant-profile session with sandboxed code execution, no network.'),
    Pipeline('guided-evidence-bound', 'F6', 'P3', ('AdjCase-PLGA',), 'session', f'{__name__}.guided',
             'Evidence-bound session whose case guidance sits in the session instructions; controlled runs only.'),
    Pipeline('mechanism-graph-v2-grounding', 'F7', 'P3', ('AdjCase-mechanism-retro',), 'host', f'{__name__}.mechanism',
             'Cutoff-bounded edge retrieval, citation gate, answer from gated edges only; the interactive form is '
             'the unified-mechanism-graph-v2 workflow.', 'retro'),
    Pipeline('mechanism-graph-v3-planning', 'F7', 'P3', ('AdjCase-mechanism-decision',), 'host',
             f'{__name__}.mechanism', 'Solver off-path predictions, host value-per-cost ranking and feedback '
             'propagation until one hypothesis is live; the interactive form is unified-mechanism-graph-v3.', 'decision'),
)}


def get(name):
    if name not in PIPELINES:
        raise KeyError(f'Unknown pipeline: {name}')
    return PIPELINES[name]


def for_slot(slot):
    return next((p for p in PIPELINES.values() if slot in p.slots), None)


def spec(name):
    p = get(name)
    if p.kind == 'controller':
        from ..orchestration.registry import WORKFLOWS
        w = WORKFLOWS[p.entry]
        return dict(workflow=w.version, status=w.status, runtime=w.runtime, session=w.session, stage_model=w.stage_model)
    module = p.module()
    if not p.setting:
        return module.SPEC
    return {k: v for k, v in module.SPEC.items() if k not in module.SETTINGS or k == p.setting}


def digest(name):
    text = json.dumps(spec(name), sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()
