"""Single registry of controller workflow versions and their routing policy."""
from dataclasses import dataclass
from importlib import import_module

from . import resource_handoff
from ..task_modes import design, graph_grounding, qa_grounded, qa_question_first

ALL_MODES = frozenset({'qa', 'design', 'discovery_design'})
DESIGN_MODES = frozenset({'design', 'discovery_design'})


@dataclass(frozen=True)
class Workflow:
    version: str
    task_types: frozenset
    status: str
    summary: str
    # 'standard' runs the shared retrieval/stage loop; 'question_first' narrows QA tools and
    # instructions; 'experimental' adds experiment guidance to design sessions.
    session: str = 'standard'
    runtime: str = ''
    resource_profile: str = ''
    stage_model: str = 'critic'
    graph: bool = False
    literature_edges: bool = False
    # Predictive graphs carry per-hypothesis predictions and cost tiers; the host seeds candidates from the pack.
    predictive_graph: bool = False

    def __post_init__(self):
        if self.status not in {'active', 'frozen'} or self.session not in {'standard', 'question_first', 'experimental'}:
            raise ValueError(f'Invalid workflow registration: {self.version}')
        if self.stage_model not in {'critic', 'solver'}:
            raise ValueError(f'Invalid stage model policy: {self.version}')

    @property
    def delegated(self):
        return import_module(f'{__package__}.{self.runtime}') if self.runtime else None


WORKFLOWS = {w.version: w for w in (
    Workflow('unified-v1', ALL_MODES, 'frozen', 'Original retrieval, critic review and design revision loop.'),
    Workflow('unified-experiment-v1', ALL_MODES, 'active',
             'Default TaskSpec version; design sessions add experiment guidance.', session='experimental'),
    Workflow(design.GRAPH_WORKFLOW, DESIGN_MODES, 'active',
             'Mechanism graph with competing hypotheses and host-ranked checkpoints.', session='experimental', graph=True),
    Workflow(graph_grounding.WORKFLOW_VERSION, DESIGN_MODES, 'active',
             'Mechanism graph whose path edges get host-retrieved, citation-gated literature status before the report.',
             session='experimental', graph=True, literature_edges=True),
    Workflow(design.PREDICTIVE_WORKFLOW, DESIGN_MODES, 'active',
             'Literature-annotated mechanism graph whose checkpoints are ranked by expected hypotheses eliminated per cost, '
             'from host-derived on-path and model-stated off-path predictions; seeded from pack mechanism families.',
             session='experimental', graph=True, literature_edges=True, predictive_graph=True),
    Workflow(resource_handoff.WORKFLOW, DESIGN_MODES, 'active', 'Resource-aware design with pinned resource manifest.',
             session='experimental', resource_profile=resource_handoff.PROFILE),
    Workflow(resource_handoff.EVIDENCE_WORKFLOW, DESIGN_MODES, 'active',
             'Evidence-first design over authorized local roots.', session='experimental',
             resource_profile=resource_handoff.EVIDENCE_PROFILE),
    Workflow(qa_question_first.WORKFLOW_VERSION, frozenset({'qa'}), 'frozen',
             'Question-first answer over a host evidence packet; superseded by claim-patch and grounded revision.',
             session='question_first', stage_model='solver'),
    Workflow('qa-claim-patch-v1', frozenset({'qa'}), 'active', 'Baseline answer with host-checked claim patches.',
             session='question_first', runtime='qa_baseline_patch_runtime', stage_model='solver'),
    Workflow(qa_grounded.WORKFLOW_VERSION, frozenset({'qa'}), 'frozen',
             'Baseline answer revised against host-retrieved PubMed/PMC passages; superseded by v2.',
             runtime='qa_grounded_runtime', stage_model='solver'),
    Workflow(qa_grounded.WORKFLOW_V2, frozenset({'qa'}), 'active',
             'Baseline answer revised against PubMed/PMC plus Scholar/web passages, then one anchor self-check.',
             runtime='qa_grounded_v2_runtime', stage_model='solver'),
)}


def resolve(spec, *, allow_frozen=True):
    workflow = WORKFLOWS.get(spec.workflow_version)
    if workflow is None:
        raise ValueError(f'Unregistered workflow version: {spec.workflow_version}')
    if spec.task_type not in workflow.task_types:
        raise ValueError(f'Workflow {workflow.version} does not support task type {spec.task_type}')
    if workflow.status == 'frozen' and not allow_frozen:
        raise ValueError(f'Workflow {workflow.version} is frozen; existing tasks stay readable, new tasks need allow_frozen')
    return workflow


def done_phases():
    return {w.delegated.DONE_PHASE for w in WORKFLOWS.values() if w.runtime}
