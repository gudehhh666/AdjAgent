"""Grounded revision QA v2: hybrid PubMed + Scholar/web retrieval, revise-or-keep, anchor self-check."""
from ..task_modes import qa_grounded
from . import qa_grounded_runtime

WORKFLOW_VERSION = qa_grounded.WORKFLOW_V2
DONE_PHASE = qa_grounded_runtime.DONE_PHASE


def run(framework, item, spec, state, owned):
    return qa_grounded_runtime.run(framework, item, spec, state, owned, v2=True)
