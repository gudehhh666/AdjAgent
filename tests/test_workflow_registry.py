import pytest

from adj_partner.application import AgentService
from adj_partner.models.base import ScriptedPort
from adj_partner.orchestration import registry
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.controller import AgentFramework


def framework(tmp_path, **kwargs):
    return AgentFramework(AgentService(tmp_path / 'state', model=ScriptedPort([])), **kwargs)


def test_every_registered_runtime_exposes_run_and_done_phase():
    for workflow in registry.WORKFLOWS.values():
        if workflow.runtime:
            assert callable(workflow.delegated.run) and workflow.delegated.DONE_PHASE
            assert workflow.delegated.WORKFLOW_VERSION == workflow.version
    assert registry.done_phases() == {'qa_claim_patch_done', 'qa_grounded_done'}


def test_unknown_or_mismatched_workflow_is_rejected_before_any_session(tmp_path):
    f = framework(tmp_path)
    with pytest.raises(ValueError, match='Unregistered'):
        f.submit(TaskSpec('qa', 'Question?', workflow_version='v2', allow_network=False))
    with pytest.raises(ValueError, match='does not support'):
        f.submit(TaskSpec('design', 'Design?', workflow_version='qa-grounded-revision-v1', allow_network=False))
    assert f.service.sessions() == []


def test_frozen_workflow_needs_explicit_opt_in_but_existing_tasks_stay_readable(tmp_path):
    spec = TaskSpec('qa', 'Question?', model='offline-script', workflow_version='qa-question-first-v1', allow_network=False)
    with pytest.raises(ValueError, match='frozen'):
        framework(tmp_path).submit(spec)
    service = AgentService(tmp_path / 'shared', model=ScriptedPort([]))
    created = AgentFramework(service, allow_frozen=True).submit(spec)
    assert AgentFramework(service).submit(spec)['fingerprint'] == created['fingerprint']
