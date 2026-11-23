"""Optional durable baseline -> retrieval -> evidence patch workflow.

Fallback delivers a previously successful answer while retaining failed run
statuses and unknown costs. It is not a successful evidence enhancement.
"""
import hashlib
import time

from .contracts import result_record
from ..runtime.journal import LeaseLost
from ..runtime.model_invocations import leased_run, inherit_guard
from ..storage import TERMINAL_STATUSES
from ..task_modes import qa_claim_patch, qa_question_first
from ..tools.catalog import canonical_json


WORKFLOW_VERSION = 'qa-claim-patch-v1'
DONE_PHASE = 'qa_claim_patch_done'
PATCH_SKILL_SHA256 = '8f78679a5afb2396a2e582d024a81584581ad218c45c4a7c8b7b5e42d8cde3a2'
# Kept in runtime code so TaskSpec implementation hashing binds the baseline.
BASELINE_SYSTEM = 'You are an expert in immunology and vaccine adjuvant science. Answer the scientific question directly and concisely in the same language as the question. Use your existing knowledge. Distinguish established facts from inference and uncertainty. Do not claim to have searched or accessed sources. Do not invent citations. Give the answer without describing private reasoning.'


class ArtifactIntegrityError(ValueError):
    """A corrupt saved answer is not eligible for a baseline fallback."""


def _guard(framework, spec, state, owned, execution, *, check_budget=True):
    def guard():
        owned()
        execution()
        if check_budget:
            framework._remaining(spec, state)
    return inherit_guard(guard, execution)


def _verify_artifact(framework, delivery):
    try:
        raw = framework.store.read_artifact(delivery['artifact_id'])
        valid = hashlib.sha256(raw).hexdigest() == delivery['sha256'] and raw.decode() == delivery['content']
    except (KeyError, ValueError, OSError, TypeError) as exc:
        raise ArtifactIntegrityError('Cached claim-patch answer artifact integrity failure') from exc
    if not valid:
        raise ArtifactIntegrityError('Cached claim-patch answer artifact integrity mismatch')


def cached(framework, item):
    result = item['result']
    if result.get('delivery'):
        _verify_artifact(framework, result['delivery'])
    baseline = item['state'].get('baseline')
    if baseline:
        _verify_artifact(framework, baseline)
    return result


def _finish(framework, item, spec, state, owned, execution_status, enhancement_status,
            *, answer=None, edits=None, sources=None, reason=None, execution=None, done_phase=DONE_PHASE,
            extra=None):
    """Local fallback may finalize after budget expiry without another request.

Never promote the failed retrieval/patch run to succeeded. Normal enhancement
success retains the existing fenced, budget-checked terminal commit.
"""
    owned()
    rid = state.get('stage_run') or state['retrieval_run']
    delivery = {}
    if answer is not None:
        artifact = framework.store.put_artifact(answer, 'answer.md', 'text/markdown',
            {'workflow':spec.fingerprint, 'enhancement_status':enhancement_status,
             'baseline_sha256':state.get('baseline', {}).get('sha256'), 'fallback_reason':reason},
            session_id=item['session_id'], run_id=rid)
        delivery = {'kind':'answer', 'content':answer, 'answer':answer,
                    'artifact_id':artifact['id'], 'sha256':artifact['sha256'],
                    'sources':sources or [], 'edits':edits or []}
    result = result_record(spec, execution_status, delivery, review_status='not_run',
        attempt_refs=state['attempts'], usage=framework._usage(state),
        enhancement_status=enhancement_status, fallback_reason=reason,
        baseline_sha256=state.get('baseline', {}).get('sha256'),
        open_gaps=[reason['message']] if reason else [], error=reason, **(extra or {}))
    state['phase'] = done_phase
    state['enhancement_status'] = enhancement_status
    state['fallback_reason'] = reason
    current = framework.store.get_run(rid)
    if current['status'] in TERMINAL_STATUSES:
        # A failed/unknown retrieval remains failed, even though its separately
        # completed baseline is available for local delivery.
        framework.repository.save(item['fingerprint'], state, guard=owned, result=result)
    else:
        def commit(run_execution):
            guard = _guard(framework, spec, state, owned, run_execution,
                           check_budget=execution_status == 'succeeded')
            terminal = execution_status if execution_status in TERMINAL_STATUSES else 'failed'
            framework.invocations.finish(rid, status=terminal,
                state={**current['state'], 'phase':'finished', 'delivery':delivery,
                       'enhancement_status':enhancement_status, 'error':reason}, guard=guard,
                on_commit=lambda conn:framework.repository.save_in_transaction(
                    conn, item['fingerprint'], state, result, guard=owned))
        if execution is not None:
            commit(execution)
        else:
            with leased_run(framework.store, rid) as run_execution:
                commit(run_execution)
    framework.store.append_event(item['session_id'], 'workflow.result', result, rid)
    return result


def ensure_baseline(framework, item, spec, state, owned, *, next_step):
    """Durable tool-free answer on the initial run; shared by the baseline-first QA workflows."""
    sid, rid = item['session_id'], state['retrieval_run']
    if 'baseline' not in state:
        if 'qa_baseline_started_at' not in state:
            state['qa_baseline_started_at'] = time.time()
            framework.repository.save(item['fingerprint'], state, guard=owned)
        with leased_run(framework.store, rid) as execution:
            guard = _guard(framework, spec, state, owned, execution)
            framework.invocations.checkpoint(rid, 'qa-independent-baseline', state, guard=guard)
            reply = framework.invocations.complete(sid, rid, framework.service.models.port(sid),
                [{'role':'system','content':BASELINE_SYSTEM},{'role':'user','content':spec.question}], [],
                role='baseline', key='qa-independent-baseline-v1', max_output_tokens=2400,
                timeout_seconds=180, guard=guard)
            if reply.refusal or reply.finish_reason not in {'stop','end_turn'} or reply.tool_calls or not isinstance(reply.content, str) or not reply.content.strip():
                raise ValueError('Independent baseline refused or incomplete')
            artifact = framework.store.put_artifact(reply.content, 'independent-baseline.md', 'text/markdown',
                {'workflow':spec.fingerprint,'kind':'independent_qa_baseline'}, session_id=sid, run_id=rid)
            # The engine uses elapsed_active_seconds, while the auxiliary
            # ledger has its own clock. Charge baseline elapsed time once
            # to their shared run before allowing the engine to continue.
            guard()
            with framework.store.transaction(write=True) as conn:
                framework.invocations._fence(conn, rid, guard)
                current = framework.store._get(conn, 'runs', rid)
                current_state = dict(current['state'])
                if not current_state.get('qa_baseline_time_charged'):
                    current_state['elapsed_active_seconds'] = current_state.get('elapsed_active_seconds', 0.) + max(0., time.time()-state['qa_baseline_started_at'])
                    current_state['qa_baseline_time_charged'] = True
                    conn.execute('UPDATE runs SET state=?,revision=revision+1 WHERE id=?',
                                 (canonical_json(current_state), rid))
            state['baseline'] = {'content':reply.content,'artifact_id':artifact['id'],'sha256':artifact['sha256']}
            state['qa_patch_step'] = next_step
            # Baseline is durable state, never a retrieval chat message.
            framework.repository.save(item['fingerprint'], state, guard=owned)
    _verify_artifact(framework, state['baseline'])


def run(framework, item, spec, state, owned):
    sid, rid = item['session_id'], state['retrieval_run']
    stage = 'baseline'
    try:
        ensure_baseline(framework, item, spec, state, owned, next_step='retrieval')
        stage = 'retrieval'
        if not state.get('qa_retrieval_complete'):
            framework._remaining(spec, state, admission=True)
            initial = framework.service.drive(rid)
            if initial['status'] not in TERMINAL_STATUSES:
                result = result_record(spec, initial['status'],
                    {'kind':'answer', **state['baseline'], 'sources':[]}, review_status='not_run',
                    enhancement_status='pending', fallback_reason=None,
                    attempt_refs=state['attempts'], usage=framework._usage(state))
                framework.repository.save(item['fingerprint'], state, guard=owned, result=result)
                return result
            if initial['status'] != 'succeeded':
                reason = {'stage':stage, **(initial['state'].get('error') or
                          {'code':initial['status'],'message':'Retrieval did not complete successfully'})}
                return _finish(framework, item, spec, state, owned, initial['status'], 'failed',
                    answer=state['baseline']['content'], reason=reason)
            state['qa_retrieval_complete'] = True
            state['qa_patch_step'] = 'patch'
            framework.repository.save(item['fingerprint'], state, guard=owned)
        stage = 'patch'
        packet = state.get('patch_packet')
        if packet is None:
            packet = qa_question_first.evidence_packet(spec.question, framework._documents(sid, rid))
            state['patch_packet'] = packet
            framework.repository.save(item['fingerprint'], state, guard=owned)
        if not packet:
            return _finish(framework, item, spec, state, owned, 'failed', 'not_attempted',
                answer=state['baseline']['content'], reason={'stage':stage,'code':'no_evidence',
                    'message':'No matching evidence was available; the independent baseline is unchanged.'})
        if hashlib.sha256(qa_claim_patch.PATCH_SKILL.read_bytes()).hexdigest() != PATCH_SKILL_SHA256:
            raise ValueError('Claim-patch skill changed; create a new versioned workflow')
        stage_rid = framework._stage_run(item, spec, state, owned)
        with leased_run(framework.store, stage_rid) as execution:
            guard = _guard(framework, spec, state, owned, execution)
            proposal = framework._call(item, state, guard,
                qa_claim_patch.patch_messages(spec.question, state['baseline']['content'], packet),
                'qa-claim-patch-v1', maximum=2400)
            applied = qa_claim_patch.apply_patch(state['baseline']['content'], proposal, packet)
            return _finish(framework, item, spec, state, owned, 'succeeded',
                'applied' if applied['edits'] else 'unchanged', answer=applied['answer'],
                edits=applied['edits'], sources=applied['used_sources'], execution=execution)
    except Exception as exc:
        # Losing ownership is not an enhancement failure that this worker may
        # publish. The next owner can recover an immutable completed receipt.
        if isinstance(exc, (LeaseLost, ArtifactIntegrityError)) or 'ownership lost' in str(exc).lower():
            raise
        status = 'budget_exhausted' if 'budget' in str(exc).lower() else 'failed'
        reason = {'stage':stage, 'code':getattr(exc, 'code', type(exc).__name__),
                  'message':framework.store.redact(str(exc))}
        baseline = state.get('baseline')
        return _finish(framework, item, spec, state, owned, status, 'failed',
            answer=baseline['content'] if baseline else None, reason=reason)
