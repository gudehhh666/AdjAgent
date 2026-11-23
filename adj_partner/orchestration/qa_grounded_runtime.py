"""Baseline -> model-written queries -> host retrieval -> revise-or-keep QA workflow.

All model requests run on the initial run. The host, not the model, searches, fetches and selects
passages, and registers each paper it passes to the model as a literature source. Any failure, an
unchanged answer and a rejected abstaining revision all deliver the independent baseline; only
'applied' is an evidence enhancement.

v2 (``qa_grounded_v2_runtime``) retrieves from PubMed/PMC plus Scholar/web pages and, before any successful
delivery, runs one anchor self-check rewrite of the answer; a failed or abstaining rewrite delivers the answer
it was given.
"""
from ..native.grounding import GroundingRetriever
from ..runtime.journal import LeaseLost
from ..runtime.model_invocations import leased_run
from ..task_modes import qa_grounded
from ..tools.catalog import canonical_json
from .qa_baseline_patch_runtime import ArtifactIntegrityError, _finish, _guard, ensure_baseline

WORKFLOW_VERSION = qa_grounded.WORKFLOW_VERSION
DONE_PHASE = 'qa_grounded_done'


def retriever(framework):
    """Tests inject ``framework.grounding``; otherwise NCBI credentials come from the host environment."""
    if getattr(framework, 'grounding', None) is None:
        from ..native.scholarly import ScholarlyGateway
        pubmed = ScholarlyGateway.from_environment(framework.service.project_root, framework.service.models.env_file).pubmed
        framework.grounding = GroundingRetriever(api_key=pubmed.api_key, email=pubmed.email)
    return framework.grounding


def hybrid_retriever(framework):
    """Tests inject ``framework.literature``; otherwise PubMed plus the Serper/Jina gateways from the environment."""
    if getattr(framework, 'literature', None) is None:
        from ..native.literature_search import HybridRetriever
        framework.literature = HybridRetriever.from_environment(
            framework.service.project_root, framework.service.models.env_file)
    return framework.literature


def _complete(framework, item, spec, state, owned, messages, key, maximum):
    sid, rid = item['session_id'], state['retrieval_run']
    framework._remaining(spec, state, admission=True)
    with leased_run(framework.store, rid) as execution:
        guard = _guard(framework, spec, state, owned, execution)
        framework.invocations.checkpoint(rid, key, state, guard=guard)
        reply = framework.invocations.complete(sid, rid, framework.service.models.port(sid), messages, [],
            role='synthesis', key=key, max_output_tokens=maximum, timeout_seconds=180, guard=guard)
    if reply.refusal or reply.finish_reason not in {'stop', 'end_turn'} or reply.tool_calls or not isinstance(reply.content, str):
        raise ValueError(f'{key} refused or incomplete')
    return reply.content


def _register(framework, item, spec, state, found):
    """One source per paper that contributed a passage; passages carry the source id and URL."""
    sid, rid = item['session_id'], state['retrieval_run']
    framework.store.put_artifact(canonical_json(found), 'grounding-retrieval.json', 'application/json',
        {'workflow': spec.fingerprint, 'kind': 'qa_grounding_retrieval'}, session_id=sid, run_id=rid)
    papers = {p.get('pmid') or p['url']: p for p in found['papers']}
    sources = {}
    for passage in found['passages']:
        key = passage.get('pmid') or passage['url']
        if key not in sources:
            paper = papers[key]
            pmid = paper.get('pmid')
            material = [x for x in found['passages'] if (x.get('pmid') or x['url']) == key]
            name = f'grounding-{pmid}.json' if pmid else f'grounding-web-{len(sources) + 1}.json'
            artifact = framework.store.put_artifact(canonical_json({'paper': paper, 'passages': material}),
                name, 'application/json',
                {'workflow': spec.fingerprint, 'kind': 'qa_grounding_passages'}, session_id=sid, run_id=rid)
            url = f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/' if pmid else paper['url']
            provenance = {'provider': 'host_grounding', 'url': url, 'pmid': pmid, 'pmcid': paper.get('pmcid'),
                          'text_availability': paper['text_availability'], 'query': paper['query'],
                          'limitations': ['Selected by keyword search and BM25; not checked to be the study in question.']}
            if not pmid:
                provenance['engine'] = paper.get('engine')
            source = framework.service.research.register_source(sid, rid, artifact_id=artifact['id'], kind='literature',
                title=paper['title'][:2000] or (f'PubMed {pmid}' if pmid else url),
                idempotency_key=f'qa-grounded:{pmid or url}', provenance=provenance)
            sources[key] = {'source_id': source['id'], 'url': url}
        passage.update(sources[key])
    return found['passages']


def run(framework, item, spec, state, owned, *, v2=False):
    question = spec.question
    stage = 'baseline'

    def finish(status, enhancement, answer, reason=None, **kwargs):
        if v2 and status == 'succeeded':
            answer, kwargs['extra'] = reflect(answer)
            if kwargs['extra']['reflection_status'] == 'applied' and kwargs.get('sources'):
                kept = qa_grounded.cited(answer, state.get('grounding_passages') or [])
                kwargs['sources'] = list(dict.fromkeys(p['url'] for p in kept))
                kwargs['edits'] = [{'source_id': p['source_id'], 'section': p['section']} for p in kept]
        return _finish(framework, item, spec, state, owned, status, enhancement,
                       answer=answer, reason=reason, done_phase=DONE_PHASE, **kwargs)

    def reflect(answer):
        if 'reflection' not in state:
            try:
                content = _complete(framework, item, spec, state, owned,
                                    qa_grounded.reflect_messages(question, answer), 'qa-grounded-reflect-v2', 2400)
                status, revised = qa_grounded.decide_reflection(content, answer)
            except Exception as exc:
                if isinstance(exc, (LeaseLost, ArtifactIntegrityError)) or 'ownership lost' in str(exc).lower():
                    raise
                status, revised = 'failed', answer
            state['reflection'] = {'status': status, 'answer': revised}
            framework.repository.save(item['fingerprint'], state, guard=owned)
        return state['reflection']['answer'], {'reflection_status': state['reflection']['status']}

    try:
        ensure_baseline(framework, item, spec, state, owned, next_step='queries')
        draft = state['baseline']['content']
        if not spec.allow_network:
            return finish('succeeded', 'not_attempted', draft, {'stage': 'queries', 'code': 'network_not_allowed',
                          'message': 'Host retrieval needs network access; the independent baseline is unchanged.'})
        stage = 'queries'
        if 'grounding_queries' not in state:
            specific, queries = qa_grounded.parse_queries(_complete(framework, item, spec, state, owned,
                qa_grounded.query_messages(question), 'qa-grounded-queries-v1', 1600))
            state['grounding_queries'] = {'specific_study': specific, 'queries': queries}
            framework.repository.save(item['fingerprint'], state, guard=owned)
        if not state['grounding_queries']['specific_study']:
            return finish('succeeded', 'not_attempted', draft, {'stage': stage, 'code': 'general_question',
                          'message': 'The question was classified as general knowledge; the independent baseline is unchanged.'})
        stage = 'retrieval'
        if 'grounding_passages' not in state:
            owned()
            queries = state['grounding_queries']['queries']
            found = (hybrid_retriever(framework).gather(question, queries, draft) if v2
                     else retriever(framework).gather(question, queries))
            state['grounding_passages'] = _register(framework, item, spec, state, found)
            framework.repository.save(item['fingerprint'], state, guard=owned)
        passages = state['grounding_passages']
        if not passages:
            return finish('succeeded', 'not_attempted', draft, {'stage': stage, 'code': 'no_evidence',
                          'message': 'Host retrieval returned no passages; the independent baseline is unchanged.'})
        stage = 'revise'
        content = _complete(framework, item, spec, state, owned,
                            qa_grounded.revise_messages(question, draft, passages), 'qa-grounded-revise-v1', 2400)
        status, answer = qa_grounded.decide(content, draft)
        used = qa_grounded.cited(answer, passages) if status == 'applied' else []
        reason = None if status != 'rejected_abstention' else {'stage': stage, 'code': 'revision_abstained',
            'message': 'The revision opened by abstaining while the baseline answered; the baseline is delivered.'}
        return finish('succeeded', status, answer, reason,
                      sources=list(dict.fromkeys(p['url'] for p in used)),
                      edits=[{'source_id': p['source_id'], 'section': p['section']} for p in used])
    except Exception as exc:
        if isinstance(exc, (LeaseLost, ArtifactIntegrityError)) or 'ownership lost' in str(exc).lower():
            raise
        status = 'budget_exhausted' if 'budget' in str(exc).lower() else 'failed'
        reason = {'stage': stage, 'code': getattr(exc, 'code', type(exc).__name__),
                  'message': framework.store.redact(str(exc))}
        baseline = state.get('baseline')
        return finish(status, 'failed', baseline['content'] if baseline else None, reason)
