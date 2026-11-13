"""Design stage between decision and report for unified-mechanism-graph-v2.

The model writes queries per path edge, the host retrieves and registers passages, the model judges
each edge, and the host gate keeps only judgments that cite that edge's own passages. The gated
annotations are written by the host into a child design revision. Any enhancement failure leaves the
recorded design unchanged and proceeds to the report; budget and lease errors still propagate.
"""
from copy import deepcopy

import httpx

from ..models.base import ProviderFailure
from ..task_modes import graph_grounding
from ..tools.catalog import canonical_json
from .qa_grounded_runtime import retriever


def _register(framework, item, spec, rid, edge, found):
    sid = item['session_id']
    papers = {p['pmid']: p for p in found['papers']}
    for passage in found['passages']:
        pmid = passage['pmid']
        artifact = framework.store.put_artifact(canonical_json({'edge': edge, 'paper': papers[pmid], 'passage': passage}),
            f'graph-edge-{pmid}.json', 'application/json',
            {'workflow': spec.fingerprint, 'kind': 'graph_edge_passage'}, session_id=sid, run_id=rid)
        source = framework.service.research.register_source(sid, rid, artifact_id=artifact['id'], kind='literature',
            title=papers[pmid]['title'][:2000] or f'PubMed {pmid}',
            idempotency_key=f"graph-edge:{edge['edge']}:{pmid}:{passage['section']}:{passage['score']}",
            provenance={'provider': 'host_grounding', 'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/', 'pmid': pmid,
                        'pmcid': papers[pmid].get('pmcid'), 'text_availability': papers[pmid]['text_availability'],
                        'query': papers[pmid]['query'],
                        'limitations': ['Keyword search and BM25 selection for one mechanism edge; not verified by a person.']})
        passage['source_id'] = source['id']
    return found['passages']


def ground(framework, item, spec, state, guard, owned):
    sid, rid, fp = item['session_id'], state['stage_run'], item['fingerprint']
    save = lambda: framework.repository.save(fp, state, guard=owned)
    edges = graph_grounding.targets(state['design']['verification']['mechanism_graph'])
    literature = state.setdefault('literature', {'edges': [e['edge'] for e in edges]})
    if not spec.allow_network:
        literature['status'] = 'network_not_allowed'
    else:
        try:
            if 'queries' not in literature:
                text = framework._call(item, state, guard, graph_grounding.query_messages(spec.question, edges),
                                       f"edge-queries-{state['round']}", maximum=2000, json_response=False)
                literature['queries'] = graph_grounding.parse_queries(text, edges)
                save()
            if 'evidence' not in literature:
                search, evidence, requests = retriever(framework), {}, 0
                for edge in edges:
                    queries = literature['queries'][edge['edge']]
                    guard()
                    found = search.gather(edge['claim'], queries, k_papers=2, k_abstracts=1, k_passages=3,
                                          max_chars=900) if queries else {'papers': [], 'passages': [], 'requests': []}
                    evidence[edge['edge']] = _register(framework, item, spec, rid, edge, found)
                    requests += len(found['requests'])
                literature.update(evidence=evidence, requests=requests)
                save()
            text = framework._call(item, state, guard,
                                   graph_grounding.judge_messages(spec.question, edges, literature['evidence']),
                                   f"edge-judge-{state['round']}", maximum=3000, json_response=False)
            annotated = deepcopy(state['design'])
            annotated['verification']['literature_edges'] = graph_grounding.gate(text, edges, literature['evidence'])
            revision = framework.service.research.create_design_revision(sid, rid, brief_id=state['brief_id'],
                design=annotated, parent_id=state['design_id'], feedback_ids=[])
            state.update(design=annotated, design_id=revision['id'])
            literature['status'] = 'annotated'
        except (ValueError, ProviderFailure, httpx.HTTPError, OSError) as exc:
            literature.update(status='not_annotated', error={'code': getattr(exc, 'code', type(exc).__name__),
                                                             'message': framework.store.redact(str(exc))[:300]})
    state['phase'] = 'report'
    save()
