"""Literature status of mechanism-graph edges; synthetic fixtures and mocked PubMed only."""
import json

import httpx
import pytest

from adj_partner.contracts import ModelReply
from adj_partner.native.grounding import GroundingRetriever
from adj_partner.research.design import validate_design
from adj_partner.research.mechanism_graph import analyze, validate_literature_edges
from adj_partner.task_modes import graph_grounding
from test_mechanism_graph import betaine, cards

JATS = '''<?xml version="1.0"?><article><front><article-meta><title-group><article-title>BHMT in splenocytes
</article-title></title-group></article-meta></front><body><sec><title>Results</title><p>Betaine exposure increased BHMT
catalytic activity in aged splenocytes, measured as methyl transfer to homocysteine with matched vehicle controls.</p>
</sec></body></article>'''


def transport():
    def handler(request):
        path = request.url.path
        if path.endswith('esearch.fcgi'):
            return httpx.Response(200, json={'esearchresult': {'idlist': ['111']}})
        if path.endswith('esummary.fcgi'):
            return httpx.Response(200, json={'result': {'111': {'title': 'BHMT in splenocytes', 'pubdate': '2022',
                                  'articleids': [{'idtype': 'pmc', 'value': 'PMC1'}]}}})
        if path.endswith('PMC1/fullTextXML'):
            return httpx.Response(200, content=JATS.encode())
        return httpx.Response(404)
    return httpx.MockTransport(handler)

PASSAGE = {'pmid': '111', 'pmcid': 'PMC1', 'title': 'BHMT activity', 'year': '2022', 'section': 'Results',
           'text': 'BHMT catalytic activity rose after betaine exposure.', 'score': 3.0}


def annotation(source, target, status='not_addressed', pmids=()):
    return dict(source=source, target=target, status=status, pmids=list(pmids), passages_retrieved=len(pmids),
                note='', gate='')


def test_targets_cover_each_hypothesis_path_edge_once():
    edges = graph_grounding.targets(betaine())
    assert [(e['edge'], e['source'], e['target']) for e in edges] == [
        ('E1', 'N0', 'N1'), ('E2', 'N1', 'N2'), ('E3', 'N2', 'N3'), ('E4', 'N3', 'N5'), ('E5', 'N0', 'N4'), ('E6', 'N4', 'N5')]
    assert edges[0]['claim'].startswith('Betaine exposure -> BHMT catalytic activity')


def test_gate_keeps_only_judgments_citing_the_same_edges_passages():
    edges = graph_grounding.targets(betaine())[:3]
    evidence = {'E1': [PASSAGE], 'E2': [dict(PASSAGE, pmid='222')], 'E3': []}
    reply = json.dumps({'edges': [
        {'edge': 'E1', 'status': 'supports', 'passages': ['E1.P1'], 'note': 'Same cell type.'},
        {'edge': 'E2', 'status': 'opposes', 'passages': ['E1.P1']},
        {'edge': 'E3', 'status': 'mixed', 'passages': []},
        {'edge': 'E9', 'status': 'supports', 'passages': ['E9.P1']}]})
    items = graph_grounding.gate(reply, edges, evidence)
    assert [(x['status'], x['pmids'], x['gate']) for x in items] == [
        ('supports', ['111'], ''), ('not_addressed', [], 'no_valid_citation'), ('not_addressed', [], 'no_valid_citation')]
    assert [x['status'] for x in graph_grounding.gate('not json', edges, evidence)] == ['not_addressed'] * 3
    assert graph_grounding.parse_queries('{"edges": [{"edge": "E1", "queries": ["a \\"b\\"", "c", "d"]}]}', edges) == {
        'E1': ['a  b', 'c'], 'E2': [], 'E3': []}


def test_max_queries_is_backward_compatible_and_configurable():
    edges = graph_grounding.targets(betaine())[:2]
    reply = '{"edges": [{"edge": "E1", "queries": ["q1", "q2", "q3"]}]}'
    assert graph_grounding.parse_queries(reply, edges)['E1'] == ['q1', 'q2']
    assert graph_grounding.parse_queries(reply, edges, max_queries=3)['E1'] == ['q1', 'q2', 'q3']
    # The default prompt is unchanged; a wider cap adds an explicit instruction for the solver.
    assert graph_grounding.query_messages('q', edges) == graph_grounding.query_messages('q', edges, max_queries=2)
    assert 'up to 3 queries' in graph_grounding.query_messages('q', edges, max_queries=3)[1]['content']


def test_literature_breaks_ties_but_never_changes_hypothesis_status():
    plain = analyze(betaine(), cards(), {})
    assert [r['checkpoint_id'] for r in plain['mechanism_order']] == ['C1', 'C4', 'C2']
    literature = [annotation('N0', 'N1', 'supports', ['111']), annotation('N0', 'N4', 'opposes', ['222'])]
    plan = analyze(betaine(), cards(), {}, literature)
    assert [r['checkpoint_id'] for r in plan['mechanism_order']] == ['C4', 'C1', 'C2']
    assert plan['mechanism_order'][0]['incoming_literature'] == {'N0': 'opposes'}
    assert [h['status'] for h in plan['hypotheses']] == [h['status'] for h in plain['hypotheses']]
    assert plan['literature_edges'] == literature


def test_literature_annotations_are_validated_against_the_graph():
    graph = betaine()
    assert validate_literature_edges([annotation('N0', 'N1')], graph)
    for bad, message in [([annotation('N0', 'N5')], 'declared edges'),
                         ([annotation('N0', 'N1', 'supports')], 'only cited'),
                         ([annotation('N0', 'N1', pmids=['111'])], 'only cited'),
                         ([annotation('N0', 'N1', 'proven', ['111'])], 'Unknown')]:
        with pytest.raises(ValueError, match=message):
            validate_literature_edges(bad, graph)
    from test_research_design import design
    d = design('claim_x'); d['verification'].update(checkpoints=cards(), literature_edges=[annotation('N0', 'N1')])
    with pytest.raises(ValueError, match='require a mechanism_graph'):
        validate_design(d, ['claim_x'])


def v2(tmp_path, *, network=True, judge=None, authored=False):
    from test_framework_design import seeded
    s, f, t, p = seeded(tmp_path, workflow_version=graph_grounding.WORKFLOW_VERSION, allow_network=network)
    f.grounding = GroundingRetriever(transport=transport())
    original, calls = p.complete, []

    def complete(messages, tools, **kwargs):
        system = messages[0].get('content', '')
        if system == graph_grounding.QUERY_SYSTEM:
            calls.append('queries')
            return ModelReply(content=json.dumps({'edges': [{'edge': 'E1', 'queries': ['betaine BHMT activity']}]}),
                              usage={'input_tokens': 9, 'output_tokens': 9})
        if system == graph_grounding.JUDGE_SYSTEM:
            calls.append('judge')
            assert '[E1.P1]' in messages[1]['content']
            return ModelReply(content=json.dumps(judge or {'edges': [
                {'edge': 'E1', 'status': 'supports', 'passages': ['E1.P1'], 'note': 'Synthetic.'}]}),
                usage={'input_tokens': 9, 'output_tokens': 9})
        result = original(messages, tools, **kwargs)
        if not tools and 'schema' in json.loads(messages[-1]['content']):
            value = json.loads(result.content); v = value['research_design']['verification']
            v.update(checkpoints=cards('P1'), mechanism_graph=betaine(('P1',) * 4))
            if authored:
                v['literature_edges'] = [annotation('N0', 'N1')]
            return ModelReply(content=json.dumps(value), usage=result.usage)
        return result
    p.complete = complete
    return s, f, t, calls


def test_v2_annotates_graph_edges_and_reports_cited_literature(tmp_path):
    s, f, t, calls = v2(tmp_path)
    try:
        result = f.run(t['fingerprint'])
        assert result['execution_status'] == 'succeeded', result.get('error')
        assert calls == ['queries', 'judge']
        revisions = [r for r in result['lineage']['records'] if r['kind'] == 'design_revision']
        edges = revisions[-1]['payload']['design']['verification']['literature_edges']
        assert edges[0] == dict(source='N0', target='N1', status='supports', pmids=['111'], passages_retrieved=edges[0]['passages_retrieved'],
                                note='Synthetic.', gate='')
        assert all(e['status'] == 'not_addressed' for e in edges[1:])
        assert 'literature_edges' not in revisions[0]['payload']['design']['verification']
        text = result['delivery']['readable_content']
        assert '机制边的文献状态' in text and 'PMID 111' in text
        sources = s.research.list_sources(t['session_id'], next(iter(f.repository.get(t['fingerprint'])['state']['attempts'])))
        assert any(src['provenance'].get('provider') == 'host_grounding' for src in sources)
    finally:
        s.close()


def test_v2_without_network_skips_literature_and_still_reports(tmp_path):
    s, f, t, calls = v2(tmp_path, network=False)
    try:
        result = f.run(t['fingerprint'])
        assert result['execution_status'] == 'succeeded', result.get('error')
        assert calls == [] and '机制边的文献状态' not in result['delivery']['readable_content']
        assert f.repository.get(t['fingerprint'])['state']['literature']['status'] == 'network_not_allowed'
    finally:
        s.close()


def test_model_cannot_author_literature_status(tmp_path):
    s, f, t, calls = v2(tmp_path, authored=True)
    try:
        result = f.run(t['fingerprint'])
        assert result['execution_status'] == 'failed' and 'host-written' in result['error']['message']
    finally:
        s.close()
