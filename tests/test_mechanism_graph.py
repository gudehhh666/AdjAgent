"""Mechanism-graph structure, host ordering and propagation; synthetic Betaine fixture, not biology."""
from copy import deepcopy
import json
import pytest
from adj_partner.research.design import validate_design, readable_report
from adj_partner.research.mechanism_graph import analyze, validate_mechanism_graph
from test_experiment_checkpoints import checkpoint


def betaine(claims=('claim_x',)*4):
    bhmt, alt, fn, shared = claims
    node = lambda i, label, level, cid: dict(id=i, label=label, level=level, context='Aged splenocytes, matched antigen and vehicle', claim_ids=[cid])
    nodes = [node('N0', 'Betaine exposure', 'intervention', shared),
             node('N1', 'BHMT catalytic activity', 'molecular_activity', bhmt),
             node('N2', 'Betaine-derived methyl flux', 'metabolic_link', bhmt),
             node('N3', 'Phenotype requires BHMT', 'causal_dependence', bhmt),
             node('N4', 'Osmolyte uptake without BHMT', 'molecular_presence', alt),
             node('N5', 'Antigen-specific T-cell function', 'antigen_function', fn)]
    edge = lambda a, b: dict(source=a, target=b, relation='would enable', claim_ids=[])
    return dict(nodes=nodes,
                edges=[edge('N0', 'N1'), edge('N1', 'N2'), edge('N2', 'N3'), edge('N3', 'N5'), edge('N0', 'N4'), edge('N4', 'N5')],
                hypotheses=[dict(id='H1', label='BHMT methyl-donor route', node_ids=['N0', 'N1', 'N2', 'N3', 'N5'], claim_ids=[bhmt]),
                            dict(id='H2', label='Osmolyte route', node_ids=['N0', 'N4', 'N5'], claim_ids=[alt])],
                checkpoint_nodes=[dict(checkpoint_id='C1', node_id='N1'), dict(checkpoint_id='C2', node_id='N3'),
                                  dict(checkpoint_id='C3', node_id='N5'), dict(checkpoint_id='C4', node_id='N4')])


def cards(cid='claim_x'):
    out = [checkpoint(1), checkpoint(2), checkpoint(3, 'function'), checkpoint(4)]
    for card, level in zip(out, ['molecular_activity', 'causal_dependence', 'antigen_function', 'molecular_presence']):
        card['decision_level'] = level; card['claim_ids'] = [cid]
    out[1]['depends_on'] = ['C1']
    return out


def test_plan_orders_upstream_discriminating_test_first_and_keeps_function_parallel():
    plan = analyze(betaine(), cards(), {})
    assert [r['checkpoint_id'] for r in plan['mechanism_order']] == ['C1', 'C4', 'C2']
    first = plan['mechanism_order'][0]
    assert first['live_hypotheses_split'] == 1 and first['downstream_checkpoints'] == ['C2', 'C3']
    assert plan['mechanism_order'][2]['open_upstream_checkpoints'] == ['C1']
    assert plan['parallel_function_checkpoints'] == [{'checkpoint_id': 'C3', 'node_id': 'N5', 'status': 'unmeasured'}]
    assert {h['status'] for h in plan['hypotheses']} == {'open'}
    assert plan['hypotheses_without_mechanism_checkpoint'] == []
    graph, cs = betaine(), cards()
    graph['checkpoint_nodes'][3]['node_id'] = 'N2'; cs[3]['decision_level'] = 'metabolic_link'
    assert analyze(graph, cs, {})['hypotheses_without_mechanism_checkpoint'] == ['H2']


def test_opposed_bhmt_weakens_only_its_path_and_deprioritizes_downstream_tests():
    graph = betaine(('claim_bhmt', 'claim_alt', 'claim_fn', 'claim_base'))
    state = analyze(graph, cards(), {'C1': 'opposes'})
    h1, h2 = state['hypotheses']
    assert h1['status'] == 'weakened_under_tested_conditions' and h1['withdrawn_downstream_nodes'] == ['N2', 'N3']
    assert h2['status'] == 'open'
    assert [r['checkpoint_id'] for r in state['mechanism_order']] == ['C4']
    assert state['deprioritized_checkpoints'] == [{'checkpoint_id': 'C2', 'node_id': 'N3', 'reason': 'only_on_weakened_hypotheses'}]
    assert state['claims_to_revisit'] == ['claim_bhmt']
    assert state['parallel_function_checkpoints'][0]['status'] == 'unmeasured'


def test_function_result_never_weakens_mechanism_and_inconclusive_stays_ranked():
    state = analyze(betaine(), cards(), {'C3': 'opposes', 'C1': 'inconclusive'})
    assert {h['status'] for h in state['hypotheses']} == {'open'}
    assert state['mechanism_order'][0]['checkpoint_id'] == 'C1' and state['mechanism_order'][0]['status'] == 'inconclusive'
    state = analyze(betaine(), cards(), {'C1': 'supports', 'C2': 'supports'})
    assert state['hypotheses'][0]['status'] == 'open'
    assert state['node_status']['N2'] == 'untested'


@pytest.mark.parametrize('mutation,message', [
    ('cycle', 'acyclic'), ('downward', 'lower evidence level'), ('off_edge', 'declared edges'),
    ('single_hypothesis', 'competing'), ('unbound_claim', 'bound in verification'),
    ('dependency_not_upstream', 'upstream'), ('function_on_mechanism', 'antigen_function'),
    ('level_mismatch', 'decision level'), ('two_roots', 'exactly one intervention'), ('unmapped', 'exactly one mechanism node')])
def test_graph_contract_rejects_untraceable_structures(mutation, message):
    graph, cs = betaine(), cards()
    if mutation == 'cycle':
        graph['nodes'].append(dict(id='N6', label='Second transporter', level='molecular_presence', context='same', claim_ids=['claim_x']))
        graph['edges'] += [dict(source='N4', target='N6', relation='x', claim_ids=[]), dict(source='N6', target='N4', relation='x', claim_ids=[])]
    if mutation == 'downward': graph['edges'].append(dict(source='N3', target='N1', relation='feedback', claim_ids=[]))
    if mutation == 'off_edge': graph['hypotheses'][1]['node_ids'] = ['N0', 'N2', 'N5']
    if mutation == 'single_hypothesis': graph['hypotheses'].pop()
    if mutation == 'unbound_claim': graph['nodes'][1]['claim_ids'] = ['claim_outside']
    if mutation == 'dependency_not_upstream': cs[3]['depends_on'] = ['C1']
    if mutation == 'function_on_mechanism': graph['checkpoint_nodes'][2]['node_id'] = 'N2'
    if mutation == 'level_mismatch': cs[0]['decision_level'] = 'metabolic_link'
    if mutation == 'two_roots': graph['nodes'][4]['level'] = 'intervention'
    if mutation == 'unmapped': graph['checkpoint_nodes'].pop()
    with pytest.raises(ValueError, match=message):
        validate_mechanism_graph(graph, cs, ['claim_x'])


def test_design_accepts_graph_only_beside_checkpoints_and_renders_host_order():
    from test_research_design import design
    d = design('claim_x'); d['verification']['checkpoints'] = cards(); d['verification']['mechanism_graph'] = betaine()
    assert validate_design(d, ['claim_x']) == d
    bare = design('claim_x'); bare['verification']['mechanism_graph'] = betaine()
    with pytest.raises(ValueError): validate_design(bare, ['claim_x'])
    from adj_partner.research.design import render_design
    text = render_design(d)
    assert '机制假设图（宿主排序，计划）' in text and text.index('1. C1') < text.index('2. C4') < text.index('3. C2')
    assert '功能检测独立并行：C3' in text


def test_store_state_propagates_reported_result_through_graph(tmp_path):
    from test_experiment_feedback import arguments
    from test_harness_research_records import setup, source
    from test_research_design import design
    store, r, sid, rid, _ = setup(tmp_path)
    src, _ = source(store, r, sid, rid)
    ev = r.select_evidence(sid, rid, src['id'], {'passage': 'First finding.'})
    claim = r.observe_evidence(sid, rid, ev['id'])
    d = design(claim['id']); d['verification']['checkpoints'] = cards(claim['id'])
    d['verification']['mechanism_graph'] = betaine((claim['id'],) * 4)
    brief = r.create_design_brief(sid, rid, goal='Synthetic graph', requirements=['Mechanism and function'], evidence_ids=[ev['id']])
    revision = r.create_design_revision(sid, rid, brief_id=brief['id'], design=d)
    plan = r.experiment_state(sid, rid, design_id=revision['id'])['mechanism_graph']
    assert [x['checkpoint_id'] for x in plan['mechanism_order']] == ['C1', 'C4', 'C2']
    lab, _ = source(store, r, sid, rid, kind='userfile', key='lab', content=json.dumps({'value': 'synthetic'}).encode(),
                    provenance={'experiment_data': True, 'text_availability': 'userfile', 'synthetic_fixture': True})
    le = r.select_evidence(sid, rid, lab['id'], {'pointer': '/value'})
    r.record_experiment_result(sid, rid, **arguments(revision, le))
    state = r.experiment_state(sid, rid, design_id=revision['id'])
    assert state['checkpoints'][0]['status'] == 'opposes'
    assert state['mechanism_graph']['hypotheses'][0]['status'] == 'weakened_under_tested_conditions'
    assert [x['checkpoint_id'] for x in state['mechanism_graph']['deprioritized_checkpoints']] == ['C2']
    assert state['decision'] == 'functional_benefit_unresolved'


def graph_workflow(tmp_path, graph=True):
    from test_framework_design import seeded
    from adj_partner.contracts import ModelReply
    s, f, t, p = seeded(tmp_path, workflow_version='unified-mechanism-graph-v1')
    original = p.complete
    def complete(messages, tools, **kwargs):
        result = original(messages, tools, **kwargs)
        if not tools and 'schema' in json.loads(messages[-1]['content']):
            value = json.loads(result.content); v = value['research_design']['verification']
            v['checkpoints'] = cards('P1')
            if graph: v['mechanism_graph'] = betaine(('P1',) * 4)
            else: v.pop('mechanism_graph', None)
            return ModelReply(content=json.dumps(value), usage=result.usage)
        return result
    p.complete = complete
    return s, f, t, p


def test_graph_workflow_requests_graph_and_delivers_host_order(tmp_path):
    s, f, t, p = graph_workflow(tmp_path)
    try:
        result = f.run(t['fingerprint'])
        assert result['execution_status'] == 'succeeded', result.get('error')
        decision = next(r for r in p.requests if not r['tools'] and 'schema' in json.loads(r['messages'][-1]['content']))
        assert 'mechanism_graph' in decision['messages'][0]['content']
        design = next(r for r in result['lineage']['records'] if r['kind'] == 'design_revision')['payload']['design']
        assert design['verification']['mechanism_graph']['nodes'][1]['claim_ids'][0].startswith('claim_')
        assert '机制假设图（宿主排序，计划）' in result['delivery']['readable_content']
    finally: s.close()


def test_graph_workflow_rejects_cards_without_graph(tmp_path):
    s, f, t, p = graph_workflow(tmp_path, graph=False)
    try:
        result = f.run(t['fingerprint'])
        assert result['execution_status'] == 'failed' and 'mechanism_graph' in result['error']['message']
    finally: s.close()
