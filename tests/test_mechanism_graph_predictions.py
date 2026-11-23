"""Predictive mechanism graphs (v3): host-derived predictions, value-per-cost ordering and off-path elimination."""
import json

import pytest

from adj_partner.contracts import ModelReply
from adj_partner.native.grounding import GroundingRetriever
from adj_partner.research.design import validate_design
from adj_partner.research.mechanism_graph import (analyze, discrimination, derived_dependencies,
                                                  predicted_outcomes, validate_mechanism_graph)
from adj_partner.research import graph_seed
from adj_partner.task_modes import design as design_mode, graph_grounding
from test_mechanism_graph import betaine, cards
from test_mechanism_graph_grounding import transport


COSTS = {'C1': 'low', 'C2': 'high', 'C3': 'medium', 'C4': 'low'}


def predictive(claims=('claim_x',) * 4, *, predictions=None, costs=None):
    """betaine() plus the off-path predictions and cost tiers v3 requires."""
    graph = betaine(claims)
    for mapping in graph['checkpoint_nodes']:
        mapping['cost'] = (costs or COSTS)[mapping['checkpoint_id']]
    # Off-path mechanism pairs: C1/C2 bypass H2's node, C4 bypasses H1's.
    graph['predictions'] = predictions if predictions is not None else [
        dict(checkpoint_id='C1', hypothesis_id='H2', outcome='not_observed', reason='The osmolyte route needs no BHMT activity.'),
        dict(checkpoint_id='C2', hypothesis_id='H2', outcome='not_observed', reason='Removing BHMT would not change an osmolyte phenotype.'),
        dict(checkpoint_id='C4', hypothesis_id='H1', outcome='no_prediction', reason='The BHMT route is silent on osmolyte uptake.')]
    return graph


def test_host_derives_on_path_predictions_and_the_model_supplies_only_off_path_ones():
    graph = predictive()
    outcomes = predicted_outcomes(graph, cards())
    assert outcomes[('C1', 'H1')] == 'observed' and outcomes[('C4', 'H2')] == 'observed'
    assert outcomes[('C1', 'H2')] == 'not_observed' and outcomes[('C4', 'H1')] == 'no_prediction'
    assert validate_mechanism_graph(graph, cards(), ['claim_x'])['predictions'] == graph['predictions']
    for bad, message in [
            ([*graph['predictions'], dict(checkpoint_id='C1', hypothesis_id='H1', outcome='observed', reason='x')], 'off a hypothesis path'),
            ([dict(checkpoint_id='C3', hypothesis_id='H2', outcome='observed', reason='x')], 'off a hypothesis path'),
            (graph['predictions'][:2], 'must cover every mechanism checkpoint'),
            ([dict(p, outcome='maybe') for p in graph['predictions']], 'Unknown predicted outcome')]:
        with pytest.raises(ValueError, match=message):
            validate_mechanism_graph(predictive(predictions=bad), cards(), ['claim_x'])
    missing_cost = predictive()
    del missing_cost['checkpoint_nodes'][0]['cost']
    with pytest.raises(ValueError, match='checkpoint node requires'):
        validate_mechanism_graph(missing_cost, cards(), ['claim_x'])
    with pytest.raises(ValueError, match='cost must be low'):
        validate_mechanism_graph(predictive(costs={**COSTS, 'C1': 'free'}), cards(), ['claim_x'])


def test_expected_eliminations_count_silent_hypotheses_as_a_coin_flip():
    assert discrimination(['observed', 'not_observed']) == (1.0, 1)
    assert discrimination(['observed', 'observed']) == (0.0, 0)
    assert discrimination(['observed', 'no_prediction']) == (0.25, 0)
    assert discrimination([]) == (0.0, 0)
    assert discrimination(['observed', 'not_observed', 'no_prediction']) == (1.0, 1)


def test_scores_follow_predictions_and_upstream_gating_still_comes_first():
    plan = analyze(predictive(), cards(), {})
    rows = {r['checkpoint_id']: r for r in plan['mechanism_order']}
    assert [r['checkpoint_id'] for r in plan['mechanism_order']] == ['C1', 'C4', 'C2']
    assert (rows['C1']['expected_hypotheses_eliminated'], rows['C1']['value_per_cost']) == (1.0, 1.0)
    assert rows['C1']['hypothesis_pairs_separated'] == 1 


def test_a_causal_dependence_off_path_result_eliminates_but_a_correlational_one_does_not():
    graph = predictive(('claim_bhmt', 'claim_alt', 'claim_fn', 'claim_base'))
    # C2 -> N3 is a causal_dependence checkpoint; H2 predicted the function would not require BHMT, but it does.
    plan = analyze(graph, cards(), {'C2': 'supports'})
    h1, h2 = plan['hypotheses']
    assert h2['status'] == 'weakened_under_tested_conditions' and h2['contradicted_by_off_path_checkpoints'] == ['C2']
    assert h1['status'] != 'weakened_under_tested_conditions'
    assert plan['claims_to_revisit'] == ['claim_alt']
    # A correlational activity result (C1 -> N1) that H2 predicted against does NOT eliminate H2, only flags conflict.
    corr = analyze(graph, cards(), {'C1': 'supports'})
    assert corr['hypotheses'][1]['status'] == 'conflicting'
    assert corr['hypotheses'][1]['contradicted_by_off_path_checkpoints'] == ['C1']
    assert {h['hypothesis_id'] for h in corr['hypotheses'] if h['status'] != 'weakened_under_tested_conditions'} == {'H1', 'H2'}
    # A silent hypothesis survives the same causal result.
    silent = predictive(('claim_bhmt', 'claim_alt', 'claim_fn', 'claim_base'),
                        predictions=[dict(checkpoint_id='C1', hypothesis_id='H2', outcome='no_prediction', reason='Silent.'),
                                     dict(checkpoint_id='C2', hypothesis_id='H2', outcome='no_prediction', reason='Silent.'),
                                     dict(checkpoint_id='C4', hypothesis_id='H1', outcome='no_prediction', reason='Silent.')])
    assert [h['status'] for h in analyze(silent, cards(), {'C2': 'supports'})['hypotheses']] == ['open', 'open']


def test_correlational_off_path_result_only_conflicts_a_supported_hypothesis_while_causal_eliminates():
    claims = ('claim_bhmt', 'claim_alt', 'claim_fn', 'claim_base')
    # C4 (osmolyte presence, molecular_presence) bypasses H1 and here predicts not_observed for H1.
    preds = [dict(checkpoint_id='C1', hypothesis_id='H2', outcome='not_observed', reason='Osmolyte route needs no BHMT.'),
             dict(checkpoint_id='C2', hypothesis_id='H2', outcome='not_observed', reason='No BHMT dependence.'),
             dict(checkpoint_id='C4', hypothesis_id='H1', outcome='not_observed', reason='BHMT route predicts no osmolyte uptake.')]
    graph = predictive(claims, predictions=preds)
    # H1 is supported on its own activity node (C1) yet a co-occurring osmolyte presence (C4) is also observed.
    plan = analyze(graph, cards(), {'C1': 'supports', 'C4': 'supports'})
    h1, h2 = plan['hypotheses']
    assert h1['status'] == 'conflicting' and h1['contradicted_by_off_path_checkpoints'] == ['C4']
    assert h2['status'] == 'conflicting'  # supported on its own node N4, contradicted off-path at C1 (activity)
    assert {h['hypothesis_id'] for h in plan['hypotheses'] if h['status'] != 'weakened_under_tested_conditions'} == {'H1', 'H2'}
    # A causal_dependence (C2 -> N3) observation that a hypothesis predicted against eliminates it even when supported.
    causal = analyze(graph, cards(), {'C4': 'supports', 'C2': 'supports'})
    h1c, h2c = causal['hypotheses']
    assert h2c['status'] == 'weakened_under_tested_conditions' and h2c['contradicted_by_off_path_checkpoints'] == ['C2']
    assert h1c['status'] != 'weakened_under_tested_conditions'  # the true route survives its competitor's failed dependence test


def test_predictions_are_rejected_outside_the_predictive_workflow_and_rendered_inside_it():
    from test_research_design import design
    d = design('claim_x')
    d['verification'].update(checkpoints=cards(), mechanism_graph=predictive())
    assert validate_design(d, ['claim_x'])
    from adj_partner.research.design import render_design
    text = render_design(d, {'claim_x': 1})
    text = text if isinstance(text, str) else '\n'.join(text)
    assert '期望淘汰 1.0 个假设' in text and '成本低' in text
    assert 'C2 ← C1' in text and '路径外假设的预期' in text


def test_host_seeds_candidate_families_only_for_adjuvants_the_question_names():
    from adj_partner.resources import ResourcePack
    from pathlib import Path
    pack_dir = Path(__file__).resolve().parents[1] / 'data/adjuvant_resources_v4'
    if not (pack_dir / 'manifest.json').is_file():
        pytest.skip('resource pack v4 not installed')
    pack = ResourcePack(pack_dir)
    seed = graph_seed.build(pack, 'Does alum act through the NLRP3 inflammasome in aged splenocytes?')
    assert [r['adjuvant'] for r in seed['rows']] == ['Alum']
    assert 'NLRP3 inflammasome activation' in seed['rows'][0]['reported_mechanism_families']
    assert 'not evidence' in seed['scope']
    assert graph_seed.build(pack, 'Design a betaine-based adjuvant for aged spleen immune response') is None
    assert graph_seed.build(None, 'alum') is None


def v3(tmp_path, *, predictions=None):
    from test_framework_design import seeded
    s, f, t, p = seeded(tmp_path, workflow_version=design_mode.PREDICTIVE_WORKFLOW, allow_network=True)
    f.grounding = GroundingRetriever(transport=transport())
    original = p.complete

    def complete(messages, tools, **kwargs):
        system = messages[0].get('content', '')
        if system == graph_grounding.QUERY_SYSTEM:
            return ModelReply(content=json.dumps({'edges': [{'edge': 'E1', 'queries': ['betaine BHMT activity']}]}),
                              usage={'input_tokens': 9, 'output_tokens': 9})
        if system == graph_grounding.JUDGE_SYSTEM:
            return ModelReply(content=json.dumps({'edges': [{'edge': 'E1', 'status': 'supports', 'passages': ['E1.P1'],
                                                             'note': 'Synthetic.'}]}), usage={'input_tokens': 9, 'output_tokens': 9})
        result = original(messages, tools, **kwargs)
        if not tools and 'schema' in json.loads(messages[-1]['content']):
            assert 'predictions' in messages[0]['content'] or 'predictions' in messages[-1]['content']
            value = json.loads(result.content); v = value['research_design']['verification']
            v.update(checkpoints=cards('P1'), mechanism_graph=predictive(('P1',) * 4, predictions=predictions))
            return ModelReply(content=json.dumps(value), usage=result.usage)
        return result
    p.complete = complete
    return s, f, t


def test_v3_runs_end_to_end_and_reports_the_predicted_ordering(tmp_path):
    s, f, t = v3(tmp_path)
    try:
        result = f.run(t['fingerprint'])
        assert result['execution_status'] == 'succeeded', result.get('error')
        text = result['delivery']['readable_content']
        assert '期望淘汰' in text and '机制边的文献状态' in text
        revision = [r for r in result['lineage']['records'] if r['kind'] == 'design_revision'][-1]['payload']['design']
        assert revision['verification']['mechanism_graph']['predictions']
        assert revision['verification']['literature_edges'][0]['status'] == 'supports'
    finally:
        s.close()


def test_v2_rejects_a_predictive_graph_and_v3_requires_one(tmp_path):
    from test_framework_design import seeded
    s, f, t, p = seeded(tmp_path, workflow_version=graph_grounding.WORKFLOW_VERSION, allow_network=False)
    original = p.complete

    def complete(messages, tools, **kwargs):
        result = original(messages, tools, **kwargs)
        if not tools and 'schema' in json.loads(messages[-1]['content']):
            value = json.loads(result.content); v = value['research_design']['verification']
            v.update(checkpoints=cards('P1'), mechanism_graph=predictive(('P1',) * 4))
            return ModelReply(content=json.dumps(value), usage=result.usage)
        return result
    p.complete = complete
    try:
        result = f.run(t['fingerprint'])
        assert result['execution_status'] == 'failed' and 'predictive workflow' in result['error']['message']
    finally:
        s.close()
