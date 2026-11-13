from copy import deepcopy

import pytest

from adj_partner.research.molecule_protocol import functional_contrast, normalize_function_design


def test_endpoint_counts_include_survival_before_starting_count_normalization():
    # Half of 1,000 surviving cells are positive, from 10,000 starting cells.
    wells = {
        'C+': {'positive_fraction': .5, 'endpoint_live_count': 1000, 'starting_count': 10000},
        'C-': {'positive_fraction': .1, 'endpoint_live_count': 1000, 'starting_count': 10000},
        'V+': {'positive_fraction': .2, 'endpoint_live_count': 10000, 'starting_count': 10000},
        'V-': {'positive_fraction': .1, 'endpoint_live_count': 10000, 'starting_count': 10000},
    }
    result = functional_contrast(wells)
    assert result['Y']['C+'] == pytest.approx(.05)
    assert result['F_C'] == pytest.approx(.04)
    assert result['F_V'] == pytest.approx(.1)
    assert result['delta'] == pytest.approx(-.06)


def test_each_arm_uses_its_own_antigen_negative_well_and_no_clipping():
    wells = {key: {'positive_fraction': p, 'endpoint_live_count': 100, 'starting_count': 100}
             for key, p in {'C+': .2, 'C-': .3, 'V+': .4, 'V-': .1}.items()}
    result = functional_contrast(wells)
    assert result['F_C'] == pytest.approx(-.1)
    assert result['delta'] == pytest.approx(-.4)
    wells['C-']['starting_count'] = 0
    with pytest.raises(ValueError): functional_contrast(wells)


def test_normalization_preserves_raw_model_and_molecule_evidence():
    plan = {'alias': 'T12', 'compound': 'Flunarizine', 'source_ids': ['local'],
            'focus_gene_ids': ['g1'], 'data_observations': [{'gene_id': 'g1', 'direction_vs_vehicle': 'up'}],
            'limitations': ['未执行湿实验。'],
            'experiments': [
                {'stage': 'molecular', 'readouts': ['本分子g1蛋白']},
                {'stage': 'function', 'groups': ['经验证的C0倍数先导'],
                 'readouts': ['IFNγ阳性比例乘以起始细胞数', '扣除vehicle无抗原背景', '增殖与存活'],
                 'function_design': {'antigen_positive_group': 'C+ vs V+',
                   'antigen_negative_group': '仅V-', 'washout_restimulation': '洗脱后再刺激。',
                   'live_cell_output': '起始细胞数乘阳性比例。', 'primary_endpoint': 'IFNγ产出'}}]}
    before = deepcopy(plan)
    published, ledger = normalize_function_design(plan)
    assert plan == before
    assert published['source_ids'] == plan['source_ids']
    assert published['data_observations'] == plan['data_observations']
    assert published['experiments'][0] == plan['experiments'][0]
    function = published['experiments'][1]
    assert '经验证' not in function['groups'][0]
    assert 'C−' in function['function_design']['antigen_negative_group']
    assert 'N_end' in function['function_design']['live_cell_output']
    assert function['readouts'][-1] == '增殖与存活'
    assert ledger['changes'] and ledger['raw_plan_sha256'] != ledger['published_plan_sha256']
    assert normalize_function_design(plan) == (published, ledger)


def test_recorded_wording_corrections_do_not_invent_protein_or_function_results():
    t23={'compound':'Thioridazine','mechanism_hypothesis':'Traj16缺失提示T细胞状态可能改变。',
         'limitations':['未测蛋白。'],'experiments':[{'stage':'molecular','readouts':[
         '分选后检测Hmox1、Slc7a11、Nlrp3、Il1b、Mmp9、Ifng和Traj16的RNA/蛋白。']}]}
    revised,ledger=normalize_function_design(t23)
    assert 'Traj16缺失' not in revised['mechanism_hypothesis']
    assert 'Traj16仅作RNA/TCR组成分析' in str(revised['experiments'])
    assert any('/readouts/' in change['path'] for change in ledger['changes'])
    t20={'compound':'Rapamycin','limitations':['未测功能。'],'experiments':[{'stage':'mechanism',
         'question':'Rapamycin导致的T细胞功能下降是否依赖可逆mTOR通路抑制，还是由髓系炎症、组成变化或非特异性毒性驱动？'}]}
    revised,_=normalize_function_design(t20)
    assert revised['experiments'][0]['question'].startswith('若E3确认')
