from copy import deepcopy
import json

import pytest

from adj_partner.research.molecule_reports import validate_plan, render_molecule, render_index


def profile():
    return {'alias': 'T01', 'compound': 'Aminosalicylic',
            'source': {'source_name': 'input.xlsx', 'source_sha256': 'a'*64},
            'context': '老年鼠脾细胞与抗原共同处理；历史剂量和时间未知。',
            'counts': {'genes': 2, 'vendor_flagged': 1, 'eligible_flagged': 1},
            'genes': [{'gene_id': 'g1', 'gene_name': 'Cdkn1a', 'vehicle_fpkm': 43.1,
                       'treatment_fpkm': 17.6, 'vendor_log2fc': -1.3, 'vendor_qval': .01,
                       'vendor_flag': True, 'panel_relative_log2fc': -.5}],
            'dose_strategy': {'status': 'requires_original_concentration', 'unit': None,
                              'pilot_series': [], 'anchor': 'C0须从原始实验记录恢复；随后按0.25/0.5/1/2倍C0校准。',
                              'reference_doses': [], 'limitations': ['异构体未知；不提供动物剂量。']}}


def plan():
    value={'alias': 'T01', 'compound': 'Aminosalicylic', 'decision': 'exploratory',
            'summary': '优先复现分子变化，功能仍待测。', 'mechanism_hypothesis': '检查点变化也可能来自组成变化。',
            'dose_rationale': '先明确化学身份并恢复原始浓度，再做暴露校准。','focus_gene_ids':['g1'],
            'experiments': [{'id': 'E'+str(i), 'stage': s, 'question': q,
                             'groups': ['匹配vehicle', '本分子'], 'readouts': ['存活', '抗原功能'],
                             'quality_rule': '独立动物为重复，检测失败保留不可判定。',
                             'continue_if': '分子复现且功能保持。', 'stop_if': '功能受损则停止。',
                             'inconclusive_if': '暴露或检测失败。', 'source_ids': ['local']}
                            for i,s,q in [(1,'exposure','条件校准'), (2,'molecular','复现Cdkn1a'), (3,'function','测抗原功能')]],
            'data_observations':[{'gene_id':'g1','direction_vs_vehicle':'down'}],
            'limitations': ['蛋白及湿实验未测。'], 'source_ids': ['local']}
    value['experiments'][2]['function_design']={'antigen_positive_group':'Ag+本分子 vs Ag+vehicle',
            'antigen_negative_group':'Ag-本分子 vs Ag-vehicle',
            'washout_restimulation':'洗脱后以同抗原再刺激，检查残留影响。',
            'live_cell_output':'活细胞绝对产出并按起始目标细胞数归一。',
            'primary_endpoint':'Ag-背景扣除后的IFNγ+活目标T细胞绝对产出。'}
    return value


def test_standalone_contains_dose_experiments_and_local_values():
    p = validate_plan(plan(), profile())
    content = render_molecule(profile(), p).decode()
    assert '剂量建议' in content and '实验推荐' in content
    assert '43.1' in content and '17.6' in content
    assert 'C0' in content and '历史剂量' in content
    assert 'Saxagliptin' not in content


@pytest.mark.parametrize('field,value', [('alias','T02'), ('compound','Betaine')])
def test_cross_molecule_identity_rejected(field, value):
    x=plan(); x[field]=value
    with pytest.raises(ValueError, match='identity'): validate_plan(x, profile())


def test_missing_function_and_unknown_sources_rejected():
    x=plan();x['experiments'][2]['stage']='mechanism'
    with pytest.raises(ValueError, match='function'): validate_plan(x, profile())
    x=plan();x['source_ids']=['invented']
    with pytest.raises(ValueError, match='source'): validate_plan(x, profile())


@pytest.mark.parametrize('dose',['本分子1000 mM','本分子十毫摩尔每升','本分子100 μg/mL'])
def test_numeric_dose_in_experiment_prose_is_rejected(dose):
    x=plan();x['experiments'][0]['groups']=[dose]
    with pytest.raises(ValueError,match='Numeric dose'):validate_plan(x,profile())


def test_input_declared_coverage_is_enforced(tmp_path):
    from adj_partner.molecule_reports import _load_bundle
    from adj_partner.research.molecule_reports import VERSION
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'schema_version':VERSION,'profiles':[profile()], 'provenance':{'treatment_coverage':26}}))
    with pytest.raises(ValueError,match='coverage'):_load_bundle(bundle)


def test_function_card_needs_explicit_comparisons_and_normalization():
    x=plan();x['experiments'][2].pop('function_design')
    with pytest.raises(ValueError,match='function design'):validate_plan(x,profile())


def test_observation_directions_are_bound_to_vehicle_values():
    x=plan();x['data_observations'][0]['direction_vs_vehicle']='up'
    with pytest.raises(ValueError,match='direction'):validate_plan(x,profile())


def test_unrelated_molecule_cannot_copy_bhmt_template():
    x=plan();x['limitations'].append('BHMT酶活仍未知。')
    with pytest.raises(ValueError,match='BHMT'):validate_plan(x,profile())


def test_function_recommendation_selects_a_concrete_primary_readout():
    x=plan();x['experiments'][2]['function_design']['primary_endpoint']='功能指标或细胞因子变化。'
    with pytest.raises(ValueError,match='primary endpoint'):validate_plan(x,profile())


def test_model_view_has_absolute_direction_without_panel_residual_or_unrelated_bhmt():
    from adj_partner.molecule_reports import make_plan_request
    p=profile();p['genes'].append({'gene_id':'g2','gene_name':'Bhmt','vehicle_fpkm':0,'treatment_fpkm':0,
                                'vendor_log2fc':0,'vendor_qval':1,'vendor_flag':False,'panel_relative_log2fc':0})
    messages=make_plan_request(p)
    data=json.loads(messages[-1]['content'])
    assert all(g['gene_name']!='Bhmt' for g in data['molecule_input']['genes'])
    assert 'panel_relative_log2fc' not in messages[-1]['content']
    assert data['molecule_input']['genes'][0]['direction_vs_vehicle']=='down'
    assert 'BHMT' not in messages[0]['content']
    assert data['output_json_schema']['properties']['decision']['enum']==['priority','exploratory','defer','comparison']


def test_numeric_pilot_requires_traceable_reference():
    p=profile();p['dose_strategy'].update(status='literature_anchored_pilot',unit='uM',pilot_series=[.1,1,10])
    with pytest.raises(ValueError, match='dose'): validate_plan(plan(), p)


def test_index_requires_every_input_once_and_escapes_html():
    p=profile();a=plan()
    with pytest.raises(ValueError, match='coverage'): render_index([p], {})
    p['compound']='<img src=x onerror=alert(1)>'
    a['compound']=p['compound']
    markdown, html=render_index([p], {'T01':a})
    assert b'<img' not in html and b'&lt;img' in html
    assert b'molecules/T01.md' in markdown


def test_numeric_dose_is_rendered_as_proposal_with_exact_reference():
    p=profile();p['dose_strategy'].update(status='literature_anchored_pilot',unit='uM',pilot_series=[.1,1,10],
        reference_doses=[{'id':'D1','url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC123/',
                         'title':'Example study','scope':'full_text','compound':'Aminosalicylic',
                         'concentrations':[1,10], 'unit':'uM','context':'different cells',
                         'locator':'Methods, paragraph 2','transfer_limit':'本体系尚未验证。'}])
    result=render_molecule(p,validate_plan(plan(),p)).decode()
    assert '0.1 / 1 / 10 uM' in result
    assert '拟议' in result and 'PMC123' in result and 'Methods, paragraph 2' in result
    altered=deepcopy(p);altered['dose_strategy']['reference_doses'][0]['compound']='Betaine'
    with pytest.raises(ValueError,match='dose'): validate_plan(plan(),altered)


def test_batch_uses_real_ledger_and_does_not_replay_success(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.contracts import ModelReply
    from adj_partner.models import ScriptedPort
    from adj_partner.molecule_reports import run_bundle, verify_delivery
    from adj_partner.research.molecule_reports import VERSION
    bundle=tmp_path/'bundle.json'
    bundle.write_text(json.dumps({'schema_version':VERSION,'profiles':[profile()],'provenance':{}}))
    model=ScriptedPort([ModelReply(content=json.dumps(plan()),usage={'input_tokens':300,'output_tokens':200})])
    svc=AgentService(tmp_path/'state',model=model)
    try:
        out=tmp_path/'delivery';result=run_bundle(bundle,out,workers=1,service=svc)
        assert result['delivered']==1 and result['execution_status']=='succeeded'
        receipt=result['receipts']['T01']
        assert svc.store.get_run(receipt['run_id'])['status']=='succeeded'
        assert receipt['source_id'] and receipt['proposal_id']
        assert verify_delivery(out)['accepted']
        again=run_bundle(bundle,out,workers=1,service=svc)
        assert again['attempts']==1
        (out/'plans/T01.json').write_text('{}')
        assert not verify_delivery(out)['accepted']
    finally:svc.close()


def test_host_publication_is_derived_and_does_not_invoke_model(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.contracts import ModelReply
    from adj_partner.models import ScriptedPort
    from adj_partner.molecule_reports import run_bundle, verify_delivery
    from adj_partner.molecule_publication import publish_reviewed_delivery
    from adj_partner.research.molecule_reports import VERSION
    bundle=tmp_path/'bundle.json'
    bundle.write_text(json.dumps({'schema_version':VERSION,'profiles':[profile()],'provenance':{}}))
    svc=AgentService(tmp_path/'state',model=ScriptedPort([ModelReply(content=json.dumps(plan()))]))
    try:
        source=tmp_path/'source';run_bundle(bundle,source,workers=1,service=svc)
        raw=(source/'plans/T01.json').read_bytes()
        final=tmp_path/'final';result=publish_reviewed_delivery(source,final)
        assert result['delivered']==1 and result['publication']['new_model_requests']==0
        assert (source/'plans/T01.json').read_bytes()==raw
        assert verify_delivery(final)['accepted']
        revised=json.loads((final/'plans/T01.json').read_text())
        revised['experiments'][2]['function_design']['primary_endpoint']='IFNγ错误背景'
        (final/'plans/T01.json').write_text(json.dumps(revised))
        assert not verify_delivery(final)['accepted']
    finally:svc.close()


def test_failed_attempt_preserved_and_retry_is_explicit(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.contracts import ModelReply
    from adj_partner.models import ScriptedPort
    from adj_partner.molecule_reports import run_bundle
    from adj_partner.research.molecule_reports import VERSION
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'schema_version':VERSION,'profiles':[profile()],'provenance':{}}))
    model=ScriptedPort([ModelReply(content='{}',usage={'input_tokens':1,'output_tokens':1}),
                       ModelReply(content=json.dumps(plan()),usage={'input_tokens':1,'output_tokens':1})])
    svc=AgentService(tmp_path/'state',model=model)
    try:
        out=tmp_path/'out';result=run_bundle(bundle,out,workers=1,service=svc)
        assert result['delivered']==0 and result['failed_attempts']==1
        assert run_bundle(bundle,out,workers=1,service=svc)['attempts']==1
        done=run_bundle(bundle,out,workers=1,retry_failed=True,service=svc)
        assert done['delivered']==1 and done['attempts']==2 and done['failed_attempts']==1
        assert json.loads((out/'attempts/T01.attempt01/receipt.json').read_text())['status']=='failed'
    finally:svc.close()


def test_resume_and_verification_reject_changed_input(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.contracts import ModelReply
    from adj_partner.models import ScriptedPort
    from adj_partner.molecule_reports import run_bundle,verify_delivery
    from adj_partner.research.molecule_reports import VERSION
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'schema_version':VERSION,'profiles':[profile()],'provenance':{}}))
    svc=AgentService(tmp_path/'state',model=ScriptedPort([ModelReply(content=json.dumps(plan()),usage={'input_tokens':1,'output_tokens':1})]))
    try:
        out=tmp_path/'out';run_bundle(bundle,out,workers=1,service=svc)
        value=json.loads(bundle.read_text());value['profiles'][0]['context']='Changed context';bundle.write_text(json.dumps(value))
        with pytest.raises(ValueError,match='changed'):run_bundle(bundle,out,workers=1,service=svc)
        copied=json.loads((out/'bundle.json').read_text());copied['provenance']['changed']=True
        (out/'bundle.json').write_text(json.dumps(copied))
        with pytest.raises(ValueError,match='Bundle'): verify_delivery(out)
    finally:svc.close()


def test_setup_failure_is_recorded_without_losing_batch(tmp_path, monkeypatch):
    from adj_partner.application import AgentService
    from adj_partner.molecule_reports import run_bundle
    from adj_partner.research.molecule_reports import VERSION
    from adj_partner.models import ScriptedPort
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'schema_version':VERSION,'profiles':[profile()], 'provenance':{}}))
    svc=AgentService(tmp_path/'state',model=ScriptedPort([]))
    def unavailable(**kwargs): raise ValueError('Synthetic setup failure')
    monkeypatch.setattr(svc,'create_session',unavailable)
    try:
        result=run_bundle(bundle,tmp_path/'out',workers=1,service=svc)
        assert result['execution_status']=='partial' and result['failed_attempts']==1
        assert result['receipts']['T01']['error_type']=='ValueError'
        assert run_bundle(bundle,tmp_path/'out',workers=1,service=svc)['attempts']==1
    finally:svc.close()


def test_committed_success_with_stale_receipt_is_reconciled_without_call(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.contracts import ModelReply
    from adj_partner.models import ScriptedPort
    from adj_partner.molecule_reports import run_bundle
    from adj_partner.research.molecule_reports import VERSION
    bundle=tmp_path/'bundle.json';bundle.write_text(json.dumps({'schema_version':VERSION,'profiles':[profile()], 'provenance':{}}))
    svc=AgentService(tmp_path/'state',model=ScriptedPort([ModelReply(content=json.dumps(plan()),usage={'input_tokens':1,'output_tokens':1})]))
    try:
        out=tmp_path/'out';run_bundle(bundle,out,workers=1,service=svc)
        path=out/'attempts/T01.attempt01/receipt.json';r=json.loads(path.read_text());r['status']='committing';path.write_text(json.dumps(r))
        result=run_bundle(bundle,out,workers=1,service=svc)
        assert result['delivered']==1 and result['attempts']==1
        assert result['receipts']['T01']['completion_reconciled'] is True
    finally:svc.close()
