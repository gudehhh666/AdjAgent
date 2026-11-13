import json
from pathlib import Path
import pytest
from adj_partner.resources.pack import ResourcePack
from adj_partner.profiles import get_profile
from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.native.capabilities import NativeCapabilities
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor
from adj_partner.runtime.context import ContextView
from test_adjuvant_resource_pack import make_pack


def test_ranked_discovery_retains_rare_entity_without_weakening_exact_filters(tmp_path):
    pack=ResourcePack(make_pack(tmp_path/'pack'))
    assert pack.search('How can aluminum hydroxide activate dendritic cells?')['hit_count']==0
    result=pack.search('How can aluminum hydroxide activate dendritic cells?',match_mode='ranked')
    assert result['results'][0]['id']=='VO_1'
    assert result['retrieval_matches']['VO_1']['matched_terms']==['aluminum','hydroxide']
    assert pack.search('unobtainiumase',match_mode='ranked')['hit_count']==0
    assert pack.search('S1002 unrelated',dataset='screen',filters={'MPLA Lum':'wrong'},match_mode='ranked')['hit_count']==0
    with pytest.raises(ValueError):pack.search('alum',match_mode='invented')


def setup_tools(tmp_path):
    pack=ResourcePack(make_pack(tmp_path/'pack'))
    pack.records['VO_1']['fields']['protocol/text']='prefix '+('长文本'*900)+' exact ending'
    store=Store(tmp_path/'state');catalog=ToolCatalog();research=ResearchStore(store)
    NativeCapabilities(store,research=research,resource_pack=pack).register(catalog)
    names=['adjuvant_skill','adjuvant_excerpt']
    sid=store.create_session(SessionConfig(allowed_tools=names,enabled_tools=names,allow_network=False).to_dict())['id']
    tid=store.create_task(sid,'public lookup',{})['id'];rid=store.create_run(sid,tid,Budget().to_dict())['id']
    return pack,store,catalog,ToolExecutor(store,catalog),sid,rid


def test_excerpt_survives_model_projection_and_keeps_literal_offsets(tmp_path):
    pack,store,catalog,executor,sid,rid=setup_tools(tmp_path)
    result=executor.execute(ToolCall('read','adjuvant_excerpt',{'record_id':'VO_1','field':'protocol/text','start':2600,'length':200}),sid,rid)
    assert result.status=='ok',result.to_dict()
    message=ContextView(store).tool_message(sid,rid,'read',result.to_dict())
    view=json.loads(message['content'])['data']['summary']
    assert view['text']==pack.records['VO_1']['fields']['protocol/text'][2600:2800]
    assert view['selector']['pointer']=='/record/fields/protocol~1text'
    assert view['start']==2600 and view['end']==2720 and view['total_chars']==2720
    assert len(message['content'].encode())<=ContextView.PREVIEW_MAX_BYTES
    bad=executor.execute(ToolCall('bad','adjuvant_excerpt',{'record_id':'VO_1','field':'missing'}),sid,rid)
    assert bad.status!='ok'


def test_skills_are_fixed_local_guidance_and_profile_does_not_expand_permissions(tmp_path):
    if not (Path(__file__).resolve().parents[1]/'config/skills/adjuvant-mechanism-evidence/SKILL.md').exists():
        pytest.skip('needs the resource entry guides, which are not part of this checkout')
    _,store,catalog,executor,sid,rid=setup_tools(tmp_path)
    result=executor.execute(ToolCall('skill','adjuvant_skill',{'skill':'mechanism'}),sid,rid)
    assert result.status=='ok',result.to_dict()
    view=json.loads(ContextView(store).tool_message(sid,rid,'skill',result.to_dict())['content'])['data']['summary']
    assert 'enzyme activity' in view['instructions'] and 'rhea' in view['instructions']
    assert view['sha256'] and len(view['sha256'])==64
    assert get_profile('adjuvant-evidence')['tools'][-2:]==['adjuvant_skill','adjuvant_excerpt']
    denied=executor.execute(ToolCall('denied','adjuvant_search',{'query':'alum'}),sid,rid)
    assert denied.status!='ok'
    bad=executor.execute(ToolCall('path','adjuvant_skill',{'skill':'../../.env'}),sid,rid)
    assert bad.status!='ok'


def test_structured_annotation_excerpt_labels_json_representation(tmp_path):
    pack,store,catalog,executor,sid,rid=setup_tools(tmp_path)
    pack.records['VO_1']['fields']['annotations']=[{'function':'curated annotation, not measured activity'}]
    result=executor.execute(ToolCall('structured','adjuvant_excerpt',{'record_id':'VO_1','field':'annotations'}),sid,rid)
    assert result.status=='ok'
    summary=result.data['summary']
    assert summary['representation']=='canonical_json'
    assert json.loads(summary['text'])==pack.records['VO_1']['fields']['annotations']
