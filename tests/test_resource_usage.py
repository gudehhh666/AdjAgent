import pytest
from adj_partner.resources.pack import ResourcePack
from test_adjuvant_resource_pack import make_pack


def test_usage_cards_cover_all_installed_datasets():
    from adj_partner.resources.usage import resource_guide,USAGE
    p=ResourcePack('data/adjuvant_resources_v4')
    assert set(p.datasets)<=set(USAGE)
    from adj_partner.resources.skills import SKILLS
    assert set(p.datasets)<=set(d for values in SKILLS.values() for d in values)
    for dataset in p.datasets:
        g=resource_guide(p,dataset)
        assert g['dataset']==dataset and g['purpose'] and g['boundary']
        assert g['fields'] and g['example_record_id'] in p.records
        assert g['next_offset'] is None or g['next_offset']>0
        assert all(x['types'] for x in g['fields'])


def test_focus_excerpt_has_exact_nested_pointer_and_typed_value():
    from adj_partner.resources.usage import excerpt_fields
    r={'id':'test','fields':{'annotations':[{'text':'prefix '*200+'enzyme inhibition is condition dependent. '+'tail '*200}], 'empty':None,'number':2.3,'formula':'=SUM(A1:A3)'}}
    out=excerpt_fields(r,['annotations'],'inhibition',limit=1,width=180)
    x=out[0]
    assert x['selector']['pointer']=='/record/fields/annotations/0/text'
    assert 'inhibition' in x['text'] and x['start']>0
    assert x['text']==r['fields']['annotations'][0]['text'][x['start']:x['end']]
    out=excerpt_fields(r,['empty','number','formula'],'',limit=3,width=180)
    assert [x['value'] for x in out]==[None,2.3,'=SUM(A1:A3)']


def test_unknown_field_or_resource_is_not_silently_replaced(tmp_path):
    from adj_partner.resources.usage import resource_guide,excerpt_fields
    p=ResourcePack(make_pack(tmp_path/'pack'))
    with pytest.raises(ValueError):resource_guide(p,'made_up')
    with pytest.raises(ValueError):excerpt_fields(p.records['VO_1'],['made_up'],'',limit=1)


def test_focused_search_keeps_query_anchor_and_changes_only_ranking():
    p=ResourcePack('data/adjuvant_resources_v4')
    r=p.search('endotoxin',dataset='ncl_protocols',kind='assay_protocol_page',focus='interference recovery',limit=3)
    assert any(x['id'].startswith('ncl:STE-1.1:page:') for x in r['results'])
    assert p.search('ZXQ_UNSEEN_ENZYME_472',dataset='uniprot_adjuvant',focus='enzyme activity')['hit_count']==0


def test_native_guides_and_lookup_expose_source_bound_fragments(tmp_path):
    import json
    from adj_partner.contracts import SessionConfig,Budget,ToolCall
    from adj_partner.storage import Store
    from adj_partner.research import ResearchStore
    from adj_partner.native.capabilities import NativeCapabilities
    from adj_partner.tools import ToolCatalog,ToolExecutor
    from adj_partner.runtime.context import ContextView
    p=ResourcePack('data/adjuvant_resources_v4');store=Store(tmp_path/'state');research=ResearchStore(store);catalog=ToolCatalog()
    NativeCapabilities(store,research=research,resource_pack=p).register(catalog)
    names=['adjuvant_resource_guide','adjuvant_lookup'];sid=store.create_session(SessionConfig(allowed_tools=names,enabled_tools=names).to_dict())['id']
    tid=store.create_task(sid,'protocol interference',{})['id'];rid=store.create_run(sid,tid,Budget().to_dict())['id'];executor=ToolExecutor(store,catalog)
    g=executor.execute(ToolCall('guide','adjuvant_resource_guide',{'dataset':'ncl_protocols','kind':'assay_protocol_page'}),sid,rid)
    assert g.status=='ok',g.to_dict()
    r=executor.execute(ToolCall('lookup','adjuvant_lookup',{'dataset':'ncl_protocols','kind':'assay_protocol_page','query':'endotoxin','focus':'interference recovery','fields':['text']}),sid,rid)
    assert r.status=='ok',r.to_dict()
    msg=ContextView(store).tool_message(sid,rid,'lookup',r.to_dict())
    view=json.loads(msg['content'])['data']['summary']
    assert len(msg['content'].encode())<=ContextView.PREVIEW_MAX_BYTES
    assert view['fragments'] and not view.get('fragments_omitted',0)
    for f in view['fragments']:
        selected=research.select_evidence(sid,rid,f['source_id'],f['selector'])
        assert f['text'] in selected['value']
        assert f['raw_sha256'] and f['record_id'] in p.records


def test_executable_recipes_target_identity_and_protocol_body():
    from adj_partner.resources.usage import resource_guide,excerpt_fields
    p=ResourcePack('data/adjuvant_resources_v4')
    protein=resource_guide(p,'uniprot_adjuvant')['lookup_example']
    q={k:v for k,v in protein.items() if k not in ('fields',)}
    rows=p.search(**q)['results']
    assert rows and all(r['fields']['gene']==protein['filters']['gene'] and r['fields']['taxon_id']==protein['filters']['taxon_id'] for r in rows)
    protocol=resource_guide(p,'ncl_protocols',kind='assay_protocol_page')['lookup_example']
    r=p.search(**{k:v for k,v in protocol.items() if k!='fields'},limit=1)['results'][0]
    assert r['id']=='ncl:ITA-14:page:3'
    assert 'assessing' in excerpt_fields(r,['text'],protocol['focus'])[0]['text']
