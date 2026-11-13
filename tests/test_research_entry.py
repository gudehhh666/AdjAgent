from types import SimpleNamespace
from pathlib import Path
import json
import pytest
from adj_partner.research_entry import resource_entries, resource_page, install_profile

def pack():
 return SimpleNamespace(datasets={'uniprot_adjuvant':{'evidence_scope':'curated annotation'},'immgen_studies':{'evidence_scope':'study metadata only'},'ncl_protocols':{'evidence_scope':'assay protocol text'}}, manifest_hash='a'*64)

def test_catalog_filters_permissions_and_does_not_invent_measurements():
 entries=resource_entries(pack(),{'adjuvant_resource_guide','adjuvant_lookup'},False)
 assert {x['id'] for x in entries}=={'uniprot_adjuvant','immgen_studies','ncl_protocols'}
 assert next(x for x in entries if x['id']=='immgen_studies')['scope']=='study metadata only'
 assert resource_entries(pack(),set(),True)==[]

def test_network_permission_is_not_resource_success():
 e=resource_entries(None,{'literature_search','web_search'},False)
 assert e==[]
 e=resource_entries(None,{'literature_search','web_search'},True)
 assert all(x['readiness']=='configured_not_live_verified' for x in e)
 assert {x['id'] for x in e}=={'literature','web'}

def test_page_is_bounded_and_need_specific():
 p=resource_page(resource_entries(pack(),{'adjuvant_resource_guide','adjuvant_lookup'},False),'assay',0)
 assert [x['id'] for x in p['resources']]==['ncl_protocols']
 assert len(json.dumps(p).encode())<1800

def test_profile_is_opt_in_and_has_bounded_initial_toolset():
 from adj_partner.profiles import get_profile
 before=get_profile('general');install_profile()
 assert get_profile('general')==before
 p=get_profile('research-resource-aware')
 assert {'research_resources','research_skill','catalog_search','tools_enable'}<=set(p['tools'])
 assert len(p['tools'])<=12
 assert 'adjuvant_lookup' not in p['tools']

def test_query_matches_resource_metadata_without_assuming_results():
 entries=[{'id':'new_matrix','need':'dataset','scope':'Public AS01 processed matrix','tools':['file_inspect']}, {'id':'unrelated','need':'dataset','scope':'Traffic counts','tools':['file_inspect']}]
 assert [e['id'] for e in resource_page(entries,'dataset',0,query='AS01')['resources']]==['new_matrix']
 assert resource_page(entries,'dataset',0,query='not_in_catalog')['resources']==[]

def test_actual_session_catalog_is_small_and_permissions_remain_authoritative(tmp_path):
 from adj_partner.application import AgentService
 from adj_partner.contracts import ModelReply
 class Fake:
  secrets=()
  metadata={'model':'test'}
  def complete(self,*a,**k):return ModelReply(content='ok')
 s=AgentService(tmp_path/'state',model=Fake(),project_root=Path(__file__).resolve().parents[1])
 try:
  x=s.create_session(profile='research-resource-aware',allowed_tools=['research_resources','tools_enable'],allow_network=False)
  assert x['config']['enabled_tools']==['research_resources','tools_enable']
  assert not x['config']['allow_network']
 finally:s.close()

def test_asset_catalog_checks_hash_root_and_real_reader(tmp_path):
 import hashlib
 from adj_partner.research_entry import file_resources
 f=tmp_path/'matrix.csv.gz';f.write_bytes(b'example-compressed-bytes')
 items=[{'id':'matrix','path':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'need':'dataset','scope':'Processed data','reader':'code_interpreter'}]
 config={'allowed_tools':['code_interpreter'],'allowed_roots':[str(tmp_path)]}
 e=file_resources(tmp_path,items,config)
 assert e[0]['tools']==['code_interpreter'] and e[0]['path']==str(f)
 assert file_resources(tmp_path,items,{**config,'allowed_tools':['file_inspect']})==[]
 assert file_resources(tmp_path,items,{**config,'allowed_roots':[]})==[]
 f.write_bytes(b'changed');assert file_resources(tmp_path,items,config)==[]

def test_catalog_and_skills_survive_real_tool_projection(tmp_path):
 from adj_partner.application import AgentService
 from adj_partner.models import ScriptedPort
 from adj_partner.contracts import ToolCall
 from adj_partner.runtime.context import ContextView
 root=Path(__file__).resolve().parents[1]
 if not (root/'config/research-resource-index.json').exists():pytest.skip('needs config/research-resource-index.json, which is not part of this checkout')
 svc=AgentService(tmp_path/'state',project_root=root,model=ScriptedPort([]))
 try:
  sid=svc.create_session(profile='research-resource-aware',allowed_tools=['research_resources','research_skill','code_interpreter','adjuvant_resource_guide','adjuvant_lookup'],allowed_roots=[str(root/'data/research_public_inputs')])['id']
  run=svc.submit(sid,'public research')
  for n,args in [('research_resources',{'need':'dataset','query':'AS01'}),('research_skill',{'stage':'design'})]:
   r=svc.executor.execute(ToolCall(n,n,args),sid,run['id']);assert r.status=='ok',r.to_dict()
   m=ContextView(svc.store).tool_message(sid,run['id'],n,r.to_dict());v=json.loads(m['content'])
   assert not v.get('context_truncated') and len(m['content'].encode())<=2400
   if n=='research_resources':assert v['data']['summary']['resources']
  assert 'code_interpreter' not in svc.session(sid)['config']['enabled_tools']
 finally:svc.close()

def test_cli_exposes_optional_profile_before_service_construction():
 from adj_partner.profiles import PROFILES
 from adj_partner.cli import parser
 saved=PROFILES.pop('research-resource-aware',None)
 try:assert parser().parse_args(['new','--profile','research-resource-aware']).profile=='research-resource-aware'
 finally:
  if saved:PROFILES['research-resource-aware']=saved

def test_paper_topic_does_not_hide_configured_search_endpoints():
 entries=resource_entries(None,{'literature_search'},True)
 p=resource_page(entries,'papers',query='low-code no-code productivity maintenance '*8)
 assert [r['id'] for r in p['resources']]==['literature']
 assert p['topic_search_executed'] is False

def test_initial_cards_match_entities_not_case_ids_and_respect_permissions(tmp_path):
 from adj_partner.research_entry import initial_resource_cards
 import hashlib
 root=tmp_path/'project';(root/'config').mkdir(parents=True);(root/'data').mkdir()
 f=root/'data'/'public.txt';f.write_text('public measurement')
 items=[{'id':'test_AS01_metadata','need':'expression','path':'data/public.txt','sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'scope':'Official AS01 sample metadata; not expression values','reader':'code_interpreter','source_family':'study1'}]
 (root/'config/research-resource-index.json').write_text(json.dumps({'resources':items}))
 svc=SimpleNamespace(project_root=root)
 cfg={'allowed_tools':['code_interpreter'],'allowed_roots':[str(root/'data')]}
 assert initial_resource_cards(svc,'AS01相关公开人体测序',cfg)[0]['id']=='test_AS01_metadata'
 assert initial_resource_cards(svc,'D2',cfg)==[]
 assert initial_resource_cards(svc,'AS01研究',{**cfg,'allowed_roots':[]})==[]
 assert initial_resource_cards(svc,'low-code productivity',cfg)==[]
 assert len(json.dumps(initial_resource_cards(svc,'AS01研究',cfg),ensure_ascii=False).encode())<=1250
