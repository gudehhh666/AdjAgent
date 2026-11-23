from pathlib import Path
import json

def test_quality_profile_is_separate_and_uses_actual_readers():
 from adj_partner.quality_entry import install_quality_profile, PROFILE
 from adj_partner.research_entry import install_profile
 from adj_partner.profiles import get_profile
 install_profile();old=get_profile('research-resource-aware');install_quality_profile()
 assert get_profile('research-resource-aware')==old
 profile=get_profile(PROFILE)
 assert {'evidence_read','geo_matrix_metadata'}<=set(profile['tools'])
 assert len(profile['tools'])<=14
 assert 'not read' in profile['instructions']

def test_evidence_cards_route_by_format_within_permissions(tmp_path):
 from adj_partner.quality_entry import evidence_cards
 from types import SimpleNamespace
 import hashlib
 (tmp_path/'config').mkdir(); f=tmp_path/'GSE123_series_matrix.txt.gz'; f.write_bytes(b'public bytes')
 item={'id':'geo_AS01','path':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'need':'expression','scope':'AS01 sample metadata','reader':'code_interpreter'}
 (tmp_path/'config/research-resource-index.json').write_text(json.dumps({'resources':[item]}))
 svc=SimpleNamespace(project_root=tmp_path)
 cfg={'allowed_roots':[str(tmp_path)],'allowed_tools':['code_interpreter','geo_matrix_metadata']}
 assert evidence_cards(svc,'AS01 paired observations',cfg)[0]['tools']==['geo_matrix_metadata']
 assert evidence_cards(svc,'AS01 paired observations',{**cfg,'allowed_tools':['code_interpreter']})[0]['tools']==['code_interpreter']
 assert evidence_cards(svc,'AS01 paired observations',{**cfg,'allowed_roots':[]})==[]

def test_cli_exposes_new_profile_without_starting_session():
 from adj_partner.cli import parser
 assert parser().parse_args(['new','--profile','research-evidence-first']).profile=='research-evidence-first'


def test_real_service_installs_new_context_and_emits_readers(tmp_path):
 from adj_partner.application import AgentService
 from adj_partner.contracts import ModelReply
 from adj_partner.models import ScriptedPort
 from adj_partner.quality_entry import EvidenceContextView
 port=ScriptedPort([ModelReply(content='No measurements have been read; only an explicit gap is justified.')])
 service=AgentService(tmp_path/'app',model=port)
 try:
  sid=service.create_session(profile='research-evidence-first')['id']
  assert isinstance(service._engine(sid).context,EvidenceContextView)
  rid=service.submit(sid,'Interpret a public data schema')['id']
  result=service.drive(rid)
  assert result['status']=='succeeded'
  names={t['function']['name'] for t in port.requests[0]['tools']}
  assert {'evidence_read','geo_matrix_metadata','web_page_read'}<=names
  assert 'not read' in port.requests[0]['messages'][0]['content']
 finally:service.close()
