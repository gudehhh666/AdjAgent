"""Zero-provider end-to-end action-stage checks using real local analysis jobs."""
import hashlib
import json

import pytest

from adj_partner.application import AgentService
from adj_partner.contracts import ModelReply, ToolCall
from adj_partner.models.base import ScriptedPort


SCHEMA={'type':'object','properties':{'sum':{'type':'integer'},'count':{'type':'integer'}},
        'required':['sum','count'],'additionalProperties':False}
ACCEPTANCE={'response_json_schema':SCHEMA,'require_bound_execution_witness':True,
            'execution_bound_delivery':True,'require_execution_action':True}
USAGE={'input_tokens':1,'output_tokens':1}


def service_for(tmp_path, replies, files):
    empty=tmp_path/'empty.env'; empty.write_text('')
    port=ScriptedPort(replies)
    service=AgentService(tmp_path/'state',project_root=tmp_path,model=port,env_file=empty)
    sid=service.create_session(profile='general',instructions='Perform the public calculation.',
        allowed_roots=[str(tmp_path/'public')],allow_network=False,
        allowed_tools=['code_interpreter','file_inspect','job_status'],
        budget={'max_steps':4,'max_input_tokens':100000,'max_output_tokens':4000,'wall_time_seconds':120})['id']
    run=service.submit(sid,'Read the bound data and return its value-column sum and row count.',
        acceptance=ACCEPTANCE,execution_binding={'allowed_roots':[str(tmp_path/'public')],
                                               'input_paths':[str(p) for p in files]})
    return service,port,sid,run['id']


@pytest.mark.real_sandbox
@pytest.mark.parametrize('kind',['csv','dta'])
def test_real_bound_analysis_obeys_required_action_and_direct_delivery(tmp_path,kind):
    public=tmp_path/'public'; public.mkdir()
    if kind=='csv':
        path=public/'values.csv'; path.write_text('value\n2\n3\n5\n'); files=[path]
        load="import csv\nvalues=[int(r['value']) for r in csv.DictReader(Path('inputs/values.csv').open())]"
    else:
        import pandas as pd
        files=[public/f'value_{v}.dta' for v in (2,3,5)]
        for path,value in zip(files,(2,3,5)):
            pd.DataFrame({'value':[value]}).to_stata(path,write_index=False,version=118)
        load="import pandas as pd\nvalues=[]\nfor path in sorted(Path('inputs').glob('*.dta')):\n    values.extend(int(x) for x in pd.read_stata(path)['value'])"
    before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    code="import json\nfrom pathlib import Path\n"+load+"\nPath('outputs/analysis_result.json').write_text(json.dumps({'summary':{'sum':sum(values),'count':len(values)}}))\n"
    # CSV path first exercises an authorized read that is not an analysis witness.
    replies=[]
    if kind=='csv':
        replies.append(ModelReply(tool_calls=[ToolCall('inspect','file_inspect',{'path':str(files[0])})],
                                  finish_reason='tool_calls',usage=USAGE))
    replies.append(ModelReply(tool_calls=[ToolCall('compute','code_interpreter',{
        'title':'Public synthetic sum','code':code,'inputs':[str(p) for p in files],
        'timeout_seconds':60})],finish_reason='tool_calls',usage=USAGE))
    service,port,sid,rid=service_for(tmp_path,replies,files)
    try:
        done=service.drive(rid); actions=service.actions(rid)
        assert done['status']=='succeeded', done['state'].get('stop_reason')
        delivery=done['state']['delivery']
        assert delivery['kind']=='execution_bound_json'
        assert json.loads(delivery['content'])=={'sum':10,'count':3}
        assert delivery['execution_witness']['witness_count']>=1
        assert len(port.requests)==len(replies)
        assert all(r['tool_choice']=='required_or_refuse' for r in port.requests)
        assert done['state'].get('execution_continuations',0)==0
        analysis=[a for a in actions if a['tool_name']=='code_interpreter']
        assert len(analysis)==1 and analysis[0]['result']['status']=='ok'
        result=analysis[0]['result']
        backend=result['data'].get('sandbox_backend')
        assert backend in {'macos-sandbox-exec','linux-bubblewrap'}, backend
        assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}==before
        (tmp_path/'canary_summary.json').write_text(json.dumps({
            'passed':True,'kind':kind,'live_provider_calls':0,'benchmark_accessed':False,
            'scripted_decisions':len(port.requests),'required_modes':[r['tool_choice'] for r in port.requests],
            'analysis_actions':len(analysis),'direct_delivery':True,'sum':10,'count':3,
            'backend':backend,'inputs':before,'semantic_validation':'synthetic arithmetic only'}))
    finally:
        service.close()


@pytest.mark.parametrize('refusal',[False,True])
def test_premature_final_or_refusal_has_no_analysis_and_no_retry(tmp_path,refusal):
    public=tmp_path/'public'; public.mkdir(); path=public/'values.csv'; path.write_text('value\n2\n')
    reply=ModelReply(usage=USAGE,refusal='Cannot perform this action.' if refusal else None,
                     content='' if refusal else '{"sum":2,"count":1}')
    service,port,sid,rid=service_for(tmp_path,[reply],[path])
    try:
        done=service.drive(rid)
        assert done['status']=='failed'
        assert done['state']['stop_reason']==('model_refusal' if refusal else 'required_action_missing')
        assert len(port.requests)==1 and port.requests[0]['tool_choice']=='required_or_refuse'
        assert not service.actions(rid)
        assert not done['state'].get('delivery')
        assert done['state'].get('execution_continuations',0)==0
        (tmp_path/'canary_summary.json').write_text(json.dumps({
            'passed':True,'kind':'refusal' if refusal else 'premature_final',
            'live_provider_calls':0,'benchmark_accessed':False,'scripted_decisions':1,
            'analysis_actions':0,'stop_reason':done['state']['stop_reason'],'no_retry':True}))
    finally:
        service.close()
