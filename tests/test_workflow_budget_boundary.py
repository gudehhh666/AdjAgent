import json
import pytest
from adj_partner.application import AgentService
from adj_partner.contracts import Budget,ModelReply
from adj_partner.models.base import ScriptedPort
from adj_partner.orchestration.contracts import TaskSpec
from adj_partner.orchestration.controller import AgentFramework


@pytest.mark.parametrize('interrupt_after_receipt',[False,True])
def test_last_reserved_request_can_finish_or_resume_without_extra_budget(tmp_path,interrupt_after_receipt):
    p=ScriptedPort([ModelReply(content='Draft',usage={'input_tokens':4,'output_tokens':2}),ModelReply(content=json.dumps({'answer':'Qualified answer','sources':[],'review':'Checked','uncertainty':'Limited'}),usage={'input_tokens':4,'output_tokens':2})])
    s=AgentService(tmp_path/'state',model=p);f=AgentFramework(s)
    spec=TaskSpec('qa','Q',model='offline-script',allow_network=False,budget=Budget(max_steps=2,max_input_tokens=100000,max_output_tokens=20000,wall_time_seconds=900).to_dict())
    t=f.submit(spec)
    if interrupt_after_receipt:
        original=f.invocations.complete
        def crash(*args,**kwargs):
            original(*args,**kwargs)
            raise KeyboardInterrupt('after durable final receipt, before workflow delivery')
        f.invocations.complete=crash
        with pytest.raises(KeyboardInterrupt):f.run(t['fingerprint'])
        f.invocations.complete=original
    result=f.run(t['fingerprint'])
    assert result['execution_status']=='succeeded',result.get('error')
    assert result['delivery']['content']=='Qualified answer'
    assert result['usage']['requests']==2
    assert len(p.requests)==2
    assert f.run(t['fingerprint'])==result
    assert len(p.requests)==2
    s.close()
