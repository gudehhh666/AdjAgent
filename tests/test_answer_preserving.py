"""Draft preflight accounting and real service handoff; no network/model APIs."""
import importlib.util
import json
from pathlib import Path
import pytest
from adj_partner.contracts import Budget, ModelReply, ToolCall
from adj_partner.models import ScriptedPort

PROJECT = Path(__file__).resolve().parents[1]

def api():
    assert importlib.util.find_spec('adj_partner.answer_preserving') is not None, 'answer-preserving preflight module is missing'
    from adj_partner.answer_preserving import prepare_draft, DraftPreparationError, ANSWER_PRESERVING_GUIDANCE
    return prepare_draft, DraftPreparationError, ANSWER_PRESERVING_GUIDANCE

def ticks(start=10, end=12):
    times=iter([start,end]);return lambda:next(times)

def test_handoff_accounts_actual_usage_and_preserves_json_boundary():
    prepare,_,_=api();port=ScriptedPort([ModelReply(content='Draft with "quotes"\nand all parts.',usage={'input_tokens':120,'output_tokens':35,'total_tokens':155})])
    budget=Budget(max_steps=12,max_input_tokens=100000,max_output_tokens=12000,wall_time_seconds=420)
    result=prepare(port,'Compare A and B.',budget,system='Exact baseline system',clock=ticks())
    assert result['remaining_budget']=={'max_steps':11,'max_input_tokens':99880,'max_output_tokens':11965,'wall_time_seconds':418}
    assert result['goal'].startswith('Compare A and B.\n\n')
    assert 'not the requested output format' in result['goal']
    payload=result['goal'].split('\nPRELIMINARY_ANSWER_DATA_JSON\n',1)[1].split('\nEND_PRELIMINARY_ANSWER_DATA_JSON',1)[0]
    assert json.loads(payload)=={'unverified_draft':result['draft']}
    assert result['goal'].endswith('Do not return the preliminary JSON or its field names.')
    assert result['accounting']['usage_complete'] is True
    assert port.requests[0]['messages']==[{'role':'system','content':'Exact baseline system'},{'role':'user','content':'Compare A and B.'}]
    assert port.requests[0]['tools']==[]
    assert result['request_limits']=={'max_output_tokens':3200,'timeout_seconds':120}
    assert budget.max_steps==12

@pytest.mark.parametrize('change',[{'max_steps':1},{'max_output_tokens':1},{'max_input_tokens':1}])
def test_insufficient_prior_budget_rejects_without_call(change):
    prepare,Error,_=api();port=ScriptedPort([]);b=Budget(**change)
    with pytest.raises(Error) as got:prepare(port,'question',b,system='baseline',clock=ticks())
    assert got.value.code=='insufficient_budget' and not got.value.record['attempted'] and port.requests==[]

def test_small_budget_caps_draft_and_unknown_usage_reserves_full_allowance():
    prepare,_,_=api();port=ScriptedPort([ModelReply(content='draft',usage={})]);b=Budget(max_steps=3,max_input_tokens=2000,max_output_tokens=100,wall_time_seconds=20)
    r=prepare(port,'Q',b,system='S',clock=ticks())
    assert r['request_limits']=={'max_output_tokens':50,'timeout_seconds':10}
    assert r['usage']=={} and r['accounting']['usage_complete'] is False
    assert r['accounting']['accounted_output_tokens']==50
    assert r['remaining_budget']['max_input_tokens']==2000-r['accounting']['estimated_input_tokens']
    assert r['remaining_budget']['max_output_tokens']==50

def test_length_truncated_draft_is_accepted_and_marked():
    prepare,_,_=api();port=ScriptedPort([ModelReply(content='partial',finish_reason='length')])
    r=prepare(port,'Q',Budget(),system='S',clock=ticks())
    assert r['draft']=='partial' and r['truncated'] is True


def test_fallback_delivery_prefers_draft_then_one_direct_call():
    from adj_partner.answer_preserving import fallback_delivery
    port=ScriptedPort([])
    assert fallback_delivery(port,'Q',system='S',draft='kept')==('kept','draft_fallback') and port.requests==[]
    port=ScriptedPort([ModelReply(content='direct answer')])
    assert fallback_delivery(port,'Q',system='S')==('direct answer','direct_fallback') and len(port.requests)==1
    assert fallback_delivery(ScriptedPort([]),'Q',system='S',timeout_seconds=1)==('','no_budget_for_fallback')


@pytest.mark.parametrize('reply',[ModelReply(content=''),ModelReply(content='draft',tool_calls=[ToolCall('x','web_search',{})]),ModelReply(content='draft',refusal='no')])
def test_partial_or_refused_draft_rejected_with_cost_and_partial_preserved(reply):
    prepare,Error,_=api();port=ScriptedPort([reply])
    with pytest.raises(Error) as got:prepare(port,'Q',Budget(),system='S',clock=ticks())
    assert got.value.code=='invalid_draft'
    assert got.value.record['partial_draft']==reply.content
    assert got.value.record['remaining_budget']['max_steps']==Budget().max_steps-1
    assert len(port.requests)==1

@pytest.mark.parametrize('reply,clock',[ (ModelReply(content='valid',usage={'input_tokens':1000000,'output_tokens':1}),ticks()),(ModelReply(content='valid'),ticks(end=1000))])
def test_observed_budget_overrun_is_not_returned_as_success(reply,clock):
    prepare,Error,_=api();port=ScriptedPort([reply])
    with pytest.raises(Error) as got:prepare(port,'Q',Budget(),system='S',clock=clock)
    assert got.value.code=='budget_exhausted' and got.value.record['partial_draft']=='valid'


def test_provider_error_retains_attempt_and_conservative_charge_no_retry():
    prepare,Error,_=api()
    class Fails:
        calls=0
        def complete(self,*args,**kwargs):self.calls+=1;raise RuntimeError('provider failed')
    port=Fails()
    with pytest.raises(Error) as got:prepare(port,'Q',Budget(),system='S',clock=ticks())
    assert got.value.code=='provider_failure' and got.value.record['attempted']
    assert got.value.record['accounting']['accounted_output_tokens']>0 and port.calls==1


def test_scripted_port_agent_service_sees_draft_and_reduced_budget(tmp_path):
    prepare,_,guidance=api()
    from adj_partner.application import AgentService
    draft='First compare A and B, then discuss limitations.'
    port=ScriptedPort([ModelReply(content=draft,usage={'input_tokens':20,'output_tokens':10}),ModelReply(content='Both parts retained; uncertainty explicit.',usage={'input_tokens':200,'output_tokens':20})])
    initial=Budget(max_steps=4,max_input_tokens=100000,max_output_tokens=10000,wall_time_seconds=60)
    ready=prepare(port,'Compare A and B; discuss limitations.',initial,system='Baseline',clock=ticks())
    service=AgentService(tmp_path/'state',project_root=PROJECT,model=port)
    try:
        session=service.create_session(instructions=guidance,budget=ready['remaining_budget'],allowed_tools=[])
        rid=service.submit(session['id'],ready['goal'])['id'];result=service.drive(rid)
        assert result['status']=='succeeded'
        assert session['config']['budget']['max_steps']==3
        assert session['config']['budget']['max_input_tokens']==99980
        assert session['config']['budget']['max_output_tokens']==9990
        assert session['config']['budget']['wall_time_seconds']==58
        assert draft in json.dumps(port.requests[1]['messages'],ensure_ascii=False)
        assert any(guidance in m['content'] for m in port.requests[1]['messages'] if m['role']=='system')
    finally:service.close()
