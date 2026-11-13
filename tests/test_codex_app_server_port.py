"""Native Codex is an isolated inference transport, not the task's tool executor."""
import json
import io
from pathlib import Path

import pytest

from adj_partner.models.base import ProviderFailure
from adj_partner.models.codex_app_server import (BASE_INSTRUCTIONS, CodexAppServerPort,
    decode_envelope, server_command, thread_params, run_native_turn, envelope_schema)


USAGE = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


def host_tool(name="echo", parameters=None):
    return {"type":"function", "function":{"name":name,
            "parameters":parameters if parameters is not None else {
                "type":"object", "properties":{"x":{"type":"number"},
                    "value":{"type":"integer"}, "code":{"type":"string"},
                    "enabled":{"type":"boolean"}, "nested":{"type":"object",
                        "properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":False}},
                "required":[],"additionalProperties":False}}}


def result(**kw):
    return {"model": "gpt-6-luna", "status": "completed", "usage": USAGE,
            "text": '{"content":"42","tool_calls":[]}', "native_activity": False, **kw}


def port(tmp_path, output=None):
    captured = []
    def run(**kwargs):
        captured.append(kwargs)
        if isinstance(output, Exception):
            raise output
        return result() if output is None else output
    return CodexAppServerPort(model="gpt-6-luna", trace_dir=tmp_path, runner=run), captured


def call(p, **kw):
    return p.complete([{"role":"user", "content":"Compute."}], [],
                      max_output_tokens=kw.get("cap", 256), timeout_seconds=10)


def test_success_keeps_known_usage_and_public_request(tmp_path):
    p, calls = port(tmp_path)
    r = call(p)
    assert r.content == "42" and r.usage == USAGE and r.finish_reason == "stop"
    assert calls[0]["request"]["messages"][0]["content"] == "Compute."
    assert p.metadata["output_limit_enforcement"] == "post_response"
    assert p.metadata["native_tools_enabled"] is False
    receipt = json.loads(next(tmp_path.glob('*/receipt.json')).read_text())
    assert receipt["actual_model"] == "gpt-6-luna" and receipt["usage"] == USAGE


@pytest.mark.parametrize("change,code", [
    ({"model":"gpt-5.6-luna"}, "codex_model_mismatch"),
    ({"status":"failed"}, "codex_turn_failed"),
    ({"native_activity":True}, "codex_native_activity"),
    ({"text":"{}"}, "malformed_reply"),
    ({"text":'{"content":"ok","content":"bad","tool_calls":[]}'}, "malformed_reply"),
    ({"usage":{}}, "codex_usage_missing"),
])
def test_failed_or_unverifiable_calls_are_explicit(tmp_path, change, code):
    p, _ = port(tmp_path, result(**change))
    with pytest.raises(ProviderFailure) as exc:
        call(p)
    assert exc.value.code == code
    assert exc.value.cost_unknown == (change.get("usage") == {})


def test_timeout_does_not_fabricate_zero_usage_or_retry(tmp_path):
    p, calls = port(tmp_path, ProviderFailure("codex_timeout", "Timed out"))
    with pytest.raises(ProviderFailure) as exc:
        call(p)
    assert exc.value.cost_unknown and exc.value.usage == {} and len(calls) == 1
    assert json.loads(next(tmp_path.glob('*/receipt.json')).read_text())["status"] == "failed"


def test_output_limit_is_postchecked_and_all_usage_retained(tmp_path):
    p, _ = port(tmp_path)
    r = call(p, cap=10)
    assert r.finish_reason == "length" and r.usage == USAGE
    assert r.content == "" and not r.tool_calls


def test_envelope_only_emits_host_tools_and_rejects_nonfinite():
    tools = [host_tool("calculate")]
    r = decode_envelope('{"content":"","tool_calls":[{"name":"calculate","arguments":{"x":2}}]}', USAGE, tools)
    assert r.finish_reason == "tool_calls" and r.tool_calls[0].arguments == {"x":2}
    with pytest.raises(ProviderFailure):
        decode_envelope('{"content":"","tool_calls":[{"name":"calculate","arguments":{"x":NaN}}]}', USAGE, tools)


def test_native_thread_is_ephemeral_without_environment_or_fallback(tmp_path):
    params = thread_params("gpt-6-luna", tmp_path)
    assert params["environments"] == params["runtimeWorkspaceRoots"] == params["dynamicTools"] == []
    assert params["ephemeral"] and params["allowProviderModelFallback"] is False
    assert params["sandbox"] == "read-only" and params["approvalPolicy"] == "never"


def test_transport_prompt_requires_one_full_envelope_without_commentary_wrappers():
    instruction = thread_params("gpt-6-luna", Path("/tmp"))["developerInstructions"]
    assert "The complete assistant message must be exactly one JSON envelope" in BASE_INSTRUCTIONS
    assert "Do not emit commentary" in BASE_INSTRUCTIONS
    assert "one JSON envelope as the complete assistant message" in instruction
    assert "Do not emit commentary" in instruction


def test_transport_contract_version_records_envelope_prompt_revision(tmp_path):
    p, _ = port(tmp_path)
    assert p.metadata["transport_version"] == "0.7"


def test_all_configured_mcp_servers_disabled_without_auth_read(tmp_path):
    config = tmp_path / 'config.toml'
    config.write_text('[mcp_servers.example]\ncommand="unused"\n')
    argv = server_command('/cli', config)
    assert 'mcp_servers.example.enabled=false' in argv
    for feature in ['shell_tool','apps','plugins','memories','browser_use','multi_agent']:
        assert argv[argv.index(feature)-1] == '--disable'
    assert '--ignore-rules' not in argv


def test_invalid_public_input_does_not_start_a_call(tmp_path):
    p, calls = port(tmp_path)
    with pytest.raises(ProviderFailure):
        p.complete([], [], max_output_tokens=0, timeout_seconds=1)
    assert calls == []


def test_no_tool_contract_rejects_unexposed_host_request(tmp_path):
    output=result(text=json.dumps({'content':'','tool_calls':[{'name':'web__run','arguments':{}}]}))
    p,calls=port(tmp_path,output)
    with pytest.raises(ProviderFailure) as exc:
        call(p)
    assert exc.value.code=='codex_unexposed_tool'
    assert exc.value.usage==USAGE and exc.value.cost_unknown is False
    assert len(calls)==1


def test_only_exposed_host_names_can_be_delivered(tmp_path):
    output=result(text=json.dumps({'content':'','tool_calls':[{'name':'unlisted','arguments':{}}]}))
    p,_=port(tmp_path,output)
    with pytest.raises(ProviderFailure) as exc:
        p.complete([{'role':'user','content':'Compute.'}],
                   [host_tool()],
                   max_output_tokens=256,timeout_seconds=10)
    assert exc.value.code=='codex_unexposed_tool'


def test_request_output_schema_has_exact_host_names_and_no_shared_mutation():
    none=envelope_schema([])
    assert none['properties']['tool_calls']['maxItems']==0
    tools=[host_tool('echo', {'type':'object','properties':{'value':{'type':'integer'}},
                             'required':['value'],'additionalProperties':False})]
    before=json.loads(json.dumps(tools))
    one=envelope_schema(tools)
    branch=one['properties']['tool_calls']['items']['anyOf'][0]
    assert branch['properties']['name']['enum']==['echo']
    assert branch['properties']['arguments']==tools[0]['function']['parameters']
    assert one['properties']['tool_calls']['maxItems']==128
    branch['properties']['arguments']['properties']['value']['type']='string'
    assert tools==before
    assert envelope_schema([])['properties']['tool_calls']['maxItems']==0


def test_exposed_host_request_still_delivered(tmp_path):
    p,_=port(tmp_path,result(text=json.dumps({'content':'','tool_calls':[{'name':'echo','arguments':{'x':1}}]})))
    r=p.complete([{'role':'user','content':'Compute.'}],
        [host_tool()],
        max_output_tokens=256,timeout_seconds=10)
    assert r.tool_calls[0].name=='echo' and r.tool_calls[0].arguments=={'x':1}


def fake_process(monkeypatch, tmp_path, terminal_items, *, thread_changes=None, extra_event=None):
    thread = {"model":"gpt-6-luna","modelProvider":"openai","thread":{"id":"thread1"},
              "runtimeWorkspaceRoots":[],"instructionSources":[],
              "sandbox":{"type":"readOnly","networkAccess":False},
              "approvalPolicy":"never","multiAgentMode":"explicitRequestOnly"}
    thread.update(thread_changes or {})
    events = [{"id":1,"result":{"userAgent":"test"}}, {"id":2,"result":thread},
              {"id":3,"result":{"turn":{"id":"turn1"}}}]
    if extra_event is not None: events.append(extra_event)
    events += [{"method":"thread/tokenUsage/updated","params":{"threadId":"thread1","turnId":"turn1",
               "tokenUsage":{"total":{"inputTokens":100,"outputTokens":20,"totalTokens":120}}}},
               {"method":"turn/completed","params":{"threadId":"thread1","turn":{"id":"turn1", "status":"completed","items":terminal_items}}}]
    class Process:
        def __init__(self, *a, **kw):
            self.stdout=io.StringIO(''.join(json.dumps(e)+'\n' for e in events)); self.stdin=io.StringIO()
        def terminate(self): pass
        def kill(self): pass
        def wait(self, **kw): return 0
    monkeypatch.setattr('adj_partner.models.codex_app_server.subprocess.Popen',Process)
    return dict(model='gpt-6-luna',effort='low',request={'messages':[], 'tools':[host_tool()], 'max_output_tokens':100,'timeout_seconds':5},
                trace_path=tmp_path/'events.jsonl',cli='/fake',config_path=tmp_path/'absent')


FINAL = {"type":"agentMessage","id":"msg1","phase":"final_answer", "text":'{"content":"42","tool_calls":[]}'}


def test_native_turn_parser_handles_real_lifecycle(monkeypatch,tmp_path):
    kwargs=fake_process(monkeypatch,tmp_path,[FINAL])
    r=run_native_turn(**kwargs)
    assert r['usage']==USAGE and r['text']==FINAL['text']


def test_host_request_in_commentary_precedes_unobserved_final(monkeypatch,tmp_path):
    planned={'type':'agentMessage','id':'planned','phase':'commentary',
             'text':json.dumps({'content':'','tool_calls':[{'name':'echo','arguments':{'value':7}}]})}
    kwargs=fake_process(monkeypatch,tmp_path,[planned,FINAL])
    r=run_native_turn(**kwargs)
    reply=decode_envelope(r['text'],r['usage'],kwargs['request']['tools'])
    assert reply.tool_calls[0].name=='echo'
    assert r['selection']=='first_host_tool_request'
    assert r['selected_message_id']=='planned'
    assert r['usage']==USAGE


@pytest.mark.parametrize('suffix', [
    ' \n</commentary>.',
    '"} RD? Need exact response format schema outer only. Need await tool result via next prompt presumably.',
])
def test_malformed_commentary_envelope_is_not_repaired_or_executed(monkeypatch,tmp_path,suffix):
    wrapped = {'type':'agentMessage','id':'wrapped','phase':'commentary',
               'text':json.dumps({'content':'','tool_calls':[{'name':'echo','arguments':{'value':7}}]})
                       + suffix}
    kwargs=fake_process(monkeypatch,tmp_path,[wrapped,FINAL])
    r=run_native_turn(**kwargs)
    assert r['text']==FINAL['text'] and r['selection']=='final_answer'
    assert r['ignored_host_request_count']==0


def test_tool_argument_escaping_survives_transport_projection():
    envelope=json.dumps({'content':'','tool_calls':[{
        'name':'calculate','arguments':{'code':'a\nb "quoted"','enabled':True}
    }]})
    r=decode_envelope(envelope,USAGE,[host_tool('calculate')])
    assert r.tool_calls[0].arguments == {'code':'a\nb "quoted"','enabled':True}


def test_multiple_host_requests_return_first_batch_only(monkeypatch,tmp_path):
    def item(i):
        return {'type':'agentMessage','id':str(i),'phase':'commentary',
                'text':json.dumps({'content':'','tool_calls':[{'name':'echo','arguments':{'value':i}}]})}
    kwargs=fake_process(monkeypatch,tmp_path,[item(1),item(2),FINAL])
    r=run_native_turn(**kwargs)
    reply=decode_envelope(r['text'],r['usage'],kwargs['request']['tools'])
    assert len(reply.tool_calls)==1 and reply.tool_calls[0].arguments=={'value':1}
    assert r['ignored_host_request_count']==1


@pytest.mark.parametrize('text',['I will call echo next.', '{"content":"","tool_calls":[',
    '{"content":"","tool_calls":[{"name":"echo","arguments":"{bad}"}]}'])
def test_incomplete_or_narrative_commentary_is_not_executed(monkeypatch,tmp_path,text):
    commentary={'type':'agentMessage','id':'commentary','phase':'commentary','text':text}
    kwargs=fake_process(monkeypatch,tmp_path,[commentary,FINAL])
    r=run_native_turn(**kwargs)
    assert r['text']==FINAL['text'] and r['selection']=='final_answer'


def test_started_only_host_request_is_not_executed(monkeypatch,tmp_path):
    partial={'method':'item/started','params':{'threadId':'thread1','turnId':'turn1','item':{
        'type':'agentMessage','id':'partial','phase':'commentary','text':json.dumps({
            'content':'','tool_calls':[{'name':'echo','arguments':{}}]})}}}
    kwargs=fake_process(monkeypatch,tmp_path,[FINAL],extra_event=partial)
    r=run_native_turn(**kwargs)
    assert r['text']==FINAL['text'] and r['selection']=='final_answer'


def test_terminal_items_are_checked_even_without_notifications(monkeypatch,tmp_path):
    kwargs=fake_process(monkeypatch,tmp_path,[{'type':'commandExecution','id':'x','command':'PRIVATE_SENTINEL'},FINAL])
    with pytest.raises(ProviderFailure) as e:
        run_native_turn(**kwargs)
    assert e.value.code=='codex_native_activity'
    assert 'PRIVATE_SENTINEL' not in (tmp_path/'events.jsonl').read_text()


@pytest.mark.parametrize('change',[{'approvalPolicy':'on-request'},{'modelProvider':'other'},{'multiAgentMode':'proactive'},
                                   {'sandbox':{'type':'readOnly','networkAccess':True}}])
def test_returned_isolation_fields_fail_closed(monkeypatch,tmp_path,change):
    kwargs=fake_process(monkeypatch,tmp_path,[FINAL],thread_changes=change)
    with pytest.raises(ProviderFailure) as e: run_native_turn(**kwargs)
    assert e.value.code=='codex_isolation' and not e.value.cost_unknown


def test_nonobject_protocol_event_is_safe_failure(monkeypatch,tmp_path):
    kwargs=fake_process(monkeypatch,tmp_path,[FINAL],extra_event=[1])
    with pytest.raises(ProviderFailure) as e: run_native_turn(**kwargs)
    assert e.value.code=='codex_protocol'


def test_transcripts_and_receipts_are_private_files(tmp_path):
    p,_=port(tmp_path); call(p)
    folder=next(tmp_path.iterdir())
    assert folder.stat().st_mode & 0o077 == 0
    for f in folder.iterdir(): assert f.stat().st_mode & 0o077 == 0


def test_structured_arguments_are_bound_to_the_named_tool_schema():
    from adj_partner.tools.catalog import validator
    tools = [host_tool('integer_tool', {'type':'object', 'properties':{'value':{'type':'integer'}},
        'required':['value'], 'additionalProperties':False}),
        host_tool('text_tool', {'type':'object', 'properties':{'text':{'type':'string','minLength':1}},
        'required':['text'], 'additionalProperties':False})]
    schema = envelope_schema(tools)
    valid = {'content':'','tool_calls':[{'name':'integer_tool','arguments':{'value':2}}]}
    assert validator(schema).is_valid(valid)
    decoded = decode_envelope(json.dumps(valid), USAGE, tools)
    assert decoded.tool_calls[0].arguments == {'value':2}
    for name, arguments in [('integer_tool',{'text':'wrong schema'}), ('text_tool',{'value':2}),
                            ('integer_tool',{'value':True}), ('integer_tool',{'value':2,'extra':0}),
                            ('text_tool',{'text':''})]:
        envelope = {'content':'','tool_calls':[{'name':name,'arguments':arguments}]}
        assert not validator(schema).is_valid(envelope)
        with pytest.raises(ProviderFailure) as exc:
            decode_envelope(json.dumps(envelope), USAGE, tools)
        assert exc.value.code == 'malformed_reply' and exc.value.usage == USAGE


@pytest.mark.parametrize('arguments', ['{"value":2}', '{"value":2', [], None, True, 2])
def test_string_or_wrong_shape_arguments_never_repaired(arguments):
    envelope = {'content':'','tool_calls':[{'name':'echo','arguments':arguments}]}
    with pytest.raises(ProviderFailure) as exc:
        decode_envelope(json.dumps(envelope), USAGE, [host_tool()])
    assert exc.value.code == 'malformed_reply' and not exc.value.cost_unknown


@pytest.mark.parametrize('raw', [
    '{"content":"","tool_calls":[{"name":"echo","arguments":{"value":1,"value":2}}]}',
    '{"content":"","tool_calls":[{"name":"echo","arguments":{"nested":{"x":1,"x":2}}}]}',
    '{"content":"","tool_calls":[{"name":"echo","arguments":{"x":Infinity}}]}',
    '{"content":"","tool_calls":[{"name":"echo","arguments":{"x":1e400}}]}',
    '{"content":null,"tool_calls":[]}',
    '{"content":"","tool_calls":[{"name":"echo","arguments":{},"extra":1}]}',
])
def test_object_protocol_remains_strict_for_duplicate_nonfinite_and_wrong_fields(raw):
    with pytest.raises(ProviderFailure) as exc:
        decode_envelope(raw, USAGE, [host_tool()])
    assert exc.value.code == 'malformed_reply' and exc.value.usage == USAGE


@pytest.mark.parametrize('kind', ['arguments', 'calls', 'envelope', 'reply'])
def test_structured_transport_preserves_existing_size_bounds(kind):
    body = {'content':'', 'tool_calls':[]}
    if kind == 'arguments': body['tool_calls']=[{'name':'echo','arguments':{'code':'x'*(1024*1024)}}]
    elif kind == 'calls': body['tool_calls']=[{'name':'echo','arguments':{}}]*129
    elif kind == 'envelope': body['content']='x'*(8*1024*1024)
    else: body['content']='x'*(4*1024*1024)
    with pytest.raises(ProviderFailure) as exc:
        decode_envelope(json.dumps(body), USAGE, [host_tool()])
    assert exc.value.code == 'malformed_reply' and exc.value.usage == USAGE


def test_invalid_schema_or_duplicate_name_fails_before_inference(tmp_path):
    cases = [[host_tool('echo',{'type':'not_a_json_schema_type'})],
             [host_tool('echo'),host_tool('echo')],
             [host_tool('echo',{'type':'object','$ref':'https://invalid.example/schema'})],
             [host_tool('echo',{'type':'object','$defs':{'v':{'type':'integer'}},
                               'properties':{'value':{'$ref':'#/$defs/v'}}})]]
    for index, tools in enumerate(cases):
        p, calls = port(tmp_path/str(index))
        with pytest.raises(ProviderFailure) as exc:
            p.complete([{'role':'user','content':'Compute'}], tools, max_output_tokens=256, timeout_seconds=10)
        assert exc.value.code == 'invalid_request'
        assert calls == []
        assert not list((tmp_path/str(index)).rglob('receipt.json'))


def test_malformed_structured_call_charges_usage_once_without_retry(tmp_path):
    p,calls=port(tmp_path,result(text='{"content":"","tool_calls":[{"name":"echo","arguments":"{\\"value\\":2"}]}'))
    with pytest.raises(ProviderFailure):
        p.complete([{'role':'user','content':'Compute'}],[host_tool()],max_output_tokens=256,timeout_seconds=10)
    assert len(calls)==1
    receipt=json.loads(next(tmp_path.glob('*/receipt.json')).read_text())
    assert receipt['status']=='failed' and receipt['usage']==USAGE and receipt['cost_unknown'] is False
    assert receipt['transport_version']=='0.7'


def test_transport_instructions_require_object_arguments_not_nested_json_strings():
    instruction=thread_params('gpt-6-luna',Path('/tmp'))['developerInstructions']
    assert 'arguments (a JSON object)' in BASE_INSTRUCTIONS
    assert 'JSON-string arguments' not in BASE_INSTRUCTIONS
    assert 'argument strings' not in instruction
    assert 'structured JSON objects' in instruction


def test_invalid_named_arguments_in_commentary_do_not_preempt_final(monkeypatch,tmp_path):
    invalid={'type':'agentMessage','id':'invalid','phase':'commentary',
             'text':json.dumps({'content':'','tool_calls':[{'name':'echo','arguments':{'value':'not an integer'}}]})}
    kwargs=fake_process(monkeypatch,tmp_path,[invalid,FINAL])
    kwargs['request']['tools']=[host_tool('echo',{'type':'object','properties':{'value':{'type':'integer'}},
                                               'required':['value'],'additionalProperties':False})]
    r=run_native_turn(**kwargs)
    assert r['selection']=='final_answer' and r['selected_message_id']=='msg1'
    assert r['ignored_host_request_count']==0


def test_real_analysis_schema_preserves_multiline_code_and_rejects_extra_field():
    from adj_partner.analysis.capabilities import AnalysisCapabilities
    from adj_partner.tools.catalog import validator
    tools=[spec.model_schema() for spec in AnalysisCapabilities.definitions() if spec.name=='code_interpreter']
    code='import json\nprint("quoted \\\" text, Unicode: α")\n'
    body={'content':'','tool_calls':[{'name':'code_interpreter','arguments':{
        'title':'Structured parameters','code':code,'inputs':['/tmp/public.json']}}]}
    assert validator(envelope_schema(tools)).is_valid(body)
    reply=decode_envelope(json.dumps(body),USAGE,tools)
    assert reply.tool_calls[0].arguments['code']==code
    body['tool_calls'][0]['arguments']['hidden_retry']=True
    with pytest.raises(ProviderFailure) as exc: decode_envelope(json.dumps(body),USAGE,tools)
    assert exc.value.code=='malformed_reply'


def test_mixed_batch_is_rejected_whole_without_delivering_valid_subset():
    tools=[host_tool('echo',{'type':'object','properties':{'value':{'type':'integer'}},
                             'required':['value'],'additionalProperties':False})]
    body={'content':'','tool_calls':[{'name':'echo','arguments':{'value':1}},
                                     {'name':'echo','arguments':{'value':'bad'}}]}
    with pytest.raises(ProviderFailure) as exc: decode_envelope(json.dumps(body),USAGE,tools)
    assert exc.value.code=='malformed_reply'


@pytest.mark.parametrize('schema', [
    {'type':'object','$id':'urn:example:isolated'},
    {'type':'object','$anchor':'local'},
    {'type':'array','items':{'type':'integer'}},
])
def test_unsupported_embedded_schema_scope_is_rejected_before_dispatch(tmp_path,schema):
    p,calls=port(tmp_path)
    with pytest.raises(ProviderFailure) as exc:
        p.complete([{'role':'user','content':'Compute'}],[host_tool('echo',schema)],
                   max_output_tokens=256,timeout_seconds=10)
    assert exc.value.code=='invalid_request' and not exc.value.cost_unknown
    assert not calls and not list(tmp_path.rglob('receipt.json'))


def test_native_entry_rejects_unsupported_schema_before_starting_subprocess(monkeypatch,tmp_path):
    def forbidden(*args,**kwargs):
        pytest.fail('invalid schema must not start an app-server subprocess')
    monkeypatch.setattr('adj_partner.models.codex_app_server.subprocess.Popen',forbidden)
    with pytest.raises(ProviderFailure) as exc:
        run_native_turn(model='gpt-6-luna',effort='low',
            request={'messages':[], 'tools':[host_tool(parameters={'type':'object'})],
                     'max_output_tokens':256,'timeout_seconds':10},
            trace_path=tmp_path/'events.jsonl',cli='/unused',config_path=tmp_path/'absent')
    assert exc.value.code=='invalid_request' and not exc.value.cost_unknown
    assert not (tmp_path/'events.jsonl').exists()


@pytest.mark.parametrize('method', ['error','turn/completed'])
def test_error_diagnostics_keep_only_fixed_labels_and_http_status(method):
    from adj_partner.models.codex_app_server import _event_snapshot
    error={'message':'invalid_json_schema required additionalProperties SECRET_TOKEN https://private.example',
           'codexErrorInfo':{'httpConnectionFailed':{'httpStatusCode':400,'raw':'PRIVATE_BODY'}},
           'additionalDetails':'PRIVATE_DETAILS','authorization':'PRIVATE_AUTH'}
    params={'error':error} if method=='error' else {'turn':{'id':'turn1','status':'failed','items':[],'error':error}}
    snapshot=_event_snapshot({'method':method,'params':params})
    assert snapshot['diagnostics']=={'codex_error_type':'httpConnectionFailed','http_status':400,
        'schema_error':True,'required_keyword':True,'additional_properties_keyword':True}
    serialized=json.dumps(snapshot)
    for secret in ['SECRET_TOKEN','private.example','PRIVATE_BODY','PRIVATE_DETAILS','PRIVATE_AUTH']:
        assert secret not in serialized
    assert 'usage' not in snapshot and 'cost_unknown' not in snapshot


@pytest.mark.parametrize('error', [None, [], 'SECRET', {'codexErrorInfo':'SECRET'},
    {'message':{'unexpected':'SECRET'},'codexErrorInfo':{'httpConnectionFailed':{'httpStatusCode':True}}},
    {'message':'ordinary error SECRET','codexErrorInfo':{'unknown':'SECRET'}}])
def test_unexpected_error_diagnostics_cannot_leak_or_break_logging(error):
    from adj_partner.models.codex_app_server import _event_snapshot
    snapshot=_event_snapshot({'method':'error','params':{'error':error}})
    assert 'SECRET' not in json.dumps(snapshot)
    assert snapshot.get('diagnostics',{}) in ({},{'codex_error_type':'httpConnectionFailed'})
