"""Offline equivalence and strict-subset checks; no provider dispatch."""
from copy import deepcopy
import importlib.util
import itertools
import json
from pathlib import Path

import pytest

from adj_partner.models.base import ProviderFailure
from adj_partner.models.codex_app_server import CodexAppServerPort, decode_envelope, envelope_schema
from adj_partner.tools.catalog import validator


def closed(properties, required=()):
    return {'type':'object', 'properties':properties, 'required':list(required), 'additionalProperties':False}


def tool(parameters, name='compute'):
    return {'type':'function', 'function':{'name':name, 'parameters':parameters}}


def arguments_schema(parameters):
    return envelope_schema([tool(parameters)])['properties']['tool_calls']['items']['anyOf'][0]['properties']['arguments']


def assert_strict(schema):
    """Independent recursive assertion of provider-required object shape."""
    if schema.get('type') == 'object':
        assert schema.get('additionalProperties') is False
        assert set(schema.get('required', ())) == set(schema.get('properties', {}))
        for child in schema.get('properties', {}).values():
            assert_strict(child)
    if 'items' in schema:
        assert_strict(schema['items'])
    for child in schema.get('anyOf', []):
        assert_strict(child)


def test_optional_fields_are_exact_keyset_branches_not_required_null_defaults():
    original=closed({'x':{'type':'integer'}, 'note':{'type':'string'}}, ['x'])
    compiled=arguments_schema(original)
    assert_strict(compiled)
    assert {tuple(branch['properties']) for branch in compiled['anyOf']} == {('x',), ('x','note')}
    assert validator(compiled).is_valid({'x':1})
    assert not validator(compiled).is_valid({'x':1,'note':None})


def test_no_tools_schema_is_strict_even_for_unreachable_array_items():
    compiled=envelope_schema([])
    assert_strict(compiled)
    assert compiled['properties']['tool_calls']['maxItems'] == 0
    assert validator(compiled).is_valid({'content':'answer','tool_calls':[]})
    assert not validator(compiled).is_valid({'content':'','tool_calls':[{'name':'compute','arguments':{}}]})


def test_small_nested_nullable_schema_has_same_acceptance_set_exhaustively():
    schema=closed({'x':{'type':'integer'}, 'note':{'type':['string','null']},
        'rows':{'type':'array','items':closed({'flag':{'type':'boolean'}}),'maxItems':2}}, ['x'])
    before=deepcopy(schema); compiled=arguments_schema(schema)
    assert_strict(compiled)
    absent=object()
    values=[absent,None,False,0,1,'',[],[{}],[{'flag':True}],[{'flag':None}],[{}, {}, {}]]
    host,wire=validator(schema),validator(compiled)
    for items in itertools.product(values, repeat=3):
        instance={key:value for key,value in zip(('x','note','rows'),items) if value is not absent}
        assert wire.is_valid(instance) == host.is_valid(instance), instance
    assert not wire.is_valid({'x':1,'unknown':0})
    assert schema == before


@pytest.mark.parametrize('arguments', [{'x':1}, {'x':1,'note':None}, {'x':1,'note':''}])
def test_optional_nullable_roundtrip_preserves_omission_and_explicit_null(arguments):
    tools=[tool(closed({'x':{'type':'integer'}, 'note':{'type':['string','null']}}, ['x']))]
    compiled=envelope_schema(tools)
    body={'content':'','tool_calls':[{'name':'compute','arguments':deepcopy(arguments)}]}
    assert validator(compiled).is_valid(body)
    reply=decode_envelope(json.dumps(body), {'input_tokens':1,'output_tokens':1,'total_tokens':2}, tools)
    assert reply.tool_calls[0].arguments == arguments


def test_nullable_containers_and_untyped_scalar_enum_remain_equivalent():
    schema=closed({'object':{**closed({'x':{'enum':[1,2]}}), 'type':['object','null']},
        'array':{'type':['array','null'],'items':{'enum':['a','b',None]}},
        'choice':{'enum':['pending','done']}})
    compiled=arguments_schema(schema);assert_strict(compiled)
    host,wire=validator(schema),validator(compiled)
    for obj,arr,choice in itertools.product([None,{}, {'x':1.0},{'x':True}],
            [None,[],['a',None],['bad']], ['pending','done',0,None]):
        sample={'object':obj,'array':arr,'choice':choice}
        assert host.is_valid(sample)==wire.is_valid(sample),sample


@pytest.mark.parametrize('bad', [
    {'type':'object'},
    closed({'dynamic':{'type':'object'}}),
    {**closed({}),'patternProperties':{'x':{'type':'integer'}}},
    {**closed({}),'allOf':[closed({})]},
    {**closed({}),'anyOf':[closed({})]},
    {**closed({}),'$id':'urn:local'},
    closed({'ref':{'$ref':'#/other'}}),
    closed({'anything':{}}),
    closed({'date':{'type':'string','format':'date'}}),
    closed({'defaulted':{'type':'integer','default':3}}),
    closed({'tuple':{'type':'array','prefixItems':[{'type':'integer'}]}}),
    {**closed({}),'required':['undeclared']},
])
def test_unsupported_schema_is_rejected_before_call_or_receipt(tmp_path,bad):
    calls=[]
    port=CodexAppServerPort(model='gpt-6-luna',trace_dir=tmp_path,
        runner=lambda **kw:calls.append(kw))
    with pytest.raises(ProviderFailure) as exc:
        port.complete([{'role':'user','content':'Compute.'}],[tool(bad)],max_output_tokens=256,timeout_seconds=10)
    assert exc.value.code=='invalid_request' and exc.value.cost_unknown is False
    assert not calls and not list(tmp_path.rglob('receipt.json'))


@pytest.mark.parametrize('kind', ['branches','properties','depth','bytes','enum','total_branches',
    'expanded_properties','expanded_bytes','enum_string','name_chars'])
def test_expansion_budgets_fail_before_dispatch(tmp_path,kind):
    if kind in {'branches','total_branches'}:
        count=9 if kind=='branches' else 8
        schema=closed({f'p{i}':{'type':'integer'} for i in range(count)})
    elif kind=='properties':
        properties={f'p{i}':{'type':'integer'} for i in range(4097)}
        schema=closed(properties,properties)
    elif kind=='depth':
        schema={'type':'integer'}
        for _ in range(11): schema=closed({'nested':schema},['nested'])
    elif kind=='expanded_properties':
        properties={f'r{i}':{'type':'integer'} for i in range(200)}
        schema=closed({**properties,**{f'o{i}':{'type':'integer'} for i in range(5)}},properties)
    elif kind=='expanded_bytes':
        schema=closed({'required':{'type':'integer','description':'x'*20_000},
            **{f'o{i}':{'type':'integer'} for i in range(7)}},['required'])
    elif kind=='enum_string':
        schema=closed({'choice':{'enum':[str(i)+'x'*60 for i in range(251)]}},['choice'])
    elif kind=='name_chars':
        properties={'x'*120_001:{'type':'integer'}}
        schema=closed(properties,properties)
    elif kind=='bytes':
        schema={**closed({}), 'description':'x'*(1024*1024)}
    else:
        schema=closed({'choice':{'type':'integer','enum':list(range(1001))}}, ['choice'])
    tools=[tool(schema)]
    if kind=='total_branches': tools.append(tool(deepcopy(schema),'other'))
    calls=[]
    port=CodexAppServerPort(model='gpt-6-luna',trace_dir=tmp_path,runner=lambda **kw:calls.append(kw))
    with pytest.raises(ProviderFailure) as exc:
        port.complete([{'role':'user','content':'Compute.'}],tools,max_output_tokens=256,timeout_seconds=10)
    assert exc.value.code=='invalid_request' and exc.value.cost_unknown is False
    assert not calls and not list(tmp_path.rglob('receipt.json'))


def test_supported_scalar_constraints_are_preserved_in_every_keyset():
    schema=closed({'score':{'type':['number','null'],'minimum':0,'exclusiveMaximum':3,'multipleOf':0.5},
        'text':{'type':'string','minLength':1,'maxLength':3,'pattern':'^a'},
        'count':{'type':'integer','enum':[0,2]}}, ['score'])
    compiled=arguments_schema(schema);assert_strict(compiled)
    host,wire=validator(schema),validator(compiled)
    for score,text,count in itertools.product([None,False,-1,0,0.5,0.6,2,3],
            ['', 'a', 'abc', 'abcd', 'b', None], [0,1,2,True,None]):
        sample={'score':score,'text':text,'count':count}
        assert host.is_valid(sample)==wire.is_valid(sample),sample


def test_unique_items_is_left_to_host_validation():
    schema=closed({'ids':{'type':'array','items':{'type':'string'},'maxItems':3,'uniqueItems':True}}, ['ids'])
    compiled=arguments_schema(schema);assert_strict(compiled)
    assert 'uniqueItems' not in compiled['properties']['ids']
    assert schema['properties']['ids']['uniqueItems'] is True
    assert validator(compiled).is_valid({'ids':['a','a']})
    assert not validator(schema).is_valid({'ids':['a','a']})
    assert not validator(compiled).is_valid({'ids':['a','b','c','d']})


def test_actual_discovery_six_tools_compile_without_changing_host_schema(tmp_path,monkeypatch):
    from adj_partner.application import AgentService
    monkeypatch.delenv('ADJ_PARTNER_ENV_FILE',raising=False)
    port=CodexAppServerPort(model='gpt-6-luna',trace_dir=tmp_path/'calls',
        runner=lambda **kw:pytest.fail('schema inspection must never dispatch'))
    env_file=tmp_path/'empty.env';env_file.write_text('')
    service=AgentService(tmp_path/'state',project_root=tmp_path,model=port,env_file=env_file)
    try:
        names=['code_interpreter','file_inspect','artifact_read','job_status','catalog_search','tools_enable']
        tools=[service.catalog.get(name).spec.model_schema() for name in names]
        before=deepcopy(tools); compiled=envelope_schema(tools)
        assert_strict(compiled)
        assert tools==before
        assert len(compiled['properties']['tool_calls']['items']['anyOf'])==6
    finally:
        service.close()


def test_analysis_definitions_support_closed_tools_and_explicitly_reject_dynamic_tools():
    from adj_partner.analysis.capabilities import AnalysisCapabilities
    for spec in AnalysisCapabilities.definitions():
        if spec.name in {'analysis_method_register','analysis_method_run'}:
            with pytest.raises(ProviderFailure) as exc: envelope_schema([spec.model_schema()])
            assert exc.value.code=='invalid_request'
        else:
            assert_strict(envelope_schema([spec.model_schema()]))


def test_existing_frozen_preflight_tool_contract_compiles_offline():
    path=Path(__file__).resolve().parents[1]/'workspace/iterative-benefit-20261022/run_transport_v05_preflight.py'
    spec=importlib.util.spec_from_file_location('transport_contract_for_schema_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    before=deepcopy(module.STAGE1_TOOLS)
    assert_strict(envelope_schema(module.STAGE1_TOOLS))
    assert module.STAGE1_TOOLS==before


def test_metadata_declares_compiler_subset_and_limits(tmp_path):
    metadata=CodexAppServerPort(model='gpt-6-luna',trace_dir=tmp_path).metadata
    assert metadata['transport_version']=='0.7'
    assert metadata['schema_compiler']=='closed-exact-keyset-v1'
    assert metadata['schema_limits']['branches']==256
    assert metadata['schema_limits']['properties']==4096
    assert metadata['schema_limits']['depth']==10
    assert metadata['schema_limits']['bytes']==1024*1024
