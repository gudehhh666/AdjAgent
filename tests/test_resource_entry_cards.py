"""Navigation must expose readers without exposing source values or permissions."""
import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

from adj_partner.resource_entry_cards import initial_navigation_cards
from adj_partner.resources.pack import ResourcePack
from adj_partner.tools.catalog import canonical_json
from test_adjuvant_resource_pack import make_pack


def setup_service(tmp_path):
    return SimpleNamespace(project_root=tmp_path,
        native=SimpleNamespace(resource_pack=ResourcePack(make_pack(tmp_path/'pack'))))


def config(**extra):
    return {'allowed_roots': [], 'allowed_tools': ['tools_enable', 'adjuvant_read_fields',
        'adjuvant_resource_guide', 'adjuvant_lookup', 'adjuvant_cite'], **extra}


def test_pack_card_has_exact_reader_but_no_evidence_or_permission_change(tmp_path):
    service=setup_service(tmp_path); cfg=config(); before=deepcopy(cfg)
    cards=initial_navigation_cards(service, 'What does alum mean?', cfg)
    assert len(cards)==1
    card=cards[0]
    assert card['open']=={'tool':'adjuvant_read_fields',
        'arguments':{'record_id':'VO_1','fields':['definition']}}
    assert card['enable']=={'tool':'tools_enable','arguments':{'name':'adjuvant_read_fields'}}
    assert 'Curated ontology definition' not in canonical_json(cards)
    assert card['pack_sha256']==service.native.resource_pack.manifest_hash
    assert cfg==before and 'path' not in card


def test_unknown_topic_field_values_and_unauthorized_tools_do_not_match(tmp_path):
    service=setup_service(tmp_path)
    assert initial_navigation_cards(service, 'Compare reinforcement learning algorithms', config())==[]
    assert initial_navigation_cards(service, 'Curated ontology definition', config())==[]
    assert initial_navigation_cards(service, 'alum', config(allowed_tools=['tools_enable']))==[]
    assert initial_navigation_cards(service, 'alum', config(allowed_tools=['adjuvant_read_fields']))==[]
    assert initial_navigation_cards(service, 'alum', config(allowed_tools=['adjuvant_read_fields'],
        enabled_tools=['adjuvant_read_fields']))


def test_authorized_registry_file_combines_with_pack_without_root_escape(tmp_path):
    service=setup_service(tmp_path); (tmp_path/'config').mkdir()
    permitted=tmp_path/'inputs'; permitted.mkdir()
    source=permitted/'ALUM.csv'; source.write_text('label,value\na,2\n')
    outside=tmp_path/'outside.csv'; outside.write_text('secret\n')
    items=[{'id':'ALUM-file-'+p.stem,'path':str(p),
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'need':'table',
        'scope':'ALUM data','reader':'code_interpreter'} for p in (source,outside)]
    (tmp_path/'config/research-resource-index.json').write_text(json.dumps({'resources':items}))
    cfg=config(allowed_roots=[str(permitted)], allowed_tools=config()['allowed_tools']+['code_interpreter'])
    cards=initial_navigation_cards(service,'Read ALUM data',cfg)
    assert len(cards)==2
    assert {c['kind'] for c in cards}=={'authorized_file','pack_record'}
    assert str(outside) not in canonical_json(cards)
    assert len(canonical_json(cards).encode())<=2200


def test_symlink_registry_file_omitted_and_cards_stay_bounded(tmp_path):
    service=setup_service(tmp_path); (tmp_path/'config').mkdir()
    outside=tmp_path/'outside.csv'; outside.write_text('secret\n')
    permitted=tmp_path/'inputs'; permitted.mkdir(); link=permitted/'ALUM.csv'; link.symlink_to(outside)
    item={'id':'ALUM','path':str(link),'sha256':hashlib.sha256(outside.read_bytes()).hexdigest(),
        'need':'table','scope':'ALUM data','reader':'code_interpreter'}
    (tmp_path/'config/research-resource-index.json').write_text(json.dumps({'resources':[item]}))
    cfg=config(allowed_roots=[str(permitted)],allowed_tools=config()['allowed_tools']+['code_interpreter'])
    assert all(c['kind']=='pack_record' for c in initial_navigation_cards(service,'ALUM',cfg))
    assert initial_navigation_cards(service,'ALUM',cfg,max_bytes=10)==[]


def test_duplicate_alias_hits_select_deterministically_without_reading_values(tmp_path):
    service=setup_service(tmp_path); pack=service.native.resource_pack
    pack.records['VO_2']['aliases']=['alum']
    class ValuesForbidden(dict):
        def __getitem__(self,key): raise AssertionError('Field values must not be read')
    for record in pack.records.values(): record['fields']=ValuesForbidden(record['fields'])
    cards=initial_navigation_cards(service,'alum',config())
    assert [c['record_id'] for c in cards]==['VO_1']  # empty schema has no read entry
    assert cards==initial_navigation_cards(service,'alum',config())


def test_generated_open_executes_actual_reader_without_pack_filesystem_grant(tmp_path):
    from adj_partner.contracts import SessionConfig, Budget, ToolCall
    from adj_partner.native.capabilities import NativeCapabilities
    from adj_partner.research import ResearchStore
    from adj_partner.storage import Store
    from adj_partner.tools import ToolCatalog, ToolExecutor
    service=setup_service(tmp_path)
    cfg=config(enabled_tools=['adjuvant_read_fields'])
    card=initial_navigation_cards(service,'What is alum?',cfg)[0]
    store=Store(tmp_path/'state')
    try:
        research=ResearchStore(store); catalog=ToolCatalog()
        NativeCapabilities(store,research=research,resource_pack=service.native.resource_pack).register(catalog)
        sid=store.create_session(SessionConfig(allowed_tools=['adjuvant_read_fields'],
            enabled_tools=['adjuvant_read_fields'],allowed_roots=[],allow_network=False).to_dict())['id']
        tid=store.create_task(sid,'Read source',{})['id']
        rid=store.create_run(sid,tid,Budget().to_dict())['id']
        result=ToolExecutor(store,catalog).execute(ToolCall('entry',card['open']['tool'],
            card['open']['arguments']),sid,rid)
        assert result.status=='ok', result.to_dict()
        assert result.data['summary']['values']=={'definition':'Curated ontology definition'}
        assert result.data['summary']['source_id']
        assert store.get_session(sid)['config']['allowed_roots']==[]
    finally:
        store.close()


def envelope(question, input_text='', **metadata):
    return 'Answer the task as JSON. Shared subject labels and adjuvant metadata follow.\n'+json.dumps(
        {'question':question,'input_text':input_text,**metadata})


def test_envelope_navigation_uses_question_only_not_format_metadata_or_cells(tmp_path):
    service=setup_service(tmp_path)
    clean=envelope('Count the supplied rows.', 'sample,value\nx,1\n', id='neutral')
    polluted=envelope('Count the supplied rows.',
        'sample,value\nVO_1,alum\nS1002,aluminum hydroxide\n',
        id='VO_1', source_family='alum', track='mineral salt')
    assert initial_navigation_cards(service,clean,config())==[]
    assert initial_navigation_cards(service,polluted,config())==[]
    assert initial_navigation_cards(service,envelope('What does alum mean?',input_text=''),config())


def test_exact_requested_record_excludes_other_entity_and_weak_fill(tmp_path):
    service=setup_service(tmp_path)
    record=service.native.resource_pack.records['VO_2']
    record.update(title='aluminum mineral',aliases=['mineral'],fields={'definition':'Other record'})
    cards=initial_navigation_cards(service,
        envelope('Read VO_1; compare its definition to the general aluminum mineral concept.'),config())
    assert [c['record_id'] for c in cards]==['VO_1']
    cards=initial_navigation_cards(service,envelope('Read VO_1 and assay:2.'),config())
    assert {c['record_id'] for c in cards}=={'VO_1','assay:2'}


def test_accession_suffix_is_exact_record_identity_not_a_generic_alias(tmp_path):
    service=setup_service(tmp_path); records=service.native.resource_pack.records
    records['trial:NCT12345678']={'id':'trial:NCT12345678','dataset':'screen','kind':'trial',
        'title':'Vaccine study','aliases':['NCT12345678'],'fields':{'status':'snapshot'},'source_locator':{}}
    cards=initial_navigation_cards(service,envelope('Read NCT12345678; alum is background context.'),config())
    assert [c['record_id'] for c in cards]==['trial:NCT12345678']


def test_two_overlapping_words_are_not_an_automatic_resource_entry(tmp_path):
    service=setup_service(tmp_path)
    records=service.native.resource_pack.records
    records['trial:X123']={'id':'trial:X123','dataset':'screen','kind':'trial',
        'title':'Previously enrolled GEN-003-002 subjects','aliases':[],
        'fields':{'status':'snapshot'},'source_locator':{}}
    assert initial_navigation_cards(service,
        'Count subjects assigned labels 002 and 003 in the supplied sample sheet.',config())==[]


def test_freeform_source_block_does_not_supply_resource_identifiers(tmp_path):
    service=setup_service(tmp_path)
    for goal in ('Count rows.\n\nInput data:\n```csv\nsample,label\nVO_1,alum\n```',
                 'Count rows.\n\nInput data:\nsample,label\nVO_1,alum\n'):
        assert initial_navigation_cards(service,goal,config())==[]
    assert initial_navigation_cards(service,'Read VO_1.\n\nInput data:\nsample,label\nx,1',config())


def test_task_inputs_gate_entity_cards_but_explicit_record_request_overrides(tmp_path):
    service=setup_service(tmp_path)
    assert initial_navigation_cards(service,'Count alum sample rows.',config(),has_task_inputs=True)==[]
    cards=initial_navigation_cards(service,'Read VO_1 and count the input rows.',config(),has_task_inputs=True)
    assert [c['record_id'] for c in cards]==['VO_1']


def test_bound_context_passes_actual_input_presence_and_preserves_id_exception(tmp_path):
    from adj_partner.evidence_bound_entry import bound_cards
    service=setup_service(tmp_path); service.store=SimpleNamespace(root=tmp_path/'private-state')
    source=tmp_path/'samples.csv'; source.write_text('sample,label\nx,alum\n')
    cfg=config(allowed_roots=[str(tmp_path)],allowed_tools=config()['allowed_tools']+['code_interpreter'])
    cards=bound_cards(service,envelope('Count alum samples.', f'Read `{source}`.'),cfg)
    assert len(cards)==1 and cards[0]['path']==str(source)
    cards=bound_cards(service,envelope('Read VO_1, then count samples.', f'Read `{source}`.'),cfg)
    assert any(c.get('path')==str(source) for c in cards)
    assert [c['record_id'] for c in cards if c.get('kind')=='pack_record']==['VO_1']


def test_named_filename_does_not_masquerade_as_requested_record(tmp_path):
    service=setup_service(tmp_path)
    assert initial_navigation_cards(service,'Read `inputs/VO_1.csv` and count rows.',
        config(),has_task_inputs=True)==[]


def test_card_exposes_original_schema_and_exact_composite_cite_without_values(tmp_path):
    service=setup_service(tmp_path)
    record=service.native.resource_pack.records['VO_1']
    record['fields'].update(status='REGISTERED',
        status_dates={'lastUpdateSubmitDate':'2041-09-21','primaryCompletionDate':'2040-03'},
        has_results=False)
    card=initial_navigation_cards(service,'Read VO_1 status_dates and has_results.',config())[0]
    assert set(card['field_names'])==set(record['fields'])
    assert card['composite_fields']=={'status_dates':'object'}
    assert '2041-09-21' not in canonical_json(card) and 'REGISTERED' not in canonical_json(card)
    assert 'lastUpdateSubmitDate' not in canonical_json(card)
    assert card['long_field_reader']=={'tool':'adjuvant_cite','arguments':{
        'record_id':'VO_1','field':'status_dates','length':600}}


def test_emitted_cite_reads_actual_nested_field_with_valid_original_schema(tmp_path):
    from adj_partner.contracts import SessionConfig, Budget, ToolCall
    from adj_partner.native.capabilities import NativeCapabilities
    from adj_partner.research import ResearchStore
    from adj_partner.runtime.context import ContextView
    from adj_partner.storage import Store
    from adj_partner.tools import ToolCatalog, ToolExecutor
    service=setup_service(tmp_path); record=service.native.resource_pack.records['VO_1']
    dates={'lastUpdateSubmitDate':'2041-09-21','primaryCompletionDate':'2040-03'}
    record['fields']['status_dates']=dates
    card=initial_navigation_cards(service,'Read VO_1 status_dates.',config())[0]
    store=Store(tmp_path/'state')
    try:
        research=ResearchStore(store); catalog=ToolCatalog()
        NativeCapabilities(store,research=research,resource_pack=service.native.resource_pack).register(catalog)
        sid=store.create_session(SessionConfig(allowed_tools=['adjuvant_cite'],
            enabled_tools=['adjuvant_cite'],allowed_roots=[],allow_network=False).to_dict())['id']
        tid=store.create_task(sid,'Read exact field',{})['id']; rid=store.create_run(sid,tid,Budget().to_dict())['id']
        opened=card['long_field_reader']
        result=ToolExecutor(store,catalog).execute(ToolCall('cite',opened['tool'],opened['arguments']),sid,rid)
        assert result.status=='ok',result.to_dict()
        projected=json.loads(ContextView(store).tool_message(sid,rid,'cite',result.to_dict())['content'])
        packet=projected['data']['summary']
        assert json.loads(packet['citation']['quote'])==dates
        assert packet['truncated'] is False
        assert packet['selector']['pointer']=='/record/fields/status_dates'
        assert packet['source_id']
    finally:
        store.close()


def test_cite_hint_does_not_bypass_tool_authorization(tmp_path):
    service=setup_service(tmp_path)
    card=initial_navigation_cards(service,'Read VO_1.',
        config(allowed_tools=['tools_enable','adjuvant_read_fields','adjuvant_lookup']))[0]
    assert card['long_field_reader']['tool']=='adjuvant_lookup'


def test_requested_composite_remains_readable_when_not_in_first_three_fields(tmp_path):
    service=setup_service(tmp_path)
    service.native.resource_pack.records['VO_1']['fields'].update(
        nct_id='NCT12345678',status='REGISTERED',has_results=False,
        status_dates={'lastUpdateSubmitDate':'2041-09-21'})
    card=initial_navigation_cards(service,
        'Read VO_1 nct_id, has_results, status, last_update_submit_date.',config())[0]
    assert 'status_dates' not in card['open']['arguments']['fields']
    assert card['long_field_reader']['arguments']['field']=='status_dates'
