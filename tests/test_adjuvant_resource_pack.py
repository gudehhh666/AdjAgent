"""Resource-pack integrity, real retrieval semantics, and native evidence import."""
import hashlib
import importlib
import json

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.native.capabilities import NativeCapabilities
from adj_partner.research import ResearchStore
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor


def make_pack(path):
    path.mkdir()
    records = [
        {'id': 'VO_1', 'dataset': 'vo', 'kind': 'ontology', 'title': 'aluminum hydroxide',
         'aliases': ['alum'], 'fields': {'definition': 'Curated ontology definition'},
         'source_locator': {'iri': 'https://example.org/VO_1'}},
        {'id': 'VO_2', 'dataset': 'vo', 'kind': 'ontology', 'title': 'mineral salt',
         'aliases': [], 'fields': {}, 'source_locator': {'iri': 'https://example.org/VO_2'}},
        {'id': 'assay:2', 'dataset': 'screen', 'kind': 'assay', 'title': 'S1002 Primary',
         'aliases': ['S1002'], 'fields': {'MPLA Lum': '0.6626992820000001', 'Viability Mask 1': ''},
         'source_locator': {'row': 2}},
    ]
    edges = [{'subject': 'VO_1', 'predicate': 'subClassOf', 'object': 'VO_2',
              'source_record': 'VO_1', 'qualifier': 'asserted_named_parent'}]
    manifest = {'schema_version': 1, 'datasets': [
        {'id': d, 'source_url': 'https://example.org/'+d, 'raw_sha256': 'a'*64,
         'retrieved_at': '2026-09-30T00:00:00+00:00', 'license': 'fixture',
         'evidence_scope': 'Fixture records only', 'limitations': ['Not vaccine efficacy']} for d in ['vo', 'screen']], 'files': {}}
    for name, value in [('records.jsonl', records), ('relations.jsonl', edges)]:
        raw = ''.join(json.dumps(x)+'\n' for x in value).encode()
        (path/name).write_bytes(raw)
        manifest['files'][name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    (path/'manifest.json').write_text(json.dumps(manifest))
    return path


def load(path):
    module = importlib.import_module('adj_partner.resources.pack')
    return module.ResourcePack(path)


def test_query_alias_exact_id_no_cross_assay_inference(tmp_path):
    p = load(make_pack(tmp_path/'pack'))
    assert p.search('alum')['results'][0]['id'] == 'VO_1'
    assert p.search('S1002', kind='ontology')['hit_count'] == 0
    assert p.search('unseen compound')['hit_count'] == 0
    row = p.read('assay:2')
    assert row['fields']['MPLA Lum'] == '0.6626992820000001'
    assert row['fields']['Viability Mask 1'] == ''
    with pytest.raises(KeyError): p.read('missing')
    with pytest.raises(ValueError): p.search('', limit=1)
    with pytest.raises(ValueError): p.search('alum', limit=True)
    assert p.neighbors('VO_1')['relations'][0]['object'] == 'VO_2'
    assert p.neighbors('VO_2', direction='incoming')['relations'][0]['subject'] == 'VO_1'


def test_corruption_duplicate_id_and_manifest_path_rejected(tmp_path):
    path = make_pack(tmp_path/'pack')
    (path/'records.jsonl').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='integrity'): load(path)
    path2 = make_pack(tmp_path/'pack2')
    raw=(path2/'records.jsonl').read_bytes()
    raw += raw.splitlines(keepends=True)[0]
    (path2/'records.jsonl').write_bytes(raw)
    m=json.loads((path2/'manifest.json').read_text())
    m['files']['records.jsonl']={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    (path2/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError, match='Duplicate'): load(path2)
    path3 = make_pack(tmp_path/'pack3')
    (path3/'records.jsonl').unlink()
    (path3/'records.jsonl').symlink_to(path2/'records.jsonl')
    with pytest.raises(ValueError, match='symlink'): load(path3)


def test_exact_filters_keep_types_missingness_and_query_intersection(tmp_path):
    p = load(make_pack(tmp_path/'pack'))
    p.records['VO_1']['fields'].update(flag=True, count=1, nullable=None)
    p.records['VO_2']['fields'].update(flag=1, count='1')
    assert p.search(dataset='vo', filters={'flag':True})['hit_count'] == 1
    assert p.search(dataset='vo', filters={'count':1})['results'][0]['id'] == 'VO_1'
    assert p.search(dataset='vo', filters={'nullable':None})['hit_count'] == 1
    assert p.search('mineral', dataset='vo', filters={'count':1})['hit_count'] == 0
    assert p.search('alum', dataset='vo', filters={'count':1})['hit_count'] == 1
    for filters in ({}, {'flag':[]}, {'flag':float('nan')}, {'unknown':None}):
        with pytest.raises(ValueError): p.search(dataset='vo', filters=filters)
    with pytest.raises(ValueError): p.search()


def test_native_offline_source_import_and_permission(tmp_path):
    p=load(make_pack(tmp_path/'pack'))
    store=Store(tmp_path/'state'); research=ResearchStore(store); catalog=ToolCatalog()
    NativeCapabilities(store,research=research,resource_pack=p).register(catalog)
    sid=store.create_session(SessionConfig(allowed_tools=['adjuvant_search','adjuvant_record'],
        enabled_tools=['adjuvant_search','adjuvant_record'],allow_network=False).to_dict())['id']
    tid=store.create_task(sid,'Read actual pack',{})['id']
    rid=store.create_run(sid,tid,Budget().to_dict())['id']; executor=ToolExecutor(store,catalog)
    result=executor.execute(ToolCall('lookup','adjuvant_record',{'record_id':'assay:2'}),sid,rid)
    assert result.status=='ok',result.to_dict()
    assert result.data['record']['fields']['Viability Mask 1']==''
    source=research.get_source(sid,rid,result.data['source_id'])
    assert source['provenance']['raw_sha256']=='a'*64
    evidence=research.select_evidence(sid,rid,source['id'],result.data['selector'])
    assert evidence['value']['id']=='assay:2'
    denied=executor.execute(ToolCall('denied','adjuvant_relations',{'record_id':'VO_1'}),sid,rid)
    assert denied.status!='ok'
    # Exact projected values and provenance must survive the actual context
    # budget; the model must not need to reread many metadata pages.
    from adj_partner.runtime.context import ContextView
    projected=executor.execute(ToolCall('projected','adjuvant_record',
        {'record_id':'assay:2','fields':['MPLA Lum']}),sid,rid)
    assert projected.status=='ok',projected.to_dict()
    message=ContextView(store).tool_message(sid,rid,'projected',projected.to_dict())
    view=json.loads(message['content'])
    summary=view['data']['summary']
    assert summary['values']=={'MPLA Lum':'0.6626992820000001'}
    assert summary['raw_sha256']=='a'*64
    assert len(message['content'].encode())<=ContextView.PREVIEW_MAX_BYTES
    spec=catalog.get('adjuvant_search').spec.input_schema['properties']
    assert spec['kind']['enum']==['assay','ontology']
    assert spec['dataset']['enum']==['screen','vo']


def test_builder_keeps_repeated_catalogs_rows_missing_values_and_owl_qualifiers(tmp_path):
    builder=importlib.import_module('adj_partner.resources.build')
    csv=tmp_path/'screen.csv'
    csv.write_text('Catalog,PME Code,SMILES,Viability Mask 1,LPS Abs\nS1,Primary,C,,0.5\nS1,PME-1,C,95,2.0\n')
    records=builder.screening_records(csv)
    assert len(records)==2 and records[0]['id']!=records[1]['id']
    assert records[0]['fields']['Viability Mask 1']==''
    assert records[1]['source_locator']['row']==3
    owl=tmp_path/'vo.owl'
    owl.write_text('''<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#">
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/VO_0000580"><rdfs:label>vaccine adjuvant</rdfs:label></owl:Class>
    <owl:Class rdf:about="http://purl.obolibrary.org/obo/VO_123"><rdfs:label>test adjuvant</rdfs:label><rdfs:subClassOf rdf:resource="http://purl.obolibrary.org/obo/VO_0000580"/><rdfs:subClassOf><owl:Restriction><owl:onProperty rdf:resource="http://example.org/has_target"/><owl:someValuesFrom rdf:resource="http://example.org/target"/></owl:Restriction></rdfs:subClassOf></owl:Class>
    <owl:Class rdf:about="http://example.org/target"><rdfs:label>target</rdfs:label></owl:Class></rdf:RDF>''')
    records,edges=builder.ontology_records(owl)
    assert {x['id'] for x in records}=={'VO_123','VO_0000580','http://example.org/target'}
    restriction=next(e for e in edges if e['predicate']=='http://example.org/has_target')
    assert restriction['qualifier']=='someValuesFrom'
    assert restriction['object']=='http://example.org/target'
