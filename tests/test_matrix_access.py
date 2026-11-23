import gzip
import hashlib
import json
from pathlib import Path

import pytest

from adj_partner.matrix_access import parse_matrix, matrix_page, register_matrix_access


TEXT = '''!Series_title\t"A public study"
!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"
!Sample_characteristics_ch1\t"donor: a"\t"donor: a"\t"donor: b"\t""
!Sample_characteristics_ch1\t"visit: 0"\t"visit: 1"\t"visit: 0"\t"visit: 0"
!Sample_title\t"一"\t"two"\t"three"\t"four"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"\t"GSM4"
"probe"\t"1"\t"2"\t"3"\t"4"
!series_matrix_table_end
'''


def test_schema_preserves_repeated_fields_and_exact_cells():
    m = parse_matrix(TEXT.encode())
    assert m['sample_count'] == 4 and m['matrix_body_present']
    fields = m['fields']
    assert [x['id'] for x in fields] == ['f0', 'f1', 'f2', 'f3']
    assert fields[1]['name'] == fields[2]['name'] == '!Sample_characteristics_ch1'
    assert [fields[x]['occurrence'] for x in (1, 2)] == [1, 2]
    assert fields[1]['label'] == 'donor' and fields[1]['nonempty'] == 3
    assert fields[1]['unique_nonempty'] == 2 and fields[1]['type'] == 'string'
    p = matrix_page(m, 'a'*64, mode='rows', fields=['f1', 'f2'])
    assert p['items'][0] == ['donor: a', 'visit: 0']
    assert p['items'][3] == ['', 'visit: 0']


def test_counts_generic_grouping_and_metadata_boundary():
    m = parse_matrix(TEXT.encode())
    p = matrix_page(m, 'a'*64, mode='counts', fields=['f2', 'f1'])
    assert sum(x['count'] for x in p['items']) == 4
    assert {'values': ['visit: 0', ''], 'count': 1} in p['items']
    assert 'metadata' in p['scope'] and p['expression_values_inspected'] is False
    with pytest.raises(ValueError): matrix_page(m, 'a'*64, mode='rows', fields=['not_present'])
    with pytest.raises(ValueError): matrix_page(m, 'a'*64, mode='counts', fields=['f1', 'f1'])


def test_metadata_shape_empty_duplicates_and_mixed_labels():
    text = TEXT.replace('"GSM4"', '"GSM1"').replace('"donor: b"', '"tissue: b"')
    m = parse_matrix(text.encode())
    assert m['duplicate_accessions'] == 1 and m['fields'][1]['label'] is None
    with pytest.raises(ValueError): parse_matrix(TEXT.replace('"GSM4"\n', '\n', 1).encode())
    with pytest.raises(ValueError): parse_matrix(b'!Sample_title\t"x"\n')


def test_size_bound_and_pagination_preserve_complete_strings():
    from adj_partner.contracts import ToolResult
    from adj_partner.tools.catalog import canonical_json
    text = TEXT.replace('"one"', '"x"')
    m = parse_matrix(text.encode())
    seen = []; offset = 0
    while True:
        p = matrix_page(m, 'b'*64, mode='schema', offset=offset, limit=1)
        assert len(canonical_json(ToolResult(data={'summary': p}).to_dict()).encode()) < 2400
        seen.extend(p['items'])
        if p['next_offset'] is None: break
        offset = p['next_offset']
    assert len(seen) == 4
    m = parse_matrix(TEXT.replace('"一"', '"' + '长'*3000 + '"').encode())
    with pytest.raises(ValueError, match='page|fit|large'):
        matrix_page(m, 'b'*64, mode='rows', fields=['f3'])


@pytest.fixture
def runtime(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.models import ScriptedPort
    from adj_partner.tools.policy import SessionToolset
    root = Path(__file__).resolve().parents[1]
    svc = AgentService(tmp_path/'state', project_root=root, model=ScriptedPort([]))
    if 'geo_matrix_metadata' not in svc.catalog.names(): register_matrix_access(svc)
    inputs = tmp_path/'inputs'; inputs.mkdir()
    sid = svc.create_session(allowed_tools=['geo_matrix_metadata'], allowed_roots=[str(inputs)])['id']
    SessionToolset(svc.store, svc.catalog, sid).enable('geo_matrix_metadata')
    run = svc.submit(sid, 'inspect study sample metadata')
    yield svc, sid, run['id'], inputs
    svc.close()


def call(runtime, path, **args):
    from adj_partner.contracts import ToolCall
    from uuid import uuid4
    svc, sid, rid, _ = runtime
    return svc.executor.execute(ToolCall('matrix'+uuid4().hex, 'geo_matrix_metadata', {'path': str(path), **args}), sid, rid)


def test_real_permissions_hash_and_context_projection(runtime):
    from adj_partner.runtime.context import ContextView
    svc, sid, rid, inputs = runtime
    p = inputs/'series.txt.gz'; p.write_bytes(gzip.compress(TEXT.encode()))
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    result = call(runtime, p, expected_sha256=digest)
    assert result.status == 'ok', result.to_dict()
    visible = ContextView(svc.store).tool_message(sid, rid, 'c', result.to_dict())
    assert len(visible['content'].encode()) < 2400
    assert not json.loads(visible['content']).get('context_truncated')
    assert result.data['summary']['sha256'] == digest
    assert call(runtime, p, expected_sha256='0'*64).status != 'ok'
    p.write_bytes(gzip.compress(TEXT.replace('A public study', 'changed').encode()))
    assert call(runtime, p, expected_sha256=digest).status != 'ok'
    outside = inputs.parent/'outside.txt'; outside.write_text(TEXT)
    assert call(runtime, outside).status != 'ok'
    secret = inputs/'.env'; secret.write_text(TEXT)
    assert call(runtime, secret).status != 'ok'
    linked = inputs/'link.txt'; linked.symlink_to(p)
    assert call(runtime, linked).status != 'ok'


def test_decompression_budget_and_redaction(runtime, monkeypatch):
    import adj_partner.matrix_access as module
    monkeypatch.setattr(module, 'MAX_DECOMPRESSED_BYTES', 100)
    p = runtime[3]/'bomb.txt.gz'; p.write_bytes(gzip.compress(b'x'*101))
    assert call(runtime, p).status != 'ok'


def test_real_secret_redaction_and_empty_roots(runtime):
    svc, sid, _, inputs = runtime
    secret = 'long-private-credential-98237892379'
    svc.store.register_secrets([secret])
    p = inputs/'series.txt'; p.write_text(TEXT.replace('"two"', '"'+secret+'"'))
    r = call(runtime, p, mode='rows', fields=['f3'])
    assert r.status == 'ok' and secret not in json.dumps(r.to_dict())
    assert r.data['summary']['items'][1][0] != secret
    session = svc.store.get_session(sid)
    svc.store.update_session(sid, {**session['config'], 'allowed_roots': []}, session['revision'])
    assert call(runtime, p).status != 'ok'


def test_other_study_field_names_and_all_pages_are_generic():
    text = TEXT.replace('donor:', 'sample origin:').replace('visit:', 'storage batch:')
    m = parse_matrix(text.encode())
    assert m['fields'][1]['label'] == 'sample origin'
    assert m['fields'][2]['label'] == 'storage batch'
    assert matrix_page(m, 'b'*64, mode='rows', fields=['f3', 'f0'])['items'][0] == ['一', 'GSM1']
    p = matrix_page(m, 'b'*64, mode='schema', offset=4)
    assert p['items'] == [] and p['next_offset'] is None
    with pytest.raises(ValueError): matrix_page(m, 'b'*64, offset=5)


def test_actual_development_matrix_schema_and_pair_counts(runtime):
    # Development regression only: never loaded as runtime hints or evaluation inputs.
    p = Path(__file__).resolve().parents[1]/'data/research_public_inputs/GSE244794_series_matrix.txt.gz'
    local = runtime[3]/p.name; local.write_bytes(p.read_bytes())
    m = parse_matrix(gzip.decompress(p.read_bytes()))
    assert m['sample_count'] == 45
    ids = [f['id'] for f in m['fields'] if f['label'] in ('individual', 'time')]
    assert len(ids) == 2
    offset = 0; groups = []
    while True:
        r = call(runtime, local, mode='counts', fields=ids, offset=offset)
        assert r.status == 'ok', r.to_dict()
        from adj_partner.runtime.context import ContextView
        svc, sid, rid, _ = runtime
        visible = ContextView(svc.store).tool_message(sid, rid, 'counts', r.to_dict())
        assert len(visible['content'].encode()) < 2400
        assert not json.loads(visible['content']).get('context_truncated')
        groups.extend(r.data['summary']['items'])
        offset = r.data['summary']['next_offset']
        if offset is None: break
    assert len(groups) == 45 and sum(x['count'] for x in groups) == 45
    assert len({x['values'][0] for x in groups}) == 9
    offset = 0; fields = []
    while True:
        r = call(runtime, local, mode='schema', offset=offset)
        assert r.status == 'ok', r.to_dict()
        visible = ContextView(svc.store).tool_message(sid, rid, 'schema', r.to_dict())
        assert len(visible['content'].encode()) < 2400
        assert not json.loads(visible['content']).get('context_truncated')
        fields.extend(r.data['summary']['items'])
        offset = r.data['summary']['next_offset']
        if offset is None: break
    assert fields == m['fields']
