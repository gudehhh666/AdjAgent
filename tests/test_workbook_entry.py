"""Literal bounded previews must survive malformed declared workbook dimensions."""
import hashlib
import importlib
import importlib.util
import io
import json
import re
import stat
import uuid
import zipfile

import pytest
from openpyxl import Workbook

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor


@pytest.fixture
def env(tmp_path):
    assert importlib.util.find_spec('adj_partner.native.workbooks'), 'workbook entry must exist'
    module = importlib.import_module('adj_partner.native.workbooks')
    store = Store(tmp_path / 'state'); inputs = tmp_path / 'inputs'; inputs.mkdir()
    catalog = ToolCatalog(); module.register_workbooks(catalog)
    sid = store.create_session(SessionConfig(allowed_roots=[str(inputs)],
        allowed_tools=catalog.names(), enabled_tools=catalog.names()).to_dict())['id']
    tid = store.create_task(sid, 'inspect original workbook', {})['id']
    rid = store.create_run(sid, tid, Budget().to_dict())['id']
    return module, store, inputs, ToolExecutor(store, catalog), sid, rid


def workbook(rows, bad_dimension=False):
    wb = Workbook(); ws = wb.active; ws.title = 'Raw'
    for row in rows: ws.append(row)
    stream = io.BytesIO(); wb.save(stream); wb.close(); raw = stream.getvalue()
    if bad_dimension:
        dest = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(raw)) as src, zipfile.ZipFile(dest, 'w') as out:
            for name in src.namelist():
                content = src.read(name)
                if name == 'xl/worksheets/sheet1.xml':
                    content = re.sub(rb'<dimension[^>]*/>', b'<dimension ref="A1"/>', content)
                out.writestr(name, content)
        raw = dest.getvalue()
    return raw


def file(env, raw, name='book.xlsx'):
    path = env[2] / name; path.write_bytes(raw); return path


def bundle(env, members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for name, raw in members: z.writestr(name, raw)
    return file(env, stream.getvalue(), 'bundle.zip')


def run(env, path, **kw):
    return env[3].execute(ToolCall('read_'+uuid.uuid4().hex, 'workbook_inspect', {'path': str(path), **kw}), env[4], env[5])


def document(env, result):
    assert result.status == 'ok', result.to_dict()
    return json.loads(env[1].read_artifact(result.data['artifact_id']))


def test_wrong_dimensions_do_not_hide_columns_or_rows(env):
    raw = workbook([['Group', None, 'Assay'], ['ID', 'Match', 'Match'], ['S1 ', 0, None]], True)
    p = file(env, raw); result = run(env, p, row_start=2, rows=2, columns=3)
    d = document(env, result)
    assert d['declared_dimensions'] == [1, 1]
    assert [(c['coordinate'], c['value']) for c in d['cells']] == [
        ('A2','ID'),('B2','Match'),('C2','Match'),('A3','S1 '),('B3',0),('C3',None)]
    assert d['whole_sheet_inspected'] is False
    assert d['observed_nonempty_outside_declared_dimensions'] is True
    assert p.read_bytes() == raw


def test_archive_inventory_and_exact_selection_hashes(env):
    raw = workbook([['x'], [5]])
    p = bundle(env, [('nested/book.xlsx', raw), ('notes.txt', b'context')])
    d = document(env, run(env, p))
    assert d['mode'] == 'archive_inventory'
    assert d['members'][0]['sha256'] == hashlib.sha256(raw).hexdigest()
    assert len(d['members']) == 2
    got = document(env, run(env, p, member='nested/book.xlsx', expected_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    assert got['member']['path'] == 'nested/book.xlsx'
    assert got['member']['sha256'] == hashlib.sha256(raw).hexdigest()
    assert got['cells'][0]['value'] == 'x'


def test_formula_is_not_executed_and_missing_cache_is_explicit(env):
    d = document(env, run(env, file(env, workbook([['=1+2', 0, None]])), rows=1, columns=3))
    formula = d['cells'][0]
    assert formula['value'] == '=1+2' and formula['type'] == 'f'
    assert formula['cached_value'] is None and formula['cache_present'] is False
    assert d['cells'][1]['value'] == 0 and d['cells'][2]['value'] is None


def test_preview_is_bounded_with_coordinates_for_continuation(env):
    p = file(env, workbook([[str(i) + '中'*200 for i in range(40)] for _ in range(10)]))
    r = run(env, p, rows=10, columns=40); d = document(env,r)
    assert len(d['cells']) == 400
    assert len(json.dumps(r.to_dict(), ensure_ascii=False).encode()) < 2400
    assert r.data['summary']['next_cell'] is not None
    assert r.data['summary']['omitted_cells'] > 0


def test_hash_change_is_denied_without_publishing(env):
    r = run(env, file(env,workbook([[1]])), expected_sha256='0'*64)
    assert r.status in {'error','denied'}
    assert not env[1].artifacts(run_id=env[5])


@pytest.mark.parametrize('name', ['../bad.xlsx','/bad.xlsx','a/../b.xlsx','a\\b.xlsx','C:/b.xlsx'])
def test_unsafe_members_are_rejected_before_selection(env,name):
    p=bundle(env,[(name,b'x'),('good.xlsx',workbook([[1]]))])
    assert run(env,p,member='good.xlsx').status in {'error','denied'}


def test_link_and_case_collision_rejected(env):
    link=zipfile.ZipInfo('alias.xlsx');link.create_system=3;link.external_attr=(stat.S_IFLNK|0o777)<<16
    assert run(env,bundle(env,[(link,b'target')])).status in {'error','denied'}
    assert run(env,bundle(env,[('A.xlsx',b'x'),('a.xlsx',b'y')])).status in {'error','denied'}


def test_expansion_budget_checked_for_outer_and_inner_zip(env,monkeypatch):
    monkeypatch.setattr(env[0],'MAX_EXPANDED_BYTES',50)
    assert run(env,bundle(env,[('note.txt',b'x'*51)])).status in {'error','denied'}
    assert run(env,file(env,workbook([[1]]))).status in {'error','denied'}


def test_unauthorized_path_and_member_type_rejected(env,tmp_path):
    p=tmp_path/'outside.xlsx';p.write_bytes(workbook([[1]]))
    assert run(env,p).status in {'error','denied'}
    assert run(env,bundle(env,[('code.py',b'print(1)')]),member='code.py').status in {'error','denied'}


@pytest.mark.parametrize('kw',[{'rows':0},{'rows':21},{'columns':257},{'row_start':5001},{'sheet':'missing'}])
def test_invalid_or_unknown_selections_rejected(env,kw):
    assert run(env,file(env,workbook([[1]])),**kw).status in {'error','denied'}


def test_empty_workbook_has_no_inferred_samples(env):
    d=document(env,run(env,file(env,workbook([]))))
    assert d['whole_sheet_inspected'] is False
    assert all(c['value'] is None for c in d['cells'])


def test_archive_prefix_is_literal_not_glob(env):
    p=bundle(env,[('a/book.xlsx',workbook([[1]])),('b/book.xlsx',workbook([[2]]))])
    d=document(env,run(env,p,member_prefix='a/'))
    assert [m['path'] for m in d['members'] if m['selected']] == ['a/book.xlsx']
    d=document(env,run(env,p,member_prefix='*'))
    assert not any(m['selected'] for m in d['members'])


def test_xml_entities_are_rejected_without_artifacts(env):
    raw=workbook([['x']]);dest=io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as src,zipfile.ZipFile(dest,'w') as out:
        for name in src.namelist():
            b=src.read(name)
            if name=='xl/workbook.xml':
                b=b'<!DOCTYPE workbook [<!ENTITY unsafe "expanded">]>'+b.replace(b'name="Raw"',b'name="&unsafe;"')
            out.writestr(name,b)
    result=run(env,file(env,dest.getvalue()))
    assert result.status in {'error','denied'}
    assert not env[1].artifacts(run_id=env[5])


def test_catalog_registers_tool_but_default_profile_does_not_enable(tmp_path):
    from adj_partner.application import AgentService
    from adj_partner.profiles import get_profile
    service=AgentService(tmp_path/'runtime')
    assert 'workbook_inspect' in service.catalog.names()
    assert 'workbook_inspect' not in get_profile('adjuvant-routed')['tools']


def test_compact_header_survives_runtime_projection(env):
    from adj_partner.runtime.context import ContextView
    headers=['Col'+str(i) for i in range(33)]
    r=run(env,file(env,workbook([headers],True)),rows=1,columns=33)
    assert r.status=='ok',r.to_dict()
    # Host output fits below ContextView's full-result projection threshold.
    from adj_partner.tools.catalog import canonical_json
    assert len(canonical_json(r.to_dict()).encode()) < ContextView.PREVIEW_MAX_BYTES
    assert [v[1] for v in r.data['summary']['cells']] == headers
