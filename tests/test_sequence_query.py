import importlib.util
import json
import zipfile

import pytest

from adj_partner.contracts import Budget, SessionConfig, ToolCall
from adj_partner.native.sequence_inventory import register_sequence_inventory
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolExecutor


def fixture(tmp_path):
    cat=ToolCatalog();register_sequence_inventory(cat)
    assert "sequence_inventory_query" in cat.names(), "query tool required"
    store=Store(tmp_path/'state')
    sid=store.create_session(SessionConfig(allowed_roots=[str(tmp_path)],allowed_tools=cat.names(),enabled_tools=cat.names()).to_dict())['id']
    task=store.create_task(sid,'inventory query')['id'];rid=store.create_run(sid,task,Budget().to_dict())['id']
    exe=ToolExecutor(store,cat)
    p=tmp_path/'data.zip'
    with zipfile.ZipFile(p,'w') as z:
        z.writestr('single/A.faa','>different_header\nACD\n>different_header\nQQ\n')
        z.writestr('multi/A.faa','>x\nMMMMMMM\n')
        z.writestr('single/B.faa','>b\nNNNN\n')
    r=exe.execute(ToolCall('inventory','sequence_inventory',{'paths':[str(p)],'member_glob':'multi/*','scope_note':'initial subset'}),sid,rid)
    assert r.status=='ok',r.to_dict()
    return store,sid,rid,exe,p,r


def query(env,**kwargs):
    st,sid,rid,exe,_,original=env
    return exe.execute(ToolCall('query','sequence_inventory_query',dict(artifact_id=original.data['artifact_id'],member_glob='single/*',file_stems=['A'],scope_note='explicit re-selection',**kwargs)),sid,rid)


def test_query_uses_frozen_records_and_preserves_count_units(tmp_path):
    env=fixture(tmp_path);env[4].unlink()
    result=query(env)
    assert result.status=='ok',result.to_dict()
    assert result.data['counts']['after_stem_filter']==dict(files=1,records=2,residues=5,gaps=0,stops=0)
    assert result.data['duplicate_record_ids']==1
    detail=json.loads(env[0].read_artifact(result.data['artifact_id']))
    assert detail['parent']['artifact_id']==env[5].data['artifact_id']
    assert len(detail['selected_records'])==2
    assert detail['selected_records'][0]['filename_stem']=='A'
    assert detail['selected_records'][0]['record_id']=='different_header'


def test_initial_inventory_exposes_actual_field_examples(tmp_path):
    env=fixture(tmp_path);d=env[5].data
    assert 'directories' in d
    single=next(x for x in d['directories'] if x['path']=='single')
    assert single['filename_stem_examples']==['A','B']
    assert single['record_id_examples']==['b','different_header']
    assert len(json.dumps(env[5].to_dict()).encode())<2400


@pytest.mark.parametrize('kind',['other_run','wrong_provider','tampered','too_large'])
def test_query_requires_current_run_host_inventory(tmp_path,kind):
    env=fixture(tmp_path);st,sid,rid,exe,_,original=env
    meta=next(a for a in st.artifacts(sid,rid) if a['id']==original.data['artifact_id'])
    aid=meta['id']
    if kind=='other_run':
        task=st.create_task(sid,'other')['id'];rid=st.create_run(sid,task,Budget().to_dict())['id']
    elif kind=='wrong_provider':
        aid=st.put_artifact(st.read_artifact(aid),'pretend.json','application/json',{'provider':'analysis'},sid,rid)['id']
    elif kind=='tampered':
        from pathlib import Path
        Path(meta['path']).write_text('{}')
    else:
        assert importlib.util.find_spec('adj_partner.native.sequence_query')
        from adj_partner.native import sequence_query
        sequence_query.MAX_INVENTORY_BYTES=1
    result=exe.execute(ToolCall('bad','sequence_inventory_query',dict(artifact_id=aid,member_glob='*',scope_note='check')),sid,rid)
    if kind=='too_large':sequence_query.MAX_INVENTORY_BYTES=32*1024*1024
    assert result.status in {'denied','error'},result.to_dict()


@pytest.mark.parametrize('selection', [dict(file_stems=[]),dict(source_indices=[]),dict(file_stems=['different_header'])])
def test_empty_and_non_filename_selection_are_zero(tmp_path,selection):
    env=fixture(tmp_path);st,sid,rid,exe,_,r=env
    args=dict(artifact_id=r.data['artifact_id'],member_glob='*',scope_note='empty',**selection)
    out=exe.execute(ToolCall('q','sequence_inventory_query',args),sid,rid)
    assert out.status=='ok',out.to_dict()
    assert out.data['counts']['after_stem_filter']['records']==0
