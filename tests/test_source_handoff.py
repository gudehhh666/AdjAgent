import pytest
from adj_partner.research import records


def test_handoff_preserves_bytes_and_checks_task_scope(tmp_path):
    from test_harness_research_records import setup,source
    assert hasattr(records.Records,'handoff_source')
    store,r,sid,rid,task=setup(tmp_path)
    s,_=source(store,r,sid,rid,content='Exact public material.',provenance={'text_availability':'full_text'})
    parent=store.get_run(rid);new=store.create_run(sid,parent['task_id'],parent['budget'])
    alias=r.handoff_source(sid,new['id'],s['id'])
    assert alias['run_id']==new['id'] and alias['id']!=s['id']
    assert alias['artifact_hash']==s['artifact_hash']
    assert store.read_artifact(alias['artifact_id'])==store.read_artifact(s['artifact_id'])
    assert alias['artifact_provenance']['handoff_source_id']==s['id']
    assert r.handoff_source(sid,new['id'],s['id'])['id']==alias['id']
    foreign=store.create_task(sid,'Unrelated',{});other=store.create_run(sid,foreign['id'],parent['budget'])
    with pytest.raises(ValueError):r.handoff_source(sid,other['id'],s['id'])
