"""Real native file inspection of GEO's nonrectangular metadata format."""
import json
from pathlib import Path
import pytest
from adj_partner.application import AgentService
from adj_partner.contracts import ToolCall
from adj_partner.models import ScriptedPort

def inspect(tmp_path,text,suffix='.tsv'):
    p=tmp_path/('metadata'+suffix);p.write_text(text)
    service=AgentService(tmp_path/'state',project_root=Path(__file__).resolve().parents[1],model=ScriptedPort([]))
    sid=service.create_session(allowed_roots=[str(tmp_path)],allowed_tools=['file_inspect'])['id']
    from adj_partner.tools.policy import SessionToolset
    SessionToolset(service.store,service.catalog,sid).enable('file_inspect')
    run=service.submit(sid,'Inspect public fixture')
    result=service.executor.execute(ToolCall('inspect','file_inspect',{'path':str(p)}),sid,run['id'])
    service.close()
    return result

VALID='\n!Series_title\t"Example"\n!Sample_geo_accession\t"GSM1"\t"GSM2"\n!Sample_characteristics_ch1\t"time: 0"\t"time: 1"\n!Sample_characteristics_ch1\t"tissue: X"\t"tissue: Y"\n'

@pytest.mark.parametrize('suffix',['.tsv','.txt'])
def test_geo_nonrectangular_metadata_is_inspected(tmp_path,suffix):
    r=inspect(tmp_path,VALID,suffix)
    assert r.status=='ok',r.to_dict()
    d=r.data['inspection']
    assert d['format']=='geo_series_matrix_metadata'
    assert d['sample_columns']==2
    assert d['unique_sample_accessions']==2
    assert d['expression_values_inspected'] is False
    fields=[f for f in d['sample_fields'] if f['name']=='!Sample_characteristics_ch1']
    assert [f['occurrence'] for f in fields]==[1,2]
    assert fields[1]['value_examples']==['tissue: X','tissue: Y']

@pytest.mark.parametrize('bad',[
 VALID.replace('"GSM1"\t"GSM2"','"GSM1"'),
 VALID.replace('!Sample_geo_accession','!Sample_title'),
 VALID+'unmarked\tgarbage\n',
 VALID+'!Sample_geo_accession\t"GSM1"\t"GSM2"\n',
 VALID.replace('"GSM1"','""'),
])
def test_malformed_geo_not_silently_accepted(tmp_path,bad):
    r=inspect(tmp_path,bad)
    assert r.status=='error',r.to_dict()


def test_matrix_body_is_explicitly_not_inspected(tmp_path):
    r=inspect(tmp_path,VALID+'!series_matrix_table_begin\n"ID_REF"\t"GSM1"\t"GSM2"\na\t1\t2\n!series_matrix_table_end\n')
    assert r.status=='ok',r.to_dict()
    assert r.data['inspection']['matrix_body_present'] is True
    assert r.data['inspection']['expression_values_inspected'] is False


def test_duplicate_accessions_are_reported_not_independent_samples(tmp_path):
    r=inspect(tmp_path,VALID.replace('GSM2','GSM1'))
    assert r.status=='ok',r.to_dict()
    assert r.data['inspection']['duplicate_sample_accessions']==1


def test_wide_metadata_preview_is_bounded(tmp_path):
    text='!Series_title\t"Example"\n!Sample_geo_accession\t'+ '\t'.join(f'GSM{i}' for i in range(900))+'\n'
    text+=''.join('!Sample_characteristics_ch1\t'+'\t'.join('long description '*20 for _ in range(900))+'\n' for _ in range(3))
    r=inspect(tmp_path,text)
    assert r.status=='ok',r.to_dict()
    assert r.data['inspection']['sample_columns']==900
    assert len(json.dumps(r.data['inspection']).encode())<6000


def test_ordinary_bad_tsv_remains_invalid(tmp_path):
    assert inspect(tmp_path,'a\tb\n1\n').status=='error'
