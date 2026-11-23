import pytest
from adj_partner.evaluation_acceptance import build_manifest, validate_metadata, summarize_results


def test_manifest_is_stable_and_never_leaks_labels():
    rows=[{'id':str(i),'question':f'Question {i}','answer':'SECRET','Reasoning_Process:':'SECRET'} for i in range(20)]
    a=build_manifest(rows,'qa',10)
    b=build_manifest(list(reversed(rows)),'qa',10)
    assert a==b
    assert 'SECRET' not in str(a)
    assert {r['split'] for r in a['items']}=={'development','holdout'}
    assert all(r['expert_review'] is None for r in a['items'])
    with pytest.raises(ValueError,match='Duplicate'):
        build_manifest(rows+[rows[0]],'qa',10)


def test_metadata_missing_replicates_and_contrast_detected():
    issues=validate_metadata({'dataset_id':'x','samples':[{'sample_id':'s1','condition':'old','biological_unit':'u1'}], 'contrasts':[{'name':'x','treatment':'missing','control':'old'}]})
    assert any('missing' in x for x in issues)
    assert any('replicate' in x for x in issues)


def test_result_summary_preserves_unknown_and_separates_modes():
    values=[{'task_fingerprint':'a','task_type':'qa','execution_status':'succeeded','review_status':'passed','usage':{'requests':3}}, {'task_fingerprint':'b','task_type':'design','execution_status':'waiting','review_status':'revision_required','usage':{}}]
    result=summarize_results(values)
    assert result['qa']['succeeded']==1 and result['design']['waiting']==1
    assert result['design']['request_count_missing']==1
    assert result['qa']['expert_acceptance_rate'] is None
    with pytest.raises(ValueError,match='Duplicate'):
        summarize_results(values+[values[0]])


def test_manifest_rejects_structured_identifier_leak():
    with pytest.raises(ValueError,match='Task ID'):
        build_manifest([{'id':{'answer':'private'},'question':'Public?'}],'qa',1)
