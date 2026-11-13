from adj_partner.research import context_index


def test_host_progress_preserves_time_for_synthesis_without_certifying_evidence():
    assert hasattr(context_index,'research_progress')
    run={'state':{'steps':5},'usage':{},'budget':{'max_steps':64,'max_input_tokens':1000,'max_output_tokens':1000}}
    refs={'sources':[{'id':'s'}],'evidence':[],'claims':[]}
    assert context_index.research_progress(run,refs)['next_stage']=='select_evidence'
    refs['evidence']=[{'id':'e'}]
    assert context_index.research_progress(run,refs)['next_stage']=='record_claims'
    run['usage']['accounted_input_tokens']=700
    p=context_index.research_progress(run,refs)
    assert p['next_stage']=='synthesize_or_report_gaps'
    assert p['scientific_adequacy']=='not_assessed'
    assert p['budget_fraction_used']==0.7
    assert 'not a minimum-source quality verdict' in p['guidance']
