import json
from adj_partner.research.qa_review_context import select_evidence


def source(content):
    return {'id':'s1','kind':'literature','provenance':{'url':'https://example.org/paper','text_availability':'derived_full_text'},'document':content}


def test_late_relevant_paragraph_survives_context_budget():
    doc={'title':'10E8 study','sections':[{'paragraphs':[{'text':'Unrelated introduction. '*1500},{'locator':'results:1','text':'10E8-B1 affinity increased after macaque immunization. W3-01 showed HCDR3 maturation.'}]}]}
    packet=select_evidence('10E8 macaque BCR maturation','Binding to 10E8-B1 and W3-01 structure.',[source(doc)],budget=2500)
    assert '10E8-B1 affinity increased' in json.dumps(packet)
    assert len(json.dumps(packet,ensure_ascii=False))<=2500


def test_figures_retain_caption_provenance_not_visual_claim():
    doc={'figures':[{'locator':'figure:1','caption':'S2P6: SARS-CoV-2 IC50 1.4, PANG/GD 0.02.'}]}
    packet=select_evidence('S2P6 neutralization','',[source(doc)])
    assert packet[0]['excerpts'][0]['locator']=='figure:1'
    assert packet[0]['excerpts'][0]['evidence_type']=='caption_text'


def test_duplicate_sources_and_reference_lists_do_not_fill_budget():
    doc={'abstract':'Relevant adjuvant result.','references':[{'title':'Unrelated literature SECRET'}]}
    packet=select_evidence('adjuvant','',[source(doc),source(doc)])
    assert len(packet)==1
    assert 'SECRET' not in json.dumps(packet)


def test_search_snippets_are_preserved_without_becoming_fulltext():
    s={'id':'web1','kind':'web','provenance':{'url':'https://example.org/paper','text_availability':'search_snippet'},
       'document':{'snippet':'sRCCs retained 22.7% versus eRCCs 10.8% protein.','title':'Paper'}}
    packet=select_evidence('sRCCs eRCCs protein retention','',[s])
    assert len(packet)==1
    assert packet[0]['text_availability']=='search_snippet'
    assert packet[0]['excerpts'][0]['evidence_type']=='search_snippet'
    assert '22.7%' in packet[0]['excerpts'][0]['text']
