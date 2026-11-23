from core.schemas import EvidenceRecord, ReliabilityReport


def test_evidence_record_defaults_and_dict():
    r = EvidenceRecord(claim="c", source_type="literature", source_id="PMID:1")
    d = r.to_dict()
    assert d["claim"] == "c"
    assert d["authority_weight"] == 0.5
    assert d["entities"] == [] and d["extra"] == {}
    assert d["retrieved_at"]  # timestamp populated


def test_reliability_report_dict_shape():
    rep = ReliabilityReport(hypothesis="H")
    d = rep.to_dict()
    for key in ("grounding", "consistency", "statistics", "reviewer_verdict",
                "confidence", "falsification"):
        assert key in d


def test_reliability_report_roundtrip():
    rep = ReliabilityReport(hypothesis="H", confidence={"total": 0.7})
    rebuilt = ReliabilityReport(**rep.to_dict())
    assert rebuilt.confidence["total"] == 0.7
