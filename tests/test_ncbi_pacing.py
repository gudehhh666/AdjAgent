import httpx

from adj_partner.native import ncbi
from adj_partner.native.grounding import GroundingRetriever
from adj_partner.native.scholarly import PubMedClient


def test_pubmed_client_and_grounding_share_one_ncbi_pace(monkeypatch):
    clock, sleeps = [100.0], []
    monkeypatch.setattr(ncbi.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(ncbi.time, 'sleep', lambda s: (sleeps.append(round(s, 2)), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(ncbi, '_last', 0.0)

    def handler(request):
        return httpx.Response(200, json={'esearchresult': {'count': '0', 'idlist': []}})

    transport = httpx.MockTransport(handler)
    PubMedClient(transport=transport).search('alum')
    GroundingRetriever(transport=transport).pubmed('alum')
    PubMedClient(transport=transport, api_key='k').search('alum')
    assert sleeps == [0.36, 0.11]
