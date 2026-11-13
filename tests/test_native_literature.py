"""Deterministic actual HTTP-client fixtures; these are not live literature searches."""

import importlib
import importlib.util
import json

import httpx
import pytest


def api():
    assert importlib.util.find_spec("adj_partner.native.literature"), "native literature adapter must exist"
    return importlib.import_module("adj_partner.native.literature")


def payload(records=None):
    return {"hitCount": 1 if records is None else len(records), "resultList": {"result": records if records is not None else [{
        "id": "123456", "source": "MED", "title": "Synthetic title", "abstractText": "<p>Fixture <b>abstract</b>.</p>",
        "authorString": "Fixture A", "pubYear": "2024", "doi": "10.1234/fixture", "pmcid": "PMC12345"}]}}


def client(handler):
    return api().EuropePMC(transport=httpx.MockTransport(handler))


def test_fixed_request_and_actual_available_text_have_provenance():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload())
    response = client(handler).search('ferritin AND TITLE:"aging"', page_size=3)
    assert len(requests) == 1
    assert requests[0].url.host == "www.ebi.ac.uk" and requests[0].url.path.endswith("/search")
    assert requests[0].url.params["resultType"] == "core" and requests[0].url.params["pageSize"] == "3"
    assert requests[0].headers["accept-encoding"] == "identity"
    assert response.hit_count == 1 and response.request_url == str(requests[0].url)
    assert response.raw == json.dumps(payload(), separators=(",", ":")).encode()
    assert response.received_at
    record = response.records[0]
    assert record["id"] == "123456" and record["source"] == "MED"
    assert record["abstract"] == "Fixture abstract."
    assert record["text_availability"] == "abstract"
    assert record["url"] == "https://europepmc.org/article/MED/123456"
    assert record["pmcid"] == "PMC12345" and record["doi"] == "10.1234/fixture"


def test_empty_result_is_success_distinct_from_provider_failure():
    result = client(lambda _: httpx.Response(200, json=payload([]))).search("no fixture matches")
    assert result.records == [] and result.hit_count == 0


def test_title_only_is_never_labeled_full_paper():
    item = {"id": "123", "source": "MED", "title": "Fixture title", "isOpenAccess": "Y", "pmcid": "PMC123"}
    result = client(lambda _: httpx.Response(200, json=payload([item]))).search("fixture")
    assert result.records[0]["text_availability"] == "title_only"
    assert result.records[0]["abstract"] == ""


@pytest.mark.parametrize("body", [{}, {"error": "failure"}, {"hitCount": 0, "resultList": []},
    {"hitCount": "one", "resultList": {"result": []}}, payload([{"title": "Missing identifier"}]),
    payload([{"id": "../private", "source": "MED", "title": "Bad identifier"}]),
    payload([{"id": "123", "source": "MED", "title": 7}])])
def test_malformed_response_is_a_typed_failure(body):
    from adj_partner.tools.executor import ToolFailure
    with pytest.raises(ToolFailure) as error:
        client(lambda _: httpx.Response(200, json=body)).search("fixture")
    assert error.value.code == "literature_response"


@pytest.mark.parametrize("status", [301, 302, 401, 429, 500])
def test_http_errors_and_redirects_do_not_become_empty_evidence_or_new_origins(status):
    from adj_partner.tools.executor import ToolFailure
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"location": "https://invalid.example/private"}, text="Fixture error")
    with pytest.raises(ToolFailure) as error:
        client(handler).search("fixture")
    assert error.value.code == "literature_http" and len(calls) == 1


def test_invalid_json_and_oversized_or_encoded_body_are_rejected():
    from adj_partner.tools.executor import ToolFailure
    for headers, content in [({}, b"not JSON"), ({}, b"x" * (2 * 1024 * 1024 + 1)),
                             ({"content-encoding": "gzip"}, b"not compressed")]:
        # A streaming response exercises the adapter's byte bound before parsing.
        def handler(_, headers=headers, content=content):
            return httpx.Response(200, headers=headers, stream=httpx.ByteStream(content))
        with pytest.raises(ToolFailure):
            client(handler).search("fixture")


def test_timeout_has_no_hidden_retry():
    from adj_partner.tools.executor import ToolFailure
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("Fixture timeout", request=request)
    with pytest.raises(ToolFailure) as error:
        client(handler).search("fixture")
    assert error.value.code == "literature_transport" and len(calls) == 1


@pytest.mark.parametrize("query,page_size", [("", 5), ("x" * 2001, 5), ("fixture", 0), ("fixture", 11), ("fixture", True)])
def test_queries_and_pages_are_bounded_before_network(query, page_size):
    calls = []
    with pytest.raises(ValueError):
        client(lambda request: calls.append(request)).search(query, page_size=page_size)
    assert calls == []


@pytest.mark.parametrize("field,value", [("abstractText", 0), ("abstractText", []), ("authorString", False), ("pubYear", 0)])
def test_explicit_nontext_metadata_is_not_silently_treated_as_missing(field, value):
    from adj_partner.tools.executor import ToolFailure
    body = payload()
    body["resultList"]["result"][0][field] = value
    with pytest.raises(ToolFailure) as error:
        client(lambda _: httpx.Response(200, json=body)).search("fixture")
    assert error.value.code == "literature_response"


def test_oversized_content_length_header_is_a_typed_failure():
    from adj_partner.tools.executor import ToolFailure
    with pytest.raises(ToolFailure) as error:
        client(lambda _: httpx.Response(200, headers={"content-length": "9" * 5000}, stream=httpx.ByteStream(b"{}"))).search("fixture")
    assert error.value.code == "literature_size"


def test_cancellation_checked_on_each_incoming_chunk_not_after_accumulating_64k():
    from adj_partner.tools.policy import ToolCancelled
    consumed = []
    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            for index in range(3000):
                consumed.append(index)
                yield b" " * 32
    def cancelled():
        if len(consumed) >= 2:
            raise ToolCancelled("Synthetic cancellation")
    with pytest.raises(ToolCancelled):
        client(lambda _: httpx.Response(200, stream=Drip())).search("fixture", check_cancelled=cancelled)
    assert len(consumed) == 2


def test_response_wall_budget_is_checked_even_when_each_network_read_succeeds(monkeypatch):
    from adj_partner.tools.executor import ToolFailure
    module = api()
    clock = [0.0]
    def now():
        clock[0] += 0.1
        return clock[0]
    monkeypatch.setattr("time.monotonic", now)
    class Drip(httpx.SyncByteStream):
        def __iter__(self):
            for _ in range(100):
                yield b" "
    with pytest.raises(ToolFailure) as error:
        module.EuropePMC(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Drip())), timeout_seconds=0.25).search("fixture")
    assert error.value.code == "literature_timeout"
