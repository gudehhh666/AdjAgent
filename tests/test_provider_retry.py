import httpx
import pytest

from adj_partner.models.base import ProviderFailure
from adj_partner.models.openai_compatible import OpenAICompatiblePort

OK = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
      "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


def port(handler, **kwargs):
    waits = []
    made = OpenAICompatiblePort(api_key="synthetic-host-credential", base_url="https://fixture.invalid/v1",
                                model="fixture", transport=httpx.MockTransport(handler), sleep=waits.append, **kwargs)
    return made, waits


def call(made, timeout=60):
    return made.complete([{"role": "user", "content": "hi"}], [], max_output_tokens=16, timeout_seconds=timeout)


def scripted(*outcomes):
    calls = []

    def handler(request):
        outcome = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(request)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return handler, calls


@pytest.mark.parametrize("first", [
    httpx.ConnectError("refused"),
    httpx.ConnectTimeout("no route"),
    httpx.Response(429, headers={"retry-after": "2"}, json={"error": "rate"}),
    httpx.Response(503, json={"error": "overloaded"}),
])
def test_unprocessed_failures_are_retried_and_logged(first):
    handler, calls = scripted(first, httpx.Response(200, json=OK))
    made, waits = port(handler)
    assert call(made).content == "done"
    assert len(calls) == 2 and len(made.retry_log) == 1 and len(waits) == 1
    if isinstance(first, httpx.Response) and first.status_code == 429:
        assert waits == [2.0] and made.retry_log[0]["reason"] == "http_429"
    made.close()


@pytest.mark.parametrize("failure, code", [
    (httpx.ReadTimeout("slow"), "provider_timeout"),
    (httpx.Response(502, json={"error": "gateway"}), "provider_http"),
    (httpx.Response(500, json={"error": "server"}), "provider_http"),
    (httpx.Response(400, json={"error": "bad"}), "provider_http"),
])
def test_possibly_processed_failures_are_never_retried(failure, code):
    handler, calls = scripted(failure, httpx.Response(200, json=OK))
    made, waits = port(handler)
    with pytest.raises(ProviderFailure) as raised:
        call(made)
    assert raised.value.code == code and len(calls) == 1 and waits == [] and made.retry_log == []
    made.close()


def test_retries_stop_at_the_limit_and_say_so():
    handler, calls = scripted(httpx.ConnectError("refused"))
    made, waits = port(handler, max_retries=2)
    with pytest.raises(ProviderFailure, match="gave up after 3 attempts") as raised:
        call(made)
    assert raised.value.code == "provider_connection" and len(calls) == 3 and len(waits) == 2
    made.close()


def test_retries_never_extend_past_the_callers_timeout():
    handler, calls = scripted(httpx.Response(429, headers={"retry-after": "20"}, json={}), httpx.Response(200, json=OK))
    made, waits = port(handler)
    with pytest.raises(ProviderFailure, match=r"\(429\)"):
        call(made, timeout=5)
    assert len(calls) == 1 and waits == []
    made.close()
