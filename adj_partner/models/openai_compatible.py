"""One synchronous installed-SDK call; runtime owns budgeting.

The port retries only failures the provider provably did not process: the connection was never
established, or the provider rejected the request as rate-limited/overloaded. Read timeouts, gateway
errors and anything after the request may have been processed are never retried, because remote
execution and cost would be unknown. Retries stay inside the caller's timeout.
"""

import random
import time

import httpx
from openai import OpenAI, APIConnectionError, APIStatusError, APITimeoutError

UNPROCESSED_STATUSES = frozenset({429, 503, 529})
MAX_RETRIES = 3
MAX_WAIT_SECONDS = 30.0

from .base import ProviderFailure
from .configuration import configuration_failure, from_environment, validate_configuration
from .projection import cost_unknown, field, project_reply, public_usage, response_body
from .requests import public_request
from .transport import BoundedTransport


class _ExplicitHeadersOpenAI(OpenAI):
    @property
    def default_headers(self):
        # SDK 3.3.1 imports OPENAI_CUSTOM_HEADERS implicitly. Do not let ambient
        # headers override the explicitly configured credential or add secrets.
        return {"Accept": "application/json", "Accept-Encoding": "identity", "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"}


class OpenAICompatiblePort:
    def __init__(self, *, api_key, base_url, model, transport=None, sleep=time.sleep, max_retries=MAX_RETRIES):
        validate_configuration(api_key, base_url, model)
        self.secrets = (api_key,)
        self._model = model
        self._sleep, self._max_retries = sleep, max_retries
        self.retry_log = []
        self._http = None
        try:
            self._http = httpx.Client(transport=BoundedTransport(transport), trust_env=False, follow_redirects=False)
            self._client = _ExplicitHeadersOpenAI(
                api_key=api_key, base_url=base_url, admin_api_key="", organization="", project="", webhook_secret="",
                max_retries=0, http_client=self._http)
        except Exception:
            if self._http is not None:
                self._http.close()
            raise configuration_failure() from None

    @classmethod
    def from_environment(cls, *, project_root, env_file=None, model=None):
        return cls(**from_environment(project_root, env_file, model))

    @property
    def metadata(self):
        model = self._model
        for secret in self.secrets:
            if secret:
                model = model.replace(secret, "[REDACTED]")
        return {"provider_kind": "openai_compatible", "model": model}

    def close(self):
        self._client.close()

    def complete(self, messages, tools, *, max_output_tokens, timeout_seconds, tool_choice='auto'):
        if tool_choice != 'auto':
            raise ProviderFailure('unsupported_tool_choice', 'OpenAI-compatible port supports only automatic tool choice',
                                  cost_unknown=False)
        request = public_request(messages, tools, max_output_tokens, timeout_seconds)
        deadline = time.monotonic() + timeout_seconds
        attempt = 0
        while True:
            remaining = timeout_seconds if attempt == 0 else max(deadline - time.monotonic(), 0.001)
            try:
                arguments = {"model": self._model, "messages": request["messages"],
                             "max_completion_tokens": max_output_tokens, "timeout": remaining}
                if request["tools"]:
                    arguments["tools"] = request["tools"]
                response = self._client.chat.completions.with_raw_response.create(**arguments)
                body = response_body(response.content)
            except Exception as error:
                wait = self._retry_wait(error, attempt)
                if wait is None or attempt >= self._max_retries or wait >= deadline - time.monotonic():
                    self._raise(error, attempt)
                self.retry_log.append({"attempt": attempt + 1, "reason": self._unprocessed_reason(error),
                                       "waited_seconds": round(wait, 3)})
                self._sleep(wait)
                attempt += 1
                continue
            return project_reply(body)

    @staticmethod
    def _unprocessed_reason(error):
        if isinstance(error, APIStatusError) and error.status_code in UNPROCESSED_STATUSES:
            return f"http_{error.status_code}"
        if isinstance(error, APIConnectionError) and isinstance(error.__cause__, (httpx.ConnectError, httpx.ConnectTimeout)):
            return "connect_failed"
        return None

    def _retry_wait(self, error, attempt):
        if self._unprocessed_reason(error) is None:
            return None
        if isinstance(error, APIStatusError):
            try:
                hinted = float(error.response.headers.get("retry-after", ""))
                if 0 <= hinted <= MAX_WAIT_SECONDS:
                    return hinted
            except (TypeError, ValueError):
                pass
        return min(MAX_WAIT_SECONDS, 2 ** attempt) * (0.5 + random.random() / 2)

    def _raise(self, error, attempt):
        usage = public_usage(field(getattr(error, "body", None), "usage"))
        if isinstance(error, APIStatusError):
            try:
                outer_usage = public_usage(response_body(error.response.content).get("usage"))
                if outer_usage:
                    usage = outer_usage
            except (ValueError, TypeError, RecursionError):
                pass
        if isinstance(error, (APITimeoutError, httpx.TimeoutException)):
            code, message = "provider_timeout", "Provider request timed out; remote execution and cost may be unknown"
        elif isinstance(error, APIStatusError):
            code, message = "provider_http", f"Provider returned an unsuccessful HTTP status ({error.status_code})"
        elif isinstance(error, APIConnectionError):
            code, message = "provider_connection", "Provider connection failed; remote execution and cost may be unknown"
        else:
            code, message = "provider_response", "Provider SDK could not return a valid response"
        if attempt:
            message += f"; gave up after {attempt + 1} attempts"
        raise ProviderFailure(code, message, usage=usage, cost_unknown=cost_unknown(usage)) from None
