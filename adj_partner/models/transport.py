"""Bound SDK response buffering and trickle duration without adding requests.

The same explicit SDK timeout bounds each blocking I/O operation. An operation
already blocked can overrun the elapsed deadline by that I/O timeout; this is
not an OS cancellation primitive and cannot establish a zero remote bill.
"""

import time

import httpx


MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class _WireStream(httpx.SyncByteStream):
    def __init__(self, response, request, deadline):
        self.response, self.request, self.deadline = response, request, deadline

    def __iter__(self):
        size = 0
        try:
            for chunk in self.response.stream:
                if time.monotonic() > self.deadline:
                    raise httpx.ReadTimeout("Provider response elapsed deadline exceeded", request=self.request)
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise httpx.ReadError("Provider response exceeds byte limit", request=self.request)
                yield chunk
        finally:
            self.close()

    def close(self):
        self.response.close()


class BoundedTransport(httpx.BaseTransport):
    def __init__(self, transport=None):
        self._transport = transport if transport is not None else httpx.HTTPTransport(retries=0, trust_env=False)

    def handle_request(self, request):
        timeout = request.extensions["timeout"]["read"]
        deadline = time.monotonic() + timeout
        response = self._transport.handle_request(request)
        try:
            if time.monotonic() > deadline:
                raise httpx.ReadTimeout("Provider response elapsed deadline exceeded", request=request)
            if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
                raise httpx.ReadError("Provider response must use identity encoding", request=request)
            length = response.headers.get("content-length")
            if length is not None and (not length.isascii() or not length.isdecimal()
                    or len(length) > 10 or int(length) > MAX_RESPONSE_BYTES):
                raise httpx.ReadError("Provider response exceeds byte limit", request=request)
            return httpx.Response(response.status_code, headers=response.headers, extensions=response.extensions,
                                  stream=_WireStream(response, request, deadline))
        except BaseException:
            response.close()
            raise

    def close(self):
        self._transport.close()
