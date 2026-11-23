"""AI Hub pass-through gateway (Bearer APP-ID:APP-KEY?provider=...&timeout=...)."""

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlsplit

from ..cli_defaults import _explicit_values
from ..tools.executor import ToolFailure


CURL_EXECUTABLE = "/usr/bin/curl"
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_TOKEN = re.compile(r"[A-Za-z0-9._-]{1,256}")


def settings(project_root, env_file, names):
    """Process environment wins over the explicit dotenv file, as for other host keys."""
    try:
        selected = _explicit_values(project_root, env_file)
    except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
        raise ValueError("Gateway configuration is missing, malformed or unsafe") from None
    return {name: (os.environ.get(name) or selected.get(name) or "").strip() for name in names}


class GatewayClient:
    def __init__(self, app_id, app_key, provider, *, base_url=None, timeout_seconds=60):
        if not _TOKEN.fullmatch(app_id or "") or not _TOKEN.fullmatch(app_key or ""):
            raise ValueError(f"{provider} gateway APP-ID/APP-KEY is missing or malformed")
        if not _TOKEN.fullmatch(provider):
            raise ValueError("Gateway provider name is malformed")
        if not base_url:
            raise ValueError(f"{provider} gateway URL is missing; set ADJ_AIHUB_GATEWAY_URL")
        base_url = base_url.rstrip("/")
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Gateway base URL must be a plain http(s) origin")
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int)
                or not 1 <= timeout_seconds <= 120):
            raise ValueError("Gateway timeout must be an integer in 1..120 seconds")
        self.app_id, self.app_key, self.provider = app_id, app_key, provider
        self.base_url, self.timeout_seconds = base_url, timeout_seconds

    @property
    def secrets(self):
        return (self.app_key,)

    def post(self, path, payload, *, headers=(), failure_prefix="gateway"):
        """POST JSON; the Authorization header travels over stdin, never argv."""
        url = self.base_url + path
        budget = self.timeout_seconds + 5
        authorization = (f"Authorization: Bearer {self.app_id}:{self.app_key}"
                         f"?provider={self.provider}&timeout={self.timeout_seconds}\n")
        with tempfile.TemporaryDirectory(prefix="adjagent-gateway-") as directory:
            header_path = Path(directory) / "headers"
            arguments = [CURL_EXECUTABLE, "--silent", "--show-error", "--no-location", "--proto", "=http,https",
                         "--connect-timeout", "5", "--max-time", str(budget),
                         "--max-filesize", str(MAX_RESPONSE_BYTES), "--request", "POST",
                         "--header", "Accept: application/json", "--header", "Accept-Encoding: identity",
                         "--header", "Content-Type: application/json", "--header", "@-"]
            for header in headers:
                arguments += ["--header", header]
            arguments += ["--data-binary", json.dumps(payload, ensure_ascii=False),
                          "--dump-header", str(header_path), "--write-out", "\n%{http_code}", url]
            try:
                completed = subprocess.run(arguments, input=authorization.encode(), stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, timeout=budget + 2, check=False)
            except subprocess.TimeoutExpired as exc:
                raise ToolFailure(f"{failure_prefix}_timeout", "Gateway response budget exhausted") from exc
            except OSError as exc:
                raise ToolFailure(f"{failure_prefix}_transport", f"Gateway curl could not start: {type(exc).__name__}") from exc
            response_headers = {}
            if header_path.is_file():
                for line in header_path.read_text(encoding="iso-8859-1").splitlines()[1:]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        response_headers[key.strip().lower()] = value.strip()
        raw, separator, status = completed.stdout.rpartition(b"\n")
        if completed.returncode == 28:
            raise ToolFailure(f"{failure_prefix}_timeout", "Gateway response budget exhausted")
        if completed.returncode != 0 or not separator or len(status) != 3 or not status.isdigit():
            raise ToolFailure(f"{failure_prefix}_transport", f"Gateway curl failed without retry (exit {completed.returncode})")
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ToolFailure(f"{failure_prefix}_size", "Gateway response exceeds size limit")
        if status != b"200":
            detail = response_headers.get("x-error-data", "")[:300]
            raise ToolFailure(f"{failure_prefix}_http", f"Gateway returned HTTP {status.decode()}; no retry. {detail}".strip())
        return response_headers, raw
