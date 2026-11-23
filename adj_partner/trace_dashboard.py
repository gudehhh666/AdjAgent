"""Read-only observability API for exported AdjAgent research trajectories."""

from __future__ import annotations

import csv
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import re
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .profiles import PROFILES


_NUMBER = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
_REQUIRED_FILES = (
    "quality-analysis.json",
    "quality-cases.csv",
    "quality-attempts.csv",
    "quality-tools-by-level.csv",
    "delivery-index.json",
)
_ATTEMPT = re.compile(r"^L[123]/qa_[0-9]+(?:-retry-[0-9]+)?$")


class DashboardRequestError(ValueError):
    """A safe client-visible dashboard request failure."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def _parse_cell(value: str):
    text = value.strip()
    if text == "True":
        return True
    if text == "False":
        return False
    if text.startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    if _NUMBER.fullmatch(text):
        return float(text) if "." in text else int(text)
    return value


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{key: _parse_cell(value) for key, value in row.items()}
                for row in csv.DictReader(handle)]


class TraceDashboardData:
    """Validated view over one machine-readable batch analysis directory."""

    def __init__(self, batch_root: Path | str):
        self.root = Path(batch_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Batch root does not exist: {self.root}")
        for name in _REQUIRED_FILES:
            path = self.root / name
            if not path.is_file():
                raise FileNotFoundError(f"Required dashboard input is missing: {name}")
        with (self.root / "quality-analysis.json").open(encoding="utf-8") as handle:
            self.analysis = json.load(handle)
        self._cases = _read_csv(self.root / "quality-cases.csv")
        self._attempts = _read_csv(self.root / "quality-attempts.csv")
        self._tools = _read_csv(self.root / "quality-tools-by-level.csv")
        self._attempt_roots = {}
        for row in self._attempts:
            attempt = row.get("attempt")
            if not isinstance(attempt, str) or not _ATTEMPT.fullmatch(attempt):
                raise ValueError(f"Invalid attempt identifier in quality-attempts.csv: {attempt!r}")
            attempt_root = (self.root / attempt).resolve()
            if not attempt_root.is_relative_to(self.root):
                raise ValueError(f"Attempt escapes batch root: {attempt}")
            self._attempt_roots[attempt] = attempt_root

    def health(self) -> dict:
        scope = self.analysis.get("scope", {})
        return {
            "status": "ok",
            "schema_version": self.analysis.get("schema_version"),
            "generated_at": self.analysis.get("generated_at"),
            "model": self.analysis.get("model"),
            "cases": len(self._cases),
            "attempts": len(self._attempts),
            "tool_calls": scope.get("all_tool_calls", sum(int(row.get("tool_calls", 0)) for row in self._attempts)),
            "batch": self.root.name,
        }

    def overview(self) -> dict:
        scope = self.analysis.get("scope", {})
        all_tools = [row for row in self._tools if row.get("scope") == "all_attempts"]
        claims = sum(int(row.get("claims", 0)) for row in self._cases)
        fully_bound = sum(int(row.get("fully_bound_claims", 0)) for row in self._cases)
        levels = {}
        for level, details in self.analysis.get("level_summary", {}).items():
            words = details.get("report_metrics", {}).get("words", {})
            levels[level] = {
                "cases": details.get("cases", 0),
                "attempts": details.get("attempts", 0),
                "successes": details.get("selected_successes", 0),
                "tool_calls": details.get("all_tool_calls", 0),
                "selected_tool_calls": details.get("selected_tool_calls", 0),
                "report_words_median": words.get("median"),
                "attempt_statuses": details.get("attempt_statuses", {}),
                "stop_reasons": details.get("stop_reasons", {}),
            }
        profiles = {name: list(dict.fromkeys(definition["tools"]))
                    for name, definition in PROFILES.items()}
        registered_tools = sorted({tool for tools in profiles.values() for tool in tools})
        return {
            "batch": self.root.name,
            "model": self.analysis.get("model"),
            "generated_at": self.analysis.get("generated_at"),
            "totals": {
                "cases": scope.get("selected_cases", len(self._cases)),
                "attempts": scope.get("attempt_directories", len(self._attempts)),
                "manifested_attempts": scope.get("manifested_attempts"),
                "tool_calls": scope.get("all_tool_calls", sum(int(row.get("calls", 0)) for row in all_tools)),
                "selected_tool_calls": scope.get("selected_tool_calls"),
                "completed_tool_calls": sum(int(row.get("completed", 0)) for row in all_tools),
                "failed_tool_calls": sum(int(row.get("failed", 0)) for row in all_tools),
                "unfinished_tool_calls": sum(int(row.get("started_or_other", 0)) for row in all_tools),
                "claims": claims,
                "fully_bound_claims": fully_bound,
                "unbound_claims": sum(int(row.get("unbound_claims", 0)) for row in self._cases),
                "delivery_passes": sum(1 for row in self._cases if row.get("gate_delivery_gates_pass") is True),
            },
            "levels": levels,
            "categories": [row for row in self.analysis.get("category_distribution", [])
                           if row.get("scope") == "all_attempts"],
            "registered_tools": registered_tools,
            "profiles": profiles,
        }

    def cases(self) -> list[dict]:
        rows = []
        for source in self._cases:
            row = dict(source)
            row["delivery_pass"] = row.get("gate_delivery_gates_pass") is True
            rows.append(row)
        return rows

    def attempts(self) -> list[dict]:
        return [dict(row) for row in self._attempts]

    def tools(self) -> list[dict]:
        return [dict(row) for row in self._tools]

    def _attempt_root(self, attempt: str) -> Path:
        root = self._attempt_roots.get(attempt)
        if root is None:
            raise DashboardRequestError(f"Unknown attempt: {attempt}", 404)
        return root

    def _safe_file(self, candidate: Path) -> Path:
        if candidate.is_symlink():
            raise DashboardRequestError("Requested file is not a safe public export", 403)
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise DashboardRequestError("Requested public export is unavailable", 404) from exc
        if not resolved.is_relative_to(self.root) or not resolved.is_file():
            raise DashboardRequestError("Requested file is not a safe public export", 403)
        return resolved

    def raw_export(self, attempt: str, kind: str) -> tuple[Path, str]:
        root = self._attempt_root(attempt)
        media = {
            "trace": "text/markdown; charset=utf-8",
            "jsonl": "application/x-ndjson; charset=utf-8",
            "report": "text/markdown; charset=utf-8",
        }
        if kind not in media:
            raise DashboardRequestError(f"Unsupported raw export: {kind}")
        if kind == "report":
            candidate = root / "report.md"
        else:
            manifest_path = self._safe_file(root / "manifest.json")
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                key = "markdown" if kind == "trace" else "jsonl"
                value = manifest["trace"][key]
                candidate = Path(value)
                if not candidate.is_absolute():
                    candidate = root / candidate
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise DashboardRequestError("Attempt has no readable public trace export", 404) from exc
        return self._safe_file(candidate), media[kind]

    @staticmethod
    def _duration_ms(start: object, finish: object):
        if not isinstance(start, str) or not isinstance(finish, str):
            return None
        try:
            left = datetime.fromisoformat(start.replace("Z", "+00:00"))
            right = datetime.fromisoformat(finish.replace("Z", "+00:00"))
        except ValueError:
            return None
        return max(0, round((right - left).total_seconds() * 1000, 3))

    @classmethod
    def _normalize_record(cls, record: dict, index: int) -> dict:
        record_type = record.get("record_type", "unknown")
        data = record.get("data") if isinstance(record.get("data"), dict) else {}
        common = {"type": record_type, "index": index, "raw": record}
        if record_type == "action":
            result = data.get("result") if isinstance(data.get("result"), dict) else {}
            return {
                **common,
                "id": data.get("id"),
                "call_id": data.get("call_id"),
                "tool": data.get("tool_name"),
                "status": data.get("status"),
                "timestamp": data.get("created_at"),
                "finished_at": data.get("finished_at"),
                "duration_ms": cls._duration_ms(data.get("created_at"), data.get("finished_at")),
                "args": data.get("args", {}),
                "result_status": result.get("status"),
                "error": result.get("error"),
                "artifacts": result.get("artifacts", []),
                "result_data": result.get("data"),
                "job_id": result.get("job_id"),
            }
        if record_type == "event":
            return {
                **common,
                "seq": data.get("seq"),
                "kind": data.get("kind"),
                "timestamp": data.get("created_at"),
                "payload": data.get("payload"),
            }
        if record_type == "message":
            message = data.get("message") if isinstance(data.get("message"), dict) else {}
            return {
                **common,
                "seq": data.get("seq"),
                "timestamp": data.get("created_at"),
                "role": message.get("role"),
                "content": message.get("content"),
                "tool_calls": message.get("tool_calls", []),
            }
        return {
            **common,
            "timestamp": data.get("created_at"),
            "id": data.get("id"),
            "summary": data,
        }

    def trajectory(self, attempt: str, *, offset: int = 0, limit: int = 50) -> dict:
        if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= 100:
            raise DashboardRequestError("Invalid trajectory pagination; require offset >= 0 and 1 <= limit <= 100")
        path, _ = self.raw_export(attempt, "jsonl")
        records = []
        with path.open(encoding="utf-8-sig") as handle:
            for index, line in enumerate(handle):
                text = line.rstrip("\r\n")
                try:
                    parsed = json.loads(text)
                    if not isinstance(parsed, dict):
                        raise json.JSONDecodeError("record is not an object", text, 0)
                except json.JSONDecodeError:
                    records.append({"type": "parse_error", "index": index,
                                    "error": "Malformed public trace JSON", "preview": text[:500]})
                    continue
                records.append(self._normalize_record(parsed, index))
        total = len(records)
        page = records[offset:offset + limit]
        next_offset = offset + len(page)
        if next_offset >= total:
            next_offset = None
        return {
            "attempt": attempt,
            "offset": offset,
            "limit": limit,
            "total": total,
            "next_offset": next_offset,
            "records": page,
        }


def create_server(data: TraceDashboardData, static_root: Path | str, *,
                  host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """Create a loopback-oriented read-only dashboard server."""

    root = Path(static_root).expanduser().resolve()
    static_files = {
        "/": root / "index.html",
        "/index.html": root / "index.html",
        "/styles.css": root / "styles.css",
        "/app.js": root / "app.js",
    }
    for path in static_files.values():
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise FileNotFoundError(f"Dashboard static asset is missing or unsafe: {path.name}")

    class Handler(BaseHTTPRequestHandler):
        server_version = "AdjAgentTraceDashboard/1"

        def log_message(self, format, *args):
            return

        @staticmethod
        def _single_query(query: dict[str, list[str]], allowed: set[str]) -> dict[str, str]:
            unexpected = set(query) - allowed
            if unexpected:
                raise DashboardRequestError(
                    "Unexpected query parameter(s): " + ", ".join(sorted(unexpected)))
            values = {}
            for key, items in query.items():
                if len(items) != 1 or items[0] == "":
                    raise DashboardRequestError(f"Query parameter must have one value: {key}")
                values[key] = items[0]
            return values

        def _headers(self, content_type: str, length: int, *, api: bool = False):
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(length))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store" if api else "no-cache")
            self.send_header("Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; style-src 'self'; "
                "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; "
                "base-uri 'none'; form-action 'none'")

        def _send_bytes(self, status: int, body: bytes, content_type: str, *, api: bool = False):
            self.send_response(status)
            self._headers(content_type, len(body), api=api)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _send_json(self, status: int, payload):
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            self._send_bytes(status, body, "application/json; charset=utf-8", api=True)

        def _api(self, path: str, query: dict[str, list[str]]):
            if path in {"/api/health", "/api/overview", "/api/cases", "/api/attempts", "/api/tools"}:
                self._single_query(query, set())
                payload = {
                    "/api/health": data.health,
                    "/api/overview": data.overview,
                    "/api/cases": data.cases,
                    "/api/attempts": data.attempts,
                    "/api/tools": data.tools,
                }[path]()
                self._send_json(200, payload)
                return
            if path == "/api/trajectory":
                values = self._single_query(query, {"attempt", "offset", "limit"})
                if "attempt" not in values:
                    raise DashboardRequestError("Missing query parameter: attempt")
                try:
                    offset = int(values.get("offset", "0"))
                    limit = int(values.get("limit", "50"))
                except ValueError as exc:
                    raise DashboardRequestError("Trajectory offset and limit must be integers") from exc
                self._send_json(200, data.trajectory(values["attempt"], offset=offset, limit=limit))
                return
            if path == "/api/raw":
                values = self._single_query(query, {"attempt", "kind"})
                if set(values) != {"attempt", "kind"}:
                    raise DashboardRequestError("Raw export requires attempt and kind")
                file_path, media_type = data.raw_export(values["attempt"], values["kind"])
                self._send_bytes(200, file_path.read_bytes(), media_type, api=True)
                return
            raise DashboardRequestError(f"Unknown API route: {path}", 404)

        def _handle(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query, keep_blank_values=True)
            try:
                if parsed.path.startswith("/api/"):
                    self._api(parsed.path, query)
                    return
                if query:
                    if parsed.path == "/":
                        self._single_query(query, {"case", "attempt"})
                    else:
                        raise DashboardRequestError("Static assets do not accept query parameters")
                path = static_files.get(parsed.path)
                if path is None:
                    raise DashboardRequestError(f"Unknown dashboard route: {parsed.path}", 404)
                media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                if media_type.startswith("text/") or media_type in {"application/javascript", "application/json"}:
                    media_type += "; charset=utf-8"
                self._send_bytes(200, path.read_bytes(), media_type)
            except DashboardRequestError as exc:
                self._send_json(exc.status, {"error": str(exc), "status": exc.status})
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(500, {"error": "Dashboard data could not be read", "status": 500,
                                      "detail": type(exc).__name__})

        def do_GET(self):
            self._handle()

        def do_HEAD(self):
            self._handle()

    return ThreadingHTTPServer((host, port), Handler)
