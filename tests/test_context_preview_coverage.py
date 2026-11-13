import json

from adj_partner.runtime.context import ContextView


class PreviewStore:
    def __init__(self, *, secret=None):
        self.secret = secret
        self.artifacts = []

    def get_run(self, run_id):
        return {"session_id": "session"}

    def redact(self, value):
        if self.secret is None:
            return value
        if isinstance(value, str):
            return value.replace(self.secret, "[REDACTED]")
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, dict):
            return {key: self.redact(item) for key, item in value.items()}
        return value

    def put_artifact(self, content, name, media_type, provenance, *, session_id, run_id):
        artifact = {"id": "artifact-1", "sha256": "a" * 64}
        self.artifacts.append((content, name, media_type, provenance, session_id, run_id))
        return artifact


def _result(*, nested=False, text=None):
    text = "x" * 600 if text is None else text
    rows = [{"id": i, "text": text} for i in range(5)]
    summary = {
        "results": rows,
        "results_shown": 5,
        "returned": 5,
        "results_omitted_from_summary": 0,
        "detail": "d" * (5000 if nested else 10),
    }
    return {
        "status": "ok",
        "data": {"source_id": "source-1", "job_id": "job-1", "status": "ok", "summary": summary},
        "artifacts": [],
    }


def test_oversized_summary_discloses_visible_and_source_result_counts():
    store = PreviewStore()
    envelope = json.loads(ContextView(store).tool_message("session", "run", "call", _result())["content"])

    assert envelope["context_truncated"] is True
    assert len(envelope["data"]["summary"]["results"]) == 2
    coverage = envelope.get("data", {}).get("preview_coverage", envelope.get("preview_coverage", []))
    row = next(item for item in coverage if item["path"] == "data.summary.results")
    assert row["source_item_count"] == 5
    assert row["visible_item_count"] == 2
    assert row["omitted_item_count"] == 3
    assert row["source_count_fields"] == {
        "results_shown": 5,
        "returned": 5,
        "results_omitted_from_summary": 0,
    }


def test_summary_fallback_preserves_true_counts_when_nested_summary_is_too_large():
    store = PreviewStore()
    envelope = json.loads(ContextView(store).tool_message("session", "run", "call", _result(nested=True))["content"])

    assert envelope["context_truncated"] is True
    coverage = envelope.get("data", {}).get("preview_coverage", envelope.get("preview_coverage", []))
    row = next(item for item in coverage if item["path"] == "data.summary.results")
    assert row["source_item_count"] == 5
    assert row["visible_item_count"] in {0, 1, 2}
    assert row["omitted_item_count"] == 5 - row["visible_item_count"]
    assert row["source_count_fields"]["returned"] == 5


def test_small_result_preview_is_unchanged():
    store = PreviewStore()
    result = _result(text="small")
    result["data"]["summary"]["results"] = [{"id": 1, "text": "small"}]
    result["data"]["summary"]["detail"] = "small"
    message = ContextView(store).tool_message("session", "run", "call", result)

    assert json.loads(message["content"]) == result
    assert "context_truncated" not in message["content"]


def test_preview_coverage_is_redacted_without_secret_leak():
    secret = "TOKEN_SECRET_123"
    store = PreviewStore(secret=secret)
    result = _result(nested=True, text=secret)
    envelope = json.loads(ContextView(store).tool_message("session", "run", "call", result)["content"])

    assert secret not in json.dumps(envelope)
    assert secret not in store.artifacts[0][0]


def test_adversarial_unicode_keys_keep_fallback_within_budget_and_mark_omission():
    store = PreviewStore()
    result = _result(nested=True)
    result["data"]["summary"] = {
        ("列" * 5000): {"results": [{"value": "x"}]},
        "detail": "d" * 6000,
    }

    message = ContextView(store).tool_message("session", "run", "call", result)
    encoded = message["content"].encode("utf-8")
    envelope = json.loads(message["content"])

    assert len(encoded) <= ContextView.PREVIEW_MAX_BYTES
    assert envelope["metadata_truncated"] is True
    assert all(len(json.dumps(item, ensure_ascii=False).encode("utf-8")) < 500
               for item in envelope.get("preview_coverage", []))


def test_oversized_keys_are_omitted_without_prefix_alias_collisions():
    prefix = "same-prefix-" * 30
    metadata = {}
    compact = ContextView._compact_value(
        {prefix + "-one": 1, prefix + "-two": 2}, metadata=metadata)

    assert compact == {}
    assert metadata["truncated"] is True


def test_coverage_paths_disambiguate_dotted_and_bracket_keys():
    source = {
        "a.b": [{"value": 1}],
        "a": {"b": [{"value": 2}]},
        "a[0]": [{"value": 3}],
    }
    original = json.loads(json.dumps(source))
    coverage = ContextView._preview_list_coverage(source, source, path="data")
    paths = {item["path"] for item in coverage}

    assert "/data/a.b" in paths
    assert "data.a.b" in paths
    assert "/data/a[0]" in paths
    assert source == original
