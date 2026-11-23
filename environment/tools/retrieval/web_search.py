"""Generic web search. Requires a provider API key (set in .env). Degrades to a
clear 'not configured' record when no key is present, so the pipeline still runs.

Supported providers: tavily | serper | serper_gateway (AI Hub gateway).
"""
from __future__ import annotations

from typing import Any

import requests

from core.config import env, load_yaml
from core.schemas import EvidenceRecord
from environment.registry import REGISTRY, ToolSpec

_UA = {"User-Agent": "Adj_Partner/0.1"}


def _providers() -> list[str]:
    raw = (env("ADJ_WEB_SEARCH_PROVIDER") or "").lower()
    return [name.strip() for name in raw.split(",") if name.strip()]


def _key(provider: str) -> str:
    specific = {"serper_gateway": "ADJ_SERPER_GATEWAY_APP_KEY", "serper": "ADJ_SERPER_API_KEY",
                "tavily": "ADJ_TAVILY_API_KEY"}.get(provider)
    if provider == "serper_gateway":
        return env(specific) or ""
    return (env(specific) if specific else None) or env("ADJ_WEB_SEARCH_API_KEY") or ""


def _enabled() -> bool:
    node = load_yaml("sources.yaml").get("web_search", {})
    return bool(node.get("enabled", False)) and any(_key(p) for p in _providers())


def _serper_gateway(query: str, max_results: int) -> list[EvidenceRecord]:
    base = (env("ADJ_AIHUB_GATEWAY_URL") or "").rstrip("/")
    if not base:
        raise ValueError("serper_gateway needs ADJ_AIHUB_GATEWAY_URL")
    auth = (f"Bearer {env('ADJ_SERPER_GATEWAY_APP_ID')}:{env('ADJ_SERPER_GATEWAY_APP_KEY')}"
            "?provider=serper&timeout=60")
    resp = requests.post(f"{base}/search", json={"q": query, "num": max_results},
                         headers={**_UA, "Authorization": auth}, timeout=70)
    resp.raise_for_status()
    out = []
    for r in resp.json().get("organic", [])[:max_results]:
        out.append(EvidenceRecord(
            claim=r.get("title", ""), source_type="web",
            source_id=r.get("link", ""), snippet=(r.get("snippet", ""))[:400],
            authority_weight=0.5))
    return out


def _tavily(query: str, key: str, max_results: int) -> list[EvidenceRecord]:
    resp = requests.post("https://api.tavily.com/search",
                         json={"api_key": key, "query": query,
                               "max_results": max_results},
                         headers=_UA, timeout=25)
    resp.raise_for_status()
    out = []
    for r in resp.json().get("results", []):
        out.append(EvidenceRecord(
            claim=r.get("title", ""), source_type="web",
            source_id=r.get("url", ""), snippet=(r.get("content", ""))[:400],
            authority_weight=0.5, extra={"score": r.get("score")}))
    return out


def _serper(query: str, key: str, max_results: int) -> list[EvidenceRecord]:
    resp = requests.post("https://google.serper.dev/search",
                         json={"q": query, "num": max_results},
                         headers={**_UA, "X-API-KEY": key}, timeout=25)
    resp.raise_for_status()
    out = []
    for r in resp.json().get("organic", [])[:max_results]:
        out.append(EvidenceRecord(
            claim=r.get("title", ""), source_type="web",
            source_id=r.get("link", ""), snippet=(r.get("snippet", ""))[:400],
            authority_weight=0.5))
    return out


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    if not _enabled():
        return [EvidenceRecord(
            claim="[web_search not configured] set ADJ_WEB_SEARCH_PROVIDER/API_KEY in .env",
            source_type="web", source_id="not_configured",
            authority_weight=0.0).to_dict()]
    errors = []
    for provider in _providers():
        key = _key(provider)
        if not key:
            continue
        try:
            if provider == "serper_gateway":
                recs = _serper_gateway(query, max_results)
            elif provider == "tavily":
                recs = _tavily(query, key, max_results)
            elif provider == "serper":
                recs = _serper(query, key, max_results)
            else:
                errors.append(f"unsupported provider {provider}")
                continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{provider}: {type(exc).__name__}")
            continue
        return [r.to_dict() for r in recs]
    return [EvidenceRecord(claim=f"[web_search error] {'; '.join(errors)}",
            source_type="web", source_id="error", authority_weight=0.0).to_dict()]


REGISTRY.register(
    ToolSpec(
        name="search_web",
        description="Generic web search (requires provider API key). Returns "
                    "EvidenceRecord dicts; emits a 'not configured' record if no key.",
        domain="retrieval",
        parameters={"query": "search string", "max_results": "cap (int)"},
        returns="list[EvidenceRecord as dict]",
    ),
    search_web,
)
