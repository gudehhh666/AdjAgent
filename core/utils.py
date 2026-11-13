"""Shared helpers. extract_json robustly pulls a JSON object out of an LLM
response that may wrap it in prose or code fences."""
from __future__ import annotations

import json
import re
from typing import Any


def extract_json(raw: str) -> dict[str, Any]:
    """Best-effort JSON object extraction. Tries, in order:
    1) direct parse; 2) fenced ```json block; 3) balanced-brace scan.
    Returns {} on failure (callers fall back)."""
    if not raw:
        return {}
    raw = raw.strip()
    # 1) direct
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:  # noqa: BLE001
        pass
    # 2) fenced code block
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:  # noqa: BLE001
            pass
    # 3) balanced-brace scan from first '{'
    start = raw.find("{")
    if start == -1:
        return {}
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:  # noqa: BLE001
                    return {}
    return {}
