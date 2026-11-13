from core.utils import extract_json


def test_direct_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_block():
    raw = "Here you go:\n```json\n{\"tools\": [\"x\"]}\n```\nDone."
    assert extract_json(raw) == {"tools": ["x"]}


def test_prose_wrapped_object():
    raw = 'Sure! {"depth": "qa", "needs_retrieval": false} that is my answer.'
    assert extract_json(raw) == {"depth": "qa", "needs_retrieval": False}


def test_nested_braces_and_strings():
    raw = 'noise {"a": {"b": "has } brace"}, "c": [1,2]} tail'
    assert extract_json(raw) == {"a": {"b": "has } brace"}, "c": [1, 2]}


def test_non_dict_returns_empty():
    assert extract_json("[1, 2, 3]") == {}


def test_garbage_returns_empty():
    assert extract_json("no json at all") == {}
    assert extract_json("") == {}
