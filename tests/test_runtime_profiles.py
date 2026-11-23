"""Profiles select guidance/default tools, never authorize capabilities."""

import importlib.util

import pytest

from adj_partner.contracts import SessionConfig
from adj_partner.storage import Store
from adj_partner.tools import ToolCatalog, ToolSpec
from adj_partner.runtime.journal import RunJournal


def api():
    assert importlib.util.find_spec("adj_partner.profiles"), "shared profiles must exist"
    from adj_partner import profiles
    return profiles


def test_protein_profile_only_enables_current_host_authorized_catalog_subset(tmp_path):
    p = api()
    store, catalog = Store(tmp_path / "state"), ToolCatalog()
    for name in ["protein_sequence", "literature_search", "catalog_search"]:
        catalog.register(ToolSpec(name, "fixture", {"type": "object"}), lambda ctx: None)
    sid = store.create_session(SessionConfig(allowed_tools=["protein_sequence", "catalog_search"],
        instructions="Host custom instruction", allow_network=False).to_dict())["id"]
    result = p.apply_profile(store, catalog, sid, "protein")
    assert result["config"]["profile"] == "protein"
    assert result["config"]["enabled_tools"] == ["catalog_search", "protein_sequence"]
    assert result["config"]["instructions"] == "Host custom instruction"
    assert result["config"]["allow_network"] is False
    assert result["config"]["allowed_tools"] == ["protein_sequence", "catalog_search"]


def test_unknown_profile_and_changes_during_a_run_fail_without_mutation(tmp_path):
    p = api()
    store, catalog = Store(tmp_path / "state"), ToolCatalog()
    session = store.create_session(SessionConfig().to_dict())
    with pytest.raises(ValueError): p.apply_profile(store, catalog, session["id"], "unknown")
    RunJournal(store).start(session["id"], "hi", {}, session["config"]["budget"])
    with pytest.raises(RuntimeError): p.apply_profile(store, catalog, session["id"], "protein")
    assert store.get_session(session["id"])["config"]["profile"] == "general"


def test_profile_definitions_are_copies_and_have_no_tool_implementation():
    p = api()
    analysis = {"code_interpreter", "bash", "analysis_method_register",
                "analysis_method_run", "analysis_history", "analysis_history_read"}
    for profile in ("general", "adjuvant", "protein"):
        assert analysis <= set(p.get_profile(profile)["tools"])
    definition = p.get_profile("adjuvant")
    assert {"deg_analyze", "deg_explore"} <= set(definition["tools"]) and "handler" not in definition
    assert "top_centered" in definition["instructions"]
    assert "immune-module" in definition["instructions"]
    assert "deg_explore" in definition["instructions"]
    assert "web_search" in definition["tools"] and "search snippets" in definition["instructions"]
    definition["tools"].clear()
    assert "deg_analyze" in p.get_profile("adjuvant")["tools"]


def test_explicit_trusted_guidance_file_is_text_only_and_bounded(tmp_path):
    p = api()
    path = tmp_path / "guide.md"
    path.write_text("General scientific guidance; no execution.")
    assert p.load_trusted_guidance(path) == path.read_text()
    alias = tmp_path / "alias.md"
    alias.symlink_to(path)
    with pytest.raises(PermissionError): p.load_trusted_guidance(alias)
    secret = tmp_path / ".env"
    secret.write_text("SYNTHETIC=value")
    with pytest.raises(PermissionError): p.load_trusted_guidance(secret)
    path.write_text("x" * 65537)
    with pytest.raises(ValueError): p.load_trusted_guidance(path)


def test_context_includes_selected_profile_guidance_without_changing_permissions(tmp_path):
    p = api()
    from adj_partner.runtime.context import ContextView
    store = Store(tmp_path / "state")
    sid = store.create_session(SessionConfig(profile="protein").to_dict())["id"]
    rid = RunJournal(store).start(sid, "properties", {}, SessionConfig().budget.to_dict())["id"]
    projection = ContextView(store).build(sid, rid, {"tools": []}, max_input_tokens=10000)
    assert p.get_profile("protein")["instructions"] in projection["messages"][0]["content"]
    assert projection["tools"] == []
